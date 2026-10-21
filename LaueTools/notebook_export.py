"""
Export Jupyter notebooks to HTML and PDF with their saved outputs (e.g. when the export menu of JupyterLab does
not work: no LaTeX, no headless browser).

    lauetools-export-notebook 2_postprocess_maps.local.ipynb            # HTML and PDF next to the notebook
    python -m LaueTools.notebook_export a.ipynb b.ipynb --to pdf --outdir reports
    python -m LaueTools.notebook_export a.ipynb --execute                # run the notebook first (copy)

From the last cell of a notebook (save the notebook first: the export reads the .ipynb file):

    from LaueTools.notebook_export import export_this_notebook
    export_this_notebook()                     # or export_this_notebook(to='pdf', outdir='reports')

Dependencies: nbconvert, nbformat (pip install "lauetools[notebook]"); PDF: one of the methods below

PDF methods ('auto': the first available): 'webpdf' (nbconvert with a headless Chromium, package playwright),
'latex' (nbconvert with xelatex), 'libreoffice' (soffice --headless: conversion of the HTML export, where the
images are first resized to fit the page width, otherwise wide figures go outside the page)
"""
import argparse
import base64
import datetime
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

PDF_METHODS = ('webpdf', 'latex', 'libreoffice')


def read_notebook(path, execute: bool = False, timeout: Optional[int] = None):
    """notebook node of path; execute: run all its cells first (in the folder of the notebook, file unchanged)"""
    import nbformat
    nb = nbformat.read(str(path), as_version=4)
    if execute:
        from nbconvert.preprocessors import ExecutePreprocessor
        ExecutePreprocessor(timeout=timeout).preprocess(nb, {'metadata': {'path': str(Path(path).parent)}})
    return nb


def export_html(nb, output: Path) -> Path:
    """HTML file of a notebook node"""
    from nbconvert import HTMLExporter
    body, _ = HTMLExporter().from_notebook_node(nb)
    output = Path(output)
    output.write_text(body, encoding='utf-8')
    return output


def fit_images(html: str, maxwidth: int = 640) -> str:
    """HTML where every embedded image (png, jpeg) has width and height attributes with a width of at most
    maxwidth pixels (A4 page with margins: ~640 px)"""
    import matplotlib.image as mpimg

    def resize(match):
        tag = match.group(0)
        data = re.search(r'src="data:image/(png|jpeg);base64,([^"]+)"', tag)
        if not data:
            return tag
        try:
            height, width = mpimg.imread(io.BytesIO(base64.b64decode(data.group(2))), format=data.group(1)).shape[:2]
        except Exception:   # image not readable: unchanged
            return tag
        scale = min(1., maxwidth / width)
        tag = re.sub(r'\s(width|height)="[^"]*"', '', tag)
        return tag.replace('<img', f'<img width="{int(width * scale)}" height="{int(height * scale)}"', 1)

    return re.sub(r'<img[^>]*>', resize, html)


def available_pdf_methods() -> List[str]:
    methods = []
    try:
        import playwright   # noqa: F401
        methods.append('webpdf')
    except ImportError:
        pass
    if shutil.which('xelatex'):
        methods.append('latex')
    if shutil.which('soffice') or shutil.which('libreoffice'):
        methods.append('libreoffice')
    return methods


def _pdf_libreoffice(html: str, output: Path, maxwidth: int) -> Path:
    soffice = shutil.which('soffice') or shutil.which('libreoffice')
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source = tmp / (output.stem + '.html')
        source.write_text(fit_images(html, maxwidth), encoding='utf-8')
        # own profile folder: works while another LibreOffice is open
        cmd = [soffice, '--headless', f'-env:UserInstallation={(tmp / "profile").as_uri()}',
               '--convert-to', 'pdf:writer_web_pdf_Export', '--outdir', str(tmp), str(source)]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
        produced = tmp / (output.stem + '.pdf')
        if not produced.exists():
            raise RuntimeError(f'LibreOffice did not write {produced.name}')
        shutil.move(str(produced), str(output))
    return output


def export_pdf(nb, output: Path, method: str = 'auto', maxwidth: int = 640, html: Optional[str] = None) -> Path:
    """PDF file of a notebook node with method 'auto', 'webpdf', 'latex' or 'libreoffice' (see module docstring).
    html: HTML export of nb (libreoffice; default: exported here)"""
    output = Path(output)
    available = available_pdf_methods()
    if method == 'auto':
        if not available:
            raise RuntimeError('no PDF method available: install playwright (pip install "nbconvert[webpdf]" and '
                               'playwright install chromium), xelatex (TeX Live) or LibreOffice')
        method = available[0]
    elif method not in available:
        raise RuntimeError(f"PDF method {method} not available here (available: {available or 'none'})")
    if method == 'libreoffice':
        if html is None:
            from nbconvert import HTMLExporter
            html, _ = HTMLExporter().from_notebook_node(nb)
        return _pdf_libreoffice(html, output, maxwidth)
    if method == 'webpdf':
        from nbconvert import WebPDFExporter
        body, _ = WebPDFExporter().from_notebook_node(nb)
    else:
        from nbconvert import PDFExporter
        body, _ = PDFExporter().from_notebook_node(nb)
    output.write_bytes(body)
    return output


def export_notebook(path, to=('html', 'pdf'), outdir=None, method: str = 'auto', maxwidth: int = 640,
                    execute: bool = False, timeout: Optional[int] = None) -> List[Path]:
    """export the notebook path to the formats `to` ('html', 'pdf') in outdir (default: folder of the notebook),
    files <notebook name>.html / .pdf. Returns the files written"""
    path = Path(path)
    outdir = Path(outdir) if outdir else path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    nb = read_notebook(path, execute=execute, timeout=timeout)
    written, html = [], None
    if 'html' in to or ('pdf' in to and method in ('auto', 'libreoffice')):
        from nbconvert import HTMLExporter
        html, _ = HTMLExporter().from_notebook_node(nb)
    if 'html' in to:
        target = outdir / (path.stem + '.html')
        target.write_text(html, encoding='utf-8')
        written.append(target)
    if 'pdf' in to:
        written.append(export_pdf(nb, outdir / (path.stem + '.pdf'), method=method, maxwidth=maxwidth, html=html))
    return written


def current_notebook_path() -> Path:
    """path of the .ipynb file of the running kernel (VS Code, JupyterLab, Jupyter Notebook)"""
    try:
        from IPython import get_ipython
        shell = get_ipython()
    except ImportError:
        shell = None
    if shell is None:
        raise RuntimeError('not running in a notebook: give the path of the .ipynb file')
    # VS Code
    vscode = shell.user_ns.get('__vsc_ipynb_file__')
    if vscode and Path(vscode).is_file():
        return Path(vscode)
    # jupyter_server >= 2: path (or name) of the notebook of the kernel session
    session = os.environ.get('JPY_SESSION_NAME')
    if session and session.endswith('.ipynb'):
        for candidate in (Path(session), Path.cwd() / Path(session).name):
            if candidate.is_file():
                return candidate.resolve()
    # sessions of the running Jupyter servers: notebook whose kernel is this one
    try:
        import ipykernel
        kernel_id = re.search(r'kernel-(.+)\.json', Path(ipykernel.get_connection_file()).name).group(1)
    except Exception:
        kernel_id = None
    if kernel_id:
        for server in _running_servers():
            try:
                url = server['url'].rstrip('/') + '/api/sessions'
                request = urllib.request.Request(url, headers={'Authorization': f"token {server.get('token', '')}"})
                with urllib.request.urlopen(request, timeout=3) as response:
                    sessions = json.load(response)
            except Exception:
                continue
            for item in sessions:
                if item.get('kernel', {}).get('id') == kernel_id:
                    relpath = item.get('path') or item.get('notebook', {}).get('path', '')
                    root = server.get('root_dir') or server.get('notebook_dir') or ''
                    candidate = Path(root, relpath)
                    if candidate.is_file():
                        return candidate
    raise RuntimeError('path of the notebook not found: export_this_notebook(path="<name>.ipynb")')


def _running_servers():
    for module in ('jupyter_server.serverapp', 'notebook.notebookapp'):
        try:
            yield from __import__(module, fromlist=['list_running_servers']).list_running_servers()
        except Exception:
            pass


def export_this_notebook(to=('html', 'pdf'), outdir=None, method: str = 'auto', maxwidth: int = 640,
                         path=None) -> List[Path]:
    """export the running notebook (its saved .ipynb file) to HTML and/or PDF, e.g. in its last cell.
    The outputs are those of the last save: save the notebook (Ctrl+S) before running this cell.
    path: the .ipynb file (default: found from the kernel)"""
    path = Path(path) if path else current_notebook_path()
    saved = path.stat().st_mtime
    age = time.time() - saved
    print(f'{path.name}: saved at {datetime.datetime.fromtimestamp(saved):%H:%M:%S} ({age / 60:.0f} min ago)')
    if age > 120:
        print('  outputs of the cells run since then are not exported: save the notebook and run this cell again')
    if isinstance(to, str):
        to = (to,)
    written = export_notebook(path, to=to, outdir=outdir, method=method, maxwidth=maxwidth)
    for f in written:
        print(f'written: {f}')
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(prog='lauetools-export-notebook',
                                     description='Export Jupyter notebooks (saved outputs) to HTML and PDF',
                                     epilog=f'PDF methods available here: {", ".join(available_pdf_methods()) or "none"}')
    parser.add_argument('notebooks', nargs='+', help='.ipynb files')
    parser.add_argument('--to', nargs='+', choices=('html', 'pdf'), default=['html', 'pdf'],
                        help='formats (default: html pdf)')
    parser.add_argument('--outdir', default=None, help='output folder (default: folder of each notebook)')
    parser.add_argument('--method', choices=('auto',) + PDF_METHODS, default='auto',
                        help='PDF method (default auto: the first available of webpdf, latex, libreoffice)')
    parser.add_argument('--maxwidth', type=int, default=640,
                        help='libreoffice: max width (pixels) of the images in the PDF (default 640: A4 width)')
    parser.add_argument('--execute', action='store_true',
                        help='run the notebook before exporting (the .ipynb file is not changed)')
    parser.add_argument('--timeout', type=int, default=None, help='--execute: max time (s) per cell')
    args = parser.parse_args(argv)
    status = 0
    for notebook in args.notebooks:
        try:
            files = export_notebook(notebook, to=args.to, outdir=args.outdir, method=args.method,
                                    maxwidth=args.maxwidth, execute=args.execute, timeout=args.timeout)
            for f in files:
                print(f'written: {f}')
        except Exception as error:
            print(f'{notebook}: export failed: {error}', file=sys.stderr)
            status = 1
    return status


if __name__ == '__main__':
    sys.exit(main())
