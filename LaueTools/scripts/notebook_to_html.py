# -*- coding: utf-8 -*-
"""
Module of LaueTools project: HTML version of notebooks, optionally anonymized

Without --execute, the saved outputs are used (nothing is re-executed):

- ipympl figures (%matplotlib widget): the PNG snapshot saved with the widget is kept
  (the widget state is not saved in the notebook, so the widget view itself cannot be rendered)
- ipywidgets cluster browsers of notebooks/indexation/3_segmentation_grains.ipynb
  (OC.visualize_cluster_interactive, OC.cluster_grod_interactive): replaced by a static PNG of their
  initial view (first cluster, default settings), computed again from the .fit files (option --segmentation)
- other widgets without image (e.g. tqdm progress bars): replaced by a short text

With --execute, a temporary copy of the notebook is executed (in the folder of the notebook, errors do not
stop the execution) with static matplotlib figures (inline backend instead of ipympl) and cluster browsers
drawn as a static view of their first cluster.

With --anonymize [RULES.yaml], proposal ids, dates, sample and dataset names, machine names and local paths
are replaced in the code, the text outputs and (with --execute) in all texts drawn in the figures
(titles, labels...). Images of saved outputs (no --execute) cannot be anonymized: they are listed to be
checked by eye. Built-in rules (see DEFAULT_PRE_PATTERNS, DEFAULT_PATTERNS) and names harvested from the /data/visitor/ paths
found in the notebook and in the files given as parameters (e.g. the YAML configuration) can be completed
by a YAML file::

    names:              # literal strings replaced first (longest first)
      MH73: sample_A
      MgO_Fe_800: map_1
    patterns:           # python regular expressions, applied after names and before built-in patterns
      'myhost\\d+': '<host>'
    defaults: true      # built-in patterns
    harvest: true       # names harvested from /data/visitor/ paths

Typical use (in notebooks/indexation/)::

    python -m LaueTools.scripts.notebook_to_html 1_index_refine.ipynb --anonymize
    python -m LaueTools.scripts.notebook_to_html 2_postprocess_maps.ipynb 3_segmentation_grains.ipynb \\
        --anonymize --execute -p CONFIG=configs/a321220_MgO.yaml

the .html files are written next to the notebooks (or in --output-dir)

With --strip, the notebook itself is written anonymized and without outputs (version to be committed)
instead of the HTML. In the folder of the notebook, the original notebook (real paths, outputs) is
kept as <name>.local.ipynb (ignored by git), e.g.::

    python -m LaueTools.scripts.notebook_to_html MyNotebook.ipynb --anonymize my_rules.local.yaml           # MyNotebook.html
    python -m LaueTools.scripts.notebook_to_html MyNotebook.ipynb --anonymize my_rules.local.yaml --strip   # MyNotebook.ipynb

HTML and stripped notebook of <name>.local.ipynb are named <name>.html and <name>.ipynb.

With --publish, both versions to be committed are written in one step from the working notebook (with outputs):
<name>.html (anonymized, with outputs) and <name>.ipynb (anonymized, without outputs; --keep-outputs: with
anonymized text outputs). The working notebook is kept as <name>.local.ipynb (ignored by git): continue to work
in <name>.local.ipynb, and publish it again. Without RULES.yaml, the first anonymize_rules.local.yaml found in
the folder of the notebook or in its parent folders is used::

    python -m LaueTools.scripts.notebook_to_html --publish quickstart/laue_maps_quickstart.ipynb
    python -m LaueTools.scripts.notebook_to_html --publish quickstart/laue_maps_quickstart.local.ipynb   # later
"""
import base64
import copy
import io
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

WIDGET = 'application/vnd.jupyter.widget-view+json'
SNAPSHOT_CAPTION = '[interactive browser (ipywidgets): static view of the first cluster with default settings]'
SETUP_TAG = 'notebook_to_html-setup'
PARAMETERS_TAG = 'injected-parameters'

_NOQUOTE = r'[^/\s\'"<>,;:()\[\]{}]'   # character of a path component
# (regex, replacement) built-in rules: local paths before the literal names, the others after
DEFAULT_PRE_PATTERNS: List[Tuple[str, str]] = [
    (r'(?:/gpfs/\w+|/mnt/multipath-shares)(?=/data/)', ''),         # mount points -> /data/...
    (rf'/data/(?!visitor/)\w+/inhouse(?:/{_NOQUOTE}+)*', '<inhouse path>'),
    (rf'/home/(?!<){_NOQUOTE}+', '/home/<user>'),
]
DEFAULT_PATTERNS: List[Tuple[str, str]] = [
    (rf'/data/visitor/{_NOQUOTE}+/({_NOQUOTE}+)/\d{{8}}', r'/data/visitor/<proposal>/\1/<date>'),
    (rf'((?:RAW|PROCESSED)_DATA)/(?!<){_NOQUOTE}+', r'\1/<sample>'),
    (rf'((?:RAW|PROCESSED)_DATA/<sample>)/(?!<){_NOQUOTE}+', r'\1/<dataset>'),
    (r'\bfrcrg-[\w-]+', '<node>'),                                     # ESRF cluster nodes
    (r'\blbm32\w*', '<host>'),                                         # BM32 computers
]
HARVEST_REGEX = re.compile(rf'/data/visitor/(?P<proposal>{_NOQUOTE}+)/{_NOQUOTE}+/(?P<date>\d{{8}})'
                           rf'(?:/(?:RAW|PROCESSED)_DATA/(?P<sample>{_NOQUOTE}+)(?:/(?P<dataset>{_NOQUOTE}+))?)?')
MIN_HARVESTED_LENGTH = 4   # shorter harvested names (e.g. 'Si') are not replaced everywhere


# --------------------------------------------------------------------------------------
#  ANONYMIZATION RULES
# --------------------------------------------------------------------------------------
def harvest_names(texts) -> Dict[str, str]:
    """{name: placeholder} for proposal ids, dates, sample and dataset names of /data/visitor/ paths in texts"""
    names = {}
    for text in texts:
        for match in HARVEST_REGEX.finditer(text):
            for group in ('dataset', 'sample', 'proposal', 'date'):
                value = match.group(group)
                if value and len(value) >= MIN_HARVESTED_LENGTH and not value.startswith('<'):
                    names.setdefault(value, f'<{group}>')
    return names


def load_rules(rulesfile: Optional[Union[str, Path]] = None, texts=()) -> List[Tuple[str, str]]:
    """list of (regex, replacement): literal names (rules file and harvested from texts), then patterns"""
    user = {}
    if rulesfile:
        import yaml
        user = yaml.safe_load(Path(rulesfile).read_text(encoding='utf-8')) or {}
    names = harvest_names(texts) if user.get('harvest', True) else {}
    names.update({str(key): str(val) for key, val in (user.get('names') or {}).items()})
    defaults = user.get('defaults', True)
    rules = list(DEFAULT_PRE_PATTERNS) if defaults else []
    rules += [(re.escape(name), names[name].replace('\\', r'\\')) for name in sorted(names, key=len, reverse=True)]
    rules += [(str(key), str(val)) for key, val in (user.get('patterns') or {}).items()]
    if defaults:
        rules += DEFAULT_PATTERNS
    return rules


def make_anonymizer(rules: List[Tuple[str, str]]) -> Callable[[str], str]:
    compiled = [(re.compile(regex), repl) for regex, repl in rules]

    def anonymize(text: str) -> str:
        for regex, repl in compiled:
            text = regex.sub(repl, text)
        return text
    return anonymize


def anonymize_notebook(nb: dict, anonymize: Callable[[str], str]) -> dict:
    """apply anonymize() to cell sources and to all text outputs (images are not modified)"""
    def apply(value):
        if isinstance(value, list):
            return anonymize(''.join(value))
        return anonymize(value) if isinstance(value, str) else value

    for cell in nb['cells']:
        cell['source'] = apply(cell['source'])
        for out in cell.get('outputs', []):
            if 'text' in out:
                out['text'] = apply(out['text'])
            if 'evalue' in out:
                out['evalue'] = apply(out['evalue'])
            if 'traceback' in out:
                out['traceback'] = [apply(line) for line in out['traceback']]
            for key, value in out.get('data', {}).items():
                if key.startswith('text/') or key in ('application/javascript',):
                    out['data'][key] = apply(value)
    return nb


def notebook_texts(nb: dict) -> List[str]:
    """all texts of a notebook (sources and text outputs)"""
    texts = []
    for cell in nb['cells']:
        texts.append(''.join(cell['source']))
        for out in cell.get('outputs', []):
            for value in [out.get('text', ''), out.get('evalue', '')] + list(out.get('traceback', [])) + \
                         [val for key, val in out.get('data', {}).items() if key.startswith('text/')]:
                texts.append(''.join(value) if isinstance(value, list) else str(value))
    return texts


# --------------------------------------------------------------------------------------
#  STATIC FIGURES (in this process or in the kernel executing the notebook)
# --------------------------------------------------------------------------------------
def _figure_png(fig, dpi: int = 90) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    return buf.getvalue()


def _display_png(png: bytes):
    from IPython.display import Image, display
    print(SNAPSHOT_CAPTION)
    display(Image(data=png, format='png'))


def install_static_browsers(callback: Callable[[bytes], None] = _display_png):
    """cluster browsers of orientationclustering (_cluster_browser) draw their first cluster (default values
    of their widgets) in a PNG image given to callback(png) instead of displaying ipywidgets"""
    import LaueTools.orientationclustering as OC

    def static_browser(result, analyzer, draw, min_size=1, figsize=(7, 6), extra_widgets=None, rows=12,
                       live=False, dpi=90):
        valid = [c for c in result.clusters if c.size >= min_size]
        if not valid:
            print(f"No clusters with size >= {min_size}")
            return
        fig = OC._offscreen_figure(figsize)
        draw(fig, valid[0].cluster_id, **{name: w.value for name, w in (extra_widgets or {}).items()})
        callback(_figure_png(fig, dpi))

    OC._cluster_browser = static_browser


def install_text_anonymizer(rules: List[Tuple[str, str]]):
    """all matplotlib texts (titles, labels, annotations...) are anonymized when drawn"""
    from matplotlib.text import Text
    anonymize = make_anonymizer(rules)
    draw = getattr(Text.draw, '_notebook_to_html_original', Text.draw)

    def anonymized_draw(self, renderer):
        text = self.get_text()
        new = anonymize(text)
        if new != text:
            self.set_text(new)
        return draw(self, renderer)

    anonymized_draw._notebook_to_html_original = draw
    Text.draw = anonymized_draw


def install_static_mode(rules: Optional[List[Tuple[str, str]]] = None, browsers: bool = True):
    """called in the kernel executing the notebook (first hidden cell):
    ipympl import fails (notebooks fall back to %matplotlib inline), static cluster browsers,
    anonymized texts in figures if rules"""
    import sys
    sys.modules['ipympl'] = None   # 'import ipympl' raises ImportError
    os.environ['LAUETOOLS_ANONYMOUS'] = '1' if rules else '0'
    if rules:
        install_text_anonymizer(rules)
    if browsers:
        install_static_browsers()


def segmentation_snapshots(configfile: Union[str, Path], fit_folder: Optional[Union[str, Path]] = None,
                           threshold: float = 0.8, min_size: int = 15,
                           rules: Optional[List[Tuple[str, str]]] = None) -> Dict[str, bytes]:
    """PNG of the initial view of the 2 cluster browsers of 3_segmentation_grains.ipynb (saved outputs mode)

    same clustering as the notebook (threshold in deg, min_size = MIN_CLUSTER_SIZE).
    fit_folder: None: output fit_folder of the configuration file
    return {function name called in the notebook cell: png bytes}
    """
    import matplotlib
    matplotlib.use('Agg')
    import LaueTools.indexing_batch as IB
    import LaueTools.orientationclustering as OC

    if rules:
        install_text_anonymizer(rules)
    cfg = IB.load_config(configfile)
    scan, post = cfg['scan'], cfg['postprocess']
    layout = IB.fitfiles_layout(cfg)
    result, analyzer = OC.analyze_ub_matrices(input_dir=str(fit_folder or cfg['output']['fit_folder']),
                                              threshold=threshold, mapdimension=scan['invmapdims'],
                                              prefix=scan['prefix'], symmetry=post['symmetry'],
                                              spatial_connectivity=True, connectivity_type='8',
                                              min_cluster_size=min_size, check_symmetry=False,
                                              nbfiles_per_folder=layout['fit_nbfiles_per_folder'],
                                              subfolder_prefix=layout['subfolder_prefix'])
    pngs = []
    install_static_browsers(pngs.append)
    snapshots = {}
    for name, func in (('visualize_cluster_interactive', OC.visualize_cluster_interactive),
                       ('cluster_grod_interactive', OC.cluster_grod_interactive)):
        # arguments of the calls in the notebook
        func(result, analyzer, min_size=min_size, zoom=False, **({'properties': True} if 'visualize' in name
                                                                 else {'show_envelope': True}))
        if pngs:
            snapshots[name] = pngs.pop()
    return snapshots


# --------------------------------------------------------------------------------------
#  NOTEBOOK PROCESSING
# --------------------------------------------------------------------------------------
def _png_outputs(png: bytes, caption: str) -> list:
    return [{'output_type': 'stream', 'name': 'stdout', 'text': caption + '\n'},
            {'output_type': 'display_data', 'metadata': {},
             'data': {'image/png': base64.b64encode(png).decode(), 'text/plain': ['<Figure>']}}]


def _widget_caption(data: dict) -> str:
    """text replacing a widget output without saved image"""
    text = data.get('text/plain', '')
    text = ''.join(text) if isinstance(text, list) else str(text)
    if text.startswith('Canvas('):
        return '[interactive figure (ipympl): image not saved in the notebook]\n'
    if 'Progress' in text or text.startswith('HBox(children=(HTML'):
        return '[progress bar]\n'
    return '[interactive widget (ipywidgets)]\n'


def static_outputs(nb: dict, snapshots: Optional[Dict[str, bytes]] = None) -> dict:
    """copy of notebook nb (json dict) with widget outputs replaced by static ones

    snapshots: {string found in the cell source (e.g. function name): png bytes}
    """
    nb = copy.deepcopy(nb)
    nb['metadata'].pop('widgets', None)
    snapshots = snapshots or {}
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        outputs = []
        for out in cell.get('outputs', []):
            data = out.get('data', {})
            if WIDGET not in data:
                outputs.append(out)
            elif 'image/png' in data:   # ipympl figure: PNG saved with the notebook
                for key in (WIDGET, 'text/html'):
                    data.pop(key, None)
                outputs.append(out)
            else:
                key = next((key for key in snapshots if key in source), None)
                if key:
                    outputs += _png_outputs(snapshots[key], SNAPSHOT_CAPTION)
                else:
                    outputs.append({'output_type': 'stream', 'name': 'stdout', 'text': _widget_caption(data)})
        cell['outputs'] = outputs
    return nb


def mask_images(nb: dict, masks: Dict[int, List[Tuple[int, int]]]) -> dict:
    """copy of nb where rows y0 to y1 (included) of PNG outputs of cells are painted white,
    e.g. to hide a non anonymized path in a figure title of saved outputs

    masks: {cell index (in the notebook, first cell 0): [(y0, y1), ...]}
    """
    import matplotlib.image as mpimg
    nb = copy.deepcopy(nb)
    for index, bands in masks.items():
        for out in nb['cells'][index].get('outputs', []):
            data = out.get('data', {})
            if 'image/png' not in data:
                continue
            png = data['image/png']
            image = mpimg.imread(io.BytesIO(base64.b64decode(''.join(png) if isinstance(png, list) else png)),
                                 format='png').copy()
            for y0, y1 in bands:
                image[y0:y1 + 1] = 1.   # white (and opaque for RGBA)
            buf = io.BytesIO()
            mpimg.imsave(buf, image, format='png')
            data['image/png'] = base64.b64encode(buf.getvalue()).decode()
    return nb


def parse_masks(items: List[str]) -> Dict[int, List[Tuple[int, int]]]:
    """['96:52-73', '118:8-22'] -> {96: [(52, 73)], 118: [(8, 22)]}"""
    masks: Dict[int, List[Tuple[int, int]]] = {}
    for item in items:
        cell, _, rows = item.partition(':')
        y0, _, y1 = rows.partition('-')
        masks.setdefault(int(cell), []).append((int(y0), int(y1)))
    return masks


def public_stem(nbfile: Union[str, Path]) -> str:
    """name of notebook without '.local' (private copy with real paths and outputs)"""
    stem = Path(nbfile).stem
    return stem[:-len('.local')] if stem.endswith('.local') else stem


def strip_outputs(nb: dict) -> dict:
    """copy of notebook without outputs, execution counts and widgets state"""
    nb = copy.deepcopy(nb)
    nb['metadata'].pop('widgets', None)
    for cell in nb['cells']:
        cell['metadata'].pop('execution', None)
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    return nb


def strip_notebook(nbfile: Union[str, Path], output_dir: Optional[Union[str, Path]] = None,
                   anonymize: bool = False, rulesfile: Optional[Union[str, Path]] = None) -> Path:
    """write <name>.ipynb without outputs (anonymized if anonymize) in output_dir (default: folder of the
    notebook). If it would overwrite the notebook, the notebook is first renamed <name>.local.ipynb"""
    nbfile = Path(nbfile).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else nbfile.parent
    nb = json.loads(nbfile.read_text(encoding='utf-8'))
    outfile = output_dir / f'{public_stem(nbfile)}.ipynb'
    if outfile == nbfile:
        localfile = nbfile.with_name(f'{nbfile.stem}.local.ipynb')
        if localfile.exists():
            raise FileExistsError(f'{localfile} exists: strip {localfile.name} instead of {nbfile.name}')
        nbfile.rename(localfile)
        print(f'original notebook kept as {localfile}')
    nb = strip_outputs(nb)
    if anonymize:
        nb = anonymize_notebook(nb, make_anonymizer(load_rules(rulesfile, notebook_texts(nb))))
    outfile.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return outfile


def has_outputs(nb: dict) -> bool:
    return any(cell.get('outputs') for cell in nb['cells'])


def default_rulesfile(nbfile: Union[str, Path], name: str = 'anonymize_rules.local.yaml',
                      levels: int = 3) -> Optional[Path]:
    """first file <name> found in the folder of nbfile or in its parent folders (levels at most)"""
    folder = Path(nbfile).resolve().parent
    for candidate in [folder] + list(folder.parents)[:levels]:
        if (candidate / name).is_file():
            return candidate / name
    return None


def publish_notebook(nbfile: Union[str, Path], rulesfile: Optional[Union[str, Path]] = None,
                     keep_outputs: bool = False, snapshots: Optional[Dict[str, bytes]] = None,
                     masks: Optional[Dict[int, List[Tuple[int, int]]]] = None) -> Tuple[Path, Path]:
    """versions of a notebook to be committed, written next to it from the working notebook (with outputs):
    <name>.html (anonymized, with outputs) and <name>.ipynb (anonymized, without outputs unless keep_outputs)

    The working notebook (real paths, outputs) is kept as <name>.local.ipynb (ignored by git). nbfile may be
    <name>.ipynb (first time) or <name>.local.ipynb. If <name>.local.ipynb exists, it is the working notebook,
    unless <name>.ipynb has outputs too (both edited: error, the other one must be renamed first).
    """
    nbfile = Path(nbfile).resolve()
    stem = public_stem(nbfile)
    public = nbfile.with_name(f'{stem}.ipynb')
    localfile = nbfile.with_name(f'{stem}.local.ipynb')
    if nbfile == public and localfile.exists():
        if has_outputs(json.loads(public.read_text(encoding='utf-8'))):
            raise FileExistsError(f'{localfile.name} exists and {public.name} has outputs: keep the one with '
                                  f'your latest work as {localfile.name} (rename or delete the other one)')
        nbfile = localfile
    if rulesfile is None:
        rulesfile = default_rulesfile(nbfile)
    print(f'working notebook: {nbfile.name}, anonymization rules: {rulesfile or "built-in only"}')

    html = notebook_to_html(nbfile, None, snapshots, anonymize=True, rulesfile=rulesfile, masks=masks)

    nb = json.loads(nbfile.read_text(encoding='utf-8'))
    rules = load_rules(rulesfile, notebook_texts(nb))   # names harvested also from the outputs, as for the HTML
    if not keep_outputs:
        nb = strip_outputs(nb)
    elif masks:
        nb = mask_images(nb, masks)
    nb = anonymize_notebook(nb, make_anonymizer(rules))
    if nbfile == public:
        nbfile.rename(localfile)
        print(f'working notebook kept as {localfile} (ignored by git): edit and run this one from now on')
    public.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return html, public


def _parameters_cell(parameters: Dict[str, object]) -> dict:
    source = '# Parameters (notebook_to_html)\n' + ''.join(f'{key} = {val!r}\n' for key, val in parameters.items())
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {'tags': [PARAMETERS_TAG]},
            'outputs': [], 'source': source}


def execute_notebook(nb: dict, workdir: Union[str, Path], parameters: Optional[Dict[str, object]] = None,
                     rules: Optional[List[Tuple[str, str]]] = None, kernel_name: Optional[str] = None,
                     timeout: Optional[int] = None) -> dict:
    """execute a copy of nb in workdir with static figures (and anonymized figure texts if rules)

    parameters: cell of assignments inserted after the cell tagged 'parameters' (as papermill)
    errors do not stop the execution. The hidden setup cell is removed, execution counts start at 1
    """
    import nbformat
    from nbclient import NotebookClient

    nb = copy.deepcopy(nb)
    cells = nb['cells']
    if parameters:
        position = next((i + 1 for i, cell in enumerate(cells)
                         if 'parameters' in cell.get('metadata', {}).get('tags', [])), 0)
        cells.insert(position, _parameters_cell(parameters))
    setup = (f'from LaueTools.scripts.notebook_to_html import install_static_mode\n'
             f'install_static_mode({json.dumps(rules) if rules else None}, '
             f'browsers={"orientationclustering" in json.dumps(cells)})\n')
    cells.insert(0, {'cell_type': 'code', 'execution_count': None, 'metadata': {'tags': [SETUP_TAG]},
                     'outputs': [], 'source': setup})

    nbnode = nbformat.reads(json.dumps(nb), as_version=4)   # sources as strings
    client = NotebookClient(nbnode, timeout=timeout, allow_errors=True,
                            kernel_name=kernel_name or nb['metadata'].get('kernelspec', {}).get('name', 'python3'),
                            resources={'metadata': {'path': str(workdir)}})
    client.execute()
    nb = json.loads(nbformat.writes(nbnode))

    nb['cells'] = [cell for cell in nb['cells'] if SETUP_TAG not in cell.get('metadata', {}).get('tags', [])]
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and cell.get('execution_count'):
            cell['execution_count'] -= 1
            for out in cell.get('outputs', []):
                if out.get('execution_count'):
                    out['execution_count'] -= 1
    return nb


def _parse_value(value: str):
    import yaml
    return yaml.safe_load(value)


def notebook_to_html(nbfile: Union[str, Path], output_dir: Optional[Union[str, Path]] = None,
                     snapshots: Optional[Dict[str, bytes]] = None, execute: bool = False,
                     parameters: Optional[Dict[str, object]] = None, anonymize: bool = False,
                     rulesfile: Optional[Union[str, Path]] = None, kernel_name: Optional[str] = None,
                     timeout: Optional[int] = None,
                     masks: Optional[Dict[int, List[Tuple[int, int]]]] = None) -> Path:
    """write <notebook name>.html (images embedded) in output_dir (default: folder of the notebook)

    execute: execute a temporary copy (with parameters) instead of using the saved outputs
    anonymize: replace proposal, date, sample, dataset names, hosts, local paths (rules: see load_rules())
    masks: rows of saved figures painted white (see mask_images()), saved outputs only
    """
    nbfile = Path(nbfile).resolve()
    output_dir = Path(output_dir) if output_dir else nbfile.parent
    nb = json.loads(nbfile.read_text(encoding='utf-8'))

    rules = None
    if anonymize:
        texts = notebook_texts(nb)
        for value in (parameters or {}).values():   # e.g. YAML configuration file
            path = nbfile.parent / str(value)
            if isinstance(value, str) and path.is_file():
                texts.append(path.read_text(encoding='utf-8', errors='replace'))
        rules = load_rules(rulesfile, texts)

    if execute:
        print(f'executing {nbfile.name} ...')
        nb = execute_notebook(nb, nbfile.parent, parameters, rules, kernel_name, timeout)
    if masks and not execute:
        nb = mask_images(nb, masks)
    nb = static_outputs(nb, snapshots)
    if rules:
        nb = anonymize_notebook(nb, make_anonymizer(rules))
        if not execute:
            images = [i for i, cell in enumerate(nb['cells'])
                      if any('image/png' in out.get('data', {}) for out in cell.get('outputs', []))]
            if images:
                print(f'WARNING: {nbfile.name}: images of saved outputs are not anonymized, check them '
                      f'(cells {images}) or use --execute (or --mask to hide rows of figures)')

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir) / f'{public_stem(nbfile)}.ipynb'
        tmpfile.write_text(json.dumps(nb, indent=1), encoding='utf-8')
        subprocess.run(['jupyter', 'nbconvert', '--to', 'html', '--embed-images', str(tmpfile),
                        '--output-dir', str(output_dir)], check=True)
    return output_dir / f'{public_stem(nbfile)}.html'


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='LaueTools: HTML of notebooks, optionally executed and anonymized')
    parser.add_argument('notebooks', nargs='+', help='.ipynb files')
    parser.add_argument('--output-dir', default=None, help='default: folder of each notebook')
    parser.add_argument('--execute', action='store_true',
                        help='execute a temporary copy (static figures) instead of using the saved outputs')
    parser.add_argument('-p', '--parameter', action='append', default=[], metavar='NAME=VALUE',
                        help='[--execute] parameter set after the cell tagged "parameters" (value read as YAML)')
    parser.add_argument('--anonymize', nargs='?', const='', default=None, metavar='RULES.yaml',
                        help='replace proposal, date, sample, dataset names, hosts and local paths '
                             '(optional YAML file of extra names and patterns)')
    parser.add_argument('--kernel', default=None, help='[--execute] kernel name (default: kernel of the notebook)')
    parser.add_argument('--timeout', type=int, default=None, help='[--execute] max time per cell (s)')
    parser.add_argument('--mask', action='append', default=[], metavar='CELL:Y0-Y1',
                        help='[saved outputs] paint white rows Y0 to Y1 of figures of cell CELL (index of the '
                             'cell in the notebook, from 0), e.g. to hide a path in a title')
    parser.add_argument('--strip', action='store_true',
                        help='write the notebook (anonymized with --anonymize) without outputs instead of the '
                             'HTML; in the folder of the notebook, the original is kept as <name>.local.ipynb')
    parser.add_argument('--publish', action='store_true',
                        help='write the versions to be committed: <name>.html (anonymized, with outputs) and '
                             '<name>.ipynb (anonymized, without outputs); the working notebook is kept as '
                             '<name>.local.ipynb. Default rules: anonymize_rules.local.yaml of the notebook '
                             'folder or of its parent folders')
    parser.add_argument('--keep-outputs', action='store_true',
                        help='[--publish] keep the (anonymized) text outputs in <name>.ipynb')
    parser.add_argument('--segmentation', metavar='CONFIG', default=None,
                        help='[saved outputs] YAML configuration file: static views of the cluster browsers '
                             '(3_segmentation_grains)')
    parser.add_argument('--fit-folder', default=None, help='[--segmentation] default: fit_folder of the configuration')
    parser.add_argument('--threshold', type=float, default=0.8, help='[--segmentation] THRESHOLD (deg)')
    parser.add_argument('--min-size', type=int, default=15, help='[--segmentation] MIN_CLUSTER_SIZE')
    args = parser.parse_args(argv)

    parameters = {}
    for item in args.parameter:
        name, _, value = item.partition('=')
        parameters[name.strip()] = _parse_value(value)
    anonymize = args.anonymize is not None
    rulesfile = args.anonymize or None

    snapshots = None
    if args.segmentation and not args.execute:
        rules = load_rules(rulesfile, [Path(args.segmentation).read_text(encoding='utf-8')]) if anonymize else None
        snapshots = segmentation_snapshots(args.segmentation, args.fit_folder, args.threshold, args.min_size,
                                           rules=rules)
    for nbfile in args.notebooks:
        if args.publish:
            print('written:', *publish_notebook(nbfile, rulesfile, keep_outputs=args.keep_outputs,
                                                snapshots=snapshots, masks=parse_masks(args.mask)))
            continue
        if args.strip:
            print('written:', strip_notebook(nbfile, args.output_dir, anonymize=anonymize, rulesfile=rulesfile))
            continue
        print('written:', notebook_to_html(nbfile, args.output_dir, snapshots, execute=args.execute,
                                           parameters=parameters, anonymize=anonymize, rulesfile=rulesfile,
                                           kernel_name=args.kernel, timeout=args.timeout,
                                           masks=parse_masks(args.mask)))


if __name__ == '__main__':
    main()
