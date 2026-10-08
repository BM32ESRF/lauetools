import os, shutil, argparse, json

try:
    from importlib import resources
except ImportError:
    import importlib_resources as resources

LAUETOOLSFOLDER = os.path.split(__file__)[0]

CONFIG_DIR = LAUETOOLSFOLDER
CONFIG_FILE = os.path.join(CONFIG_DIR, "lauetools_config.json")

def save_config(data):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}

# resources installed with LaueTools that a user can copy to his own folder:
# name: (source relative to LAUETOOLSFOLDER, destination relative to the chosen folder, patterns)
# the notebooks folders are copied with their layout (they use relative paths: configs/, ../indexation/)
RESOURCES = {
    "notebooks": [("notebooks/quickstart", "notebooks/quickstart", ["*.ipynb", "*.md"]),
                  ("notebooks/peaksearch", "notebooks/peaksearch", ["*.ipynb", "*.md", "*.yaml"]),
                  ("notebooks/indexation", "notebooks/indexation", ["*.ipynb", "*.md", "*.yaml"])],
    "tutorials": [("notebooks", "tutorials", ["*.ipynb"])],
    "scripts": [("scripts", "scripts", ["*.py"])],
    "images": [("LaueImages", "LaueImages", ["small_example_*"])],
}
# files never copied (personal or temporary files)
EXCLUDED = ["*.local.*", "Untitled*", "*_jsm.ipynb", "__init__.py"]


def _resource_files(src, patterns):
    """List the files of folder src (and its subfolders) matching one of patterns."""
    import fnmatch
    files = []
    for root, dirs, names in os.walk(src):
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "__")))
        if os.path.basename(src) == "notebooks":  # tutorials: top level only
            dirs[:] = []
        for name in sorted(names):
            if any(fnmatch.fnmatch(name, p) for p in patterns) and not any(
                    fnmatch.fnmatch(name, p) for p in EXCLUDED):
                files.append(os.path.join(root, name))
    return files


def copy_examples(dest, what=("notebooks", "tutorials", "scripts", "images"), force=False,
                  verbose=True):
    """Copy notebooks, scripts and example images installed with LaueTools to folder dest.

    Existing files are not overwritten (they may have been edited by the user) unless force=True.

    :return: (list of copied files, list of skipped existing files)
    """
    dest = os.path.abspath(os.path.expanduser(dest))
    copied, skipped = [], []
    for key in what:
        for src_rel, dest_rel, patterns in RESOURCES[key]:
            src = os.path.join(LAUETOOLSFOLDER, src_rel)
            if not os.path.isdir(src):
                continue
            for s in _resource_files(src, patterns):
                d = os.path.join(dest, dest_rel, os.path.relpath(s, src))
                if os.path.exists(d) and not force:
                    skipped.append(d)
                    continue
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copy2(s, d)
                copied.append(d)
    if verbose:
        for d in copied:
            print("  copied ", os.path.relpath(d, dest))
        if skipped:
            print(f"{len(skipped)} existing files not overwritten (use --force to overwrite them)")
        print(f"Great! {len(copied)} LaueTools files copied to: {dest}")
        if "notebooks" in what:
            print("To start: jupyter lab "
                  + os.path.join(dest, "notebooks", "quickstart", "laue_maps_quickstart.ipynb"))
    return copied, skipped


# small default files installed with LaueTools (Examples/<name>) and used by the GUIs to test them rapidly
# (e.g. Ge peaks lists for Detector Calibration and Autoindexation in mainGUI)
DEFAULT_EXAMPLES = {"Ge": ["img_Ge_sCMOS_0000_181peaks.cor", "img_Ge_sCMOS_0000_2peaks.cor"]}
USER_EXAMPLES_DIR = os.path.join(os.path.expanduser("~"), ".lauetools", "Examples")


def writable_examples_folder(name="Ge", verbose=True):
    """Folder of the default example files Examples/<name> where the GUIs can also write their results.

    - the installed folder LaueTools/Examples/<name> if it is writable (git clone, user's own environment)
    - otherwise (read-only installation, e.g. shared python environment) ~/.lauetools/Examples/<name>,
      where the default files are copied the first time (files already there are not overwritten)

    :return: path of the folder (the installed folder if no writable copy can be made)
    """
    src = os.path.join(LAUETOOLSFOLDER, "Examples", name)
    if os.access(src, os.W_OK):
        return src
    dest = os.path.join(USER_EXAMPLES_DIR, name)
    try:
        os.makedirs(dest, exist_ok=True)
        for filename in DEFAULT_EXAMPLES.get(name, []):
            s, d = os.path.join(src, filename), os.path.join(dest, filename)
            if os.path.exists(s) and not os.path.exists(d):
                shutil.copy2(s, d)
                os.chmod(d, os.stat(d).st_mode | 0o200)  # user can overwrite it
                if verbose:
                    print(f"copied {filename} to {dest}")
    except OSError as err:
        print(f"Cannot make a writable copy of {src} in {dest}: {err}")
        return src
    return dest


# example images (too large for the PyPI package) are files attached to this GitHub release:
# every file attached to it is downloaded by lauetools-copy --download (no code change to add one)
GITHUB_REPO = "BM32ESRF/lauetools"
EXAMPLES_RELEASE = "examples-data"


def release_assets(tag=EXAMPLES_RELEASE, repo=GITHUB_REPO):
    """[(name, size in bytes, download url)] of the files attached to the GitHub release tag"""
    import urllib.request
    request = urllib.request.Request(f"https://api.github.com/repos/{repo}/releases/tags/{tag}",
                                     headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)
    return [(a["name"], a["size"], a["browser_download_url"]) for a in release["assets"]]


def download_examples(dest, tag=EXAMPLES_RELEASE, subfolder="LaueImages", force=False, verbose=True):
    """Download the files attached to the GitHub release tag to <dest>/<subfolder>.

    Files already downloaded (same size) are skipped unless force=True.

    :return: (list of downloaded files, list of skipped existing files)
    """
    import urllib.error
    import urllib.request
    folder = os.path.join(os.path.abspath(os.path.expanduser(dest)), subfolder)
    try:
        assets = release_assets(tag)
    except urllib.error.HTTPError as err:
        print(f"No example files: release '{tag}' of github.com/{GITHUB_REPO} not found ({err})")
        return [], []
    except (urllib.error.URLError, OSError) as err:
        print(f"Cannot reach github.com (no internet access or proxy needed?): {err}")
        return [], []
    downloaded, skipped = [], []
    os.makedirs(folder, exist_ok=True)
    for name, size, url in assets:
        d = os.path.join(folder, name)
        if os.path.exists(d) and os.path.getsize(d) == size and not force:
            skipped.append(d)
            continue
        if verbose:
            print(f"  downloading {name} ({size / 1e6:.1f} MB)")
        urllib.request.urlretrieve(url, d + ".part")
        os.replace(d + ".part", d)
        downloaded.append(d)
    if verbose:
        print(f"{len(downloaded)} example files downloaded to: {folder}"
              + (f" ({len(skipped)} already there)" if skipped else ""))
    return downloaded, skipped


def _choose_folder_gui():
    """Ask the destination folder in a dialog (wxPython). Return None if cancelled."""
    import wx
    app = wx.App(False)
    with wx.DirDialog(None, "Folder where LaueTools notebooks and examples will be copied",
                      os.path.expanduser("~"), wx.DD_DEFAULT_STYLE) as dlg:
        folder = dlg.GetPath() if dlg.ShowModal() == wx.ID_OK else None
    app.Destroy()
    return folder


def copy_resources(argv=None, gui=False):
    """Command lauetools-copy: copy notebooks, scripts and example images to a folder."""
    parser = argparse.ArgumentParser(
        description="Copy LaueTools notebooks (quickstart, peak search, indexation with their "
                    "YAML configuration files), tutorials notebooks, scripts and example images "
                    "to a folder of your choice.")
    parser.add_argument("-d", "--destination", default=None,
                        help="Destination folder (default: ./lauetools_examples, "
                             "or chosen in a dialog with --gui)")
    parser.add_argument("-w", "--what", nargs="+", choices=list(RESOURCES) + ["all"],
                        default=["all"], help="What to copy (default: all)")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Overwrite files already present in the destination folder")
    parser.add_argument("--download", action="store_true",
                        help=f"Also download the example images (files of the GitHub release "
                             f"'{EXAMPLES_RELEASE}', too large for the package) to <destination>/LaueImages")
    parser.add_argument("--release", default=EXAMPLES_RELEASE, help="[--download] GitHub release tag")
    parser.add_argument("--gui", action="store_true", help="Choose the destination folder in a dialog")
    args = parser.parse_args(argv)

    what = list(RESOURCES) if "all" in args.what else args.what
    dest = args.destination
    gui = gui or args.gui
    download = args.download
    if dest is None and gui:
        dest = _choose_folder_gui()
        if dest is None:
            print("Cancelled.")
            return
        import wx
        app = wx.App(False)
        download = download or wx.MessageBox(
            "Also download the example images from GitHub (internet access needed)?",
            "LaueTools examples", wx.YES_NO | wx.ICON_QUESTION) == wx.YES
        app.Destroy()
    if dest is None:
        dest = os.path.join(os.getcwd(), "lauetools_examples")
    copied, skipped = copy_examples(dest, what=what, force=args.force)
    downloaded = []
    if download:
        downloaded, skipped_dl = download_examples(dest, tag=args.release, force=args.force)
        skipped += skipped_dl

    if gui:
        import wx
        app = wx.App(False)
        msg = f"{len(copied)} files copied to:\n{os.path.abspath(dest)}"
        if download:
            msg += f"\n\n{len(downloaded)} example images downloaded to LaueImages/"
        if skipped:
            msg += f"\n\n{len(skipped)} existing files were not overwritten."
        wx.MessageBox(msg, "LaueTools examples", wx.OK | wx.ICON_INFORMATION)
        app.Destroy()


def copy_resources_gui():
    """Command lauetools-copy-gui (no terminal needed): choose the folder in a dialog."""
    copy_resources(gui=True)


def copy_materials():
    """Copy materials.yaml to a user-writable folder."""
    parser = argparse.ArgumentParser(
        description="Copy LaueTools materials.yaml to a user-writable folder."
    )
    parser.add_argument(
        "-d", "--destination",
        default=os.path.join(os.path.expanduser("~"), ".lauetools"),
        help="Destination folder (default: ~/.lauetools)",
    )
    args = parser.parse_args()

    dest = os.path.abspath(args.destination)
    os.makedirs(dest, exist_ok=True)

    with resources.path("LaueTools", "materials.yaml") as src:
        dst_file = os.path.join(dest, "materials.yaml")
        shutil.copy2(src, dst_file)

    # Save chosen folder in persistent config
    cfg = load_config()
    cfg["materials_dir"] = dest
    save_config(cfg)

    print(f"Great materials.yaml copied to: {dst_file}")
    print("You can now edit this file freely.")
