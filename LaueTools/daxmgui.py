# script to launch DaxmAnalyzerGui  (3D Laue, wire scans)
import importlib

# packages needed only by DAXM codes: pip install "lauetools[daxm]"
DAXM_REQUIREMENTS = {"numba": "numba", "photutils": "photutils", "pubsub": "pypubsub"}


def missing_daxm_packages():
    """return the list of pip package names required by DAXM codes and not installed"""
    missing = []
    for module_name, pip_name in DAXM_REQUIREMENTS.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(pip_name)
    return missing


def start():
    """launch the DAXM GUI (entry point 'daxmgui')"""
    missing = missing_daxm_packages()
    if missing:
        raise SystemExit("DAXM codes need the packages: %s\n"
                         "install them with:  pip install \"lauetools[daxm]\"" % ", ".join(missing))

    import LaueTools.Daxm.DaxmAnalyzerGui as dGUI

    dGUI.start()


if __name__ == "__main__":
    start()
