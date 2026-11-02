# -*- coding: utf-8 -*-
"""
The core modules (computation, file reading and writing) must work without wxPython, and importing a module must
not change the matplotlib backend (notebooks and scripts choose their own).

Imports are run in a new interpreter where wx is blocked (sys.modules['wx'] = None), so the check also holds on
machines where wxPython is installed.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# top-level modules that are not checked, with the reason
NOT_CORE = {
    # launchers of the wx GUIs
    "LaueToolsGUI": "GUI launcher", "lauetools": "GUI launcher", "peaksearch": "GUI launcher",
    "indexrefine": "GUI launcher", "buildsummary": "GUI launcher", "mapanalyzer": "GUI launcher",
    "plotmap": "GUI launcher", "plotmeshspecGUI": "GUI launcher",
    # known problems, not related to wx
    "addscriptsfolder": "imports a missing module (copyexamplescripts)",
    "parabolmirror": "not distributed; needs pkg_resources",
}
# modules that need an optional package: checked only when it is installed
OPTIONAL_PACKAGE = {"daxm_workflow": "numba", "Lauehdf5": "tables"}

# wx GUIs: importing them must not change the matplotlib backend either (checked only when wx is installed)
GUI_MODULES = ["FileSeries.Peak_Search", "FileSeries.Index_Refine", "FileSeries.Build_Summary",
               "FileSeries.mainFileSeriesGUI"]

IMPORT_ALL = """
import importlib, json, sys
if {block_wx}:
    sys.modules['wx'] = None
import matplotlib
backend = matplotlib.rcParams['backend']
result = {{}}
for name in {modules!r}:
    try:
        importlib.import_module('LaueTools.' + name)
    except BaseException as exc:
        result[name] = '%s: %s' % (type(exc).__name__, exc)
        continue
    if matplotlib.rcParams['backend'] != backend:
        result[name] = 'changed the matplotlib backend to %s' % matplotlib.rcParams['backend']
        matplotlib.rcParams['backend'] = backend
    else:
        result[name] = 'ok'
print(json.dumps(result))
"""


def core_modules():
    names = []
    for path in sorted((ROOT / "LaueTools").glob("*.py")):
        name = path.stem
        if name == "__init__" or name in NOT_CORE:
            continue
        if name in OPTIONAL_PACKAGE:
            try:
                __import__(OPTIONAL_PACKAGE[name])
            except ImportError:
                continue
        names.append(name)
    return names


def import_in_new_interpreter(modules, block_wx):
    code = IMPORT_ALL.format(modules=modules, block_wx=block_wx)
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, env={**os.environ, "MPLBACKEND": "Agg"})
    assert proc.returncode == 0, proc.stderr.strip().splitlines()[-1:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_core_modules_import_without_wx():
    result = import_in_new_interpreter(core_modules(), block_wx=True)
    problems = {name: status for name, status in result.items() if status != "ok"}
    assert not problems, problems


def test_gui_import_keeps_matplotlib_backend():
    pytest.importorskip("wx")
    result = import_in_new_interpreter(GUI_MODULES, block_wx=False)
    problems = {name: status for name, status in result.items() if status != "ok"}
    assert not problems, problems
