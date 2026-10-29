# -*- coding: utf-8 -*-
"""
import of the main modules (no GUI) and check of the commands of pyproject.toml

The commands are checked without importing their modules (GUIs need a display): the module file must exist
and define the function (a broken command was 'plotmeshGUI = LaueTools.plotmeshspecGUI:start', a script
without start())
"""
import ast
import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

MODULES = ["CrystalParameters", "LaueGeometry", "IOLaueTools", "IOimagefile", "generaltools", "lauecore",
           "FitOrient", "readmccd", "imageprocessing", "indexingAnglesLUT", "indexingSpotsSet", "indexing_batch",
           "indexing_neighbours", "indexing_sequence", "peaksearch_batch", "matchingrate", "fitfilereader",
           "orientationclustering", "fitfile_strainfix", "map_workflow", "ipfmap", "LaueDataResultsPlots",
           "grainimaging", "detectorbands", "dict_LaueTools", "logfile_reader", "daxm_workflow", "notebook_export",
           "scripts.workflows", "scripts.workflowlauncher", "scripts.roicountersplot", "FileSeries.multigrainFS",
           "FileSeries.Build_Summary", "Daxm.quickdaxm"]


def pyproject_commands():
    """{command: 'module:function'} of [project.scripts] and [project.gui-scripts] (no tomllib before python 3.11)"""
    commands, section = {}, None
    for line in (ROOT / "pyproject.toml").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line.startswith("["):
            section = line
        elif section in ("[project.scripts]", "[project.gui-scripts]"):
            match = re.match(r'([\w.-]+)\s*=\s*"([\w.]+:\w+)"', line)
            if match:
                commands[match.group(1)] = match.group(2)
    return commands


@pytest.mark.parametrize("module", MODULES)
def test_import(module):
    importlib.import_module("LaueTools." + module)


def test_commands_found():
    assert "lauetools" in pyproject_commands()


@pytest.mark.parametrize("command", sorted(pyproject_commands()))
def test_command_target_exists(command):
    modulename, function = pyproject_commands()[command].split(":")
    path = ROOT.joinpath(*modulename.split("."))
    sourcefile = path.with_suffix(".py") if path.with_suffix(".py").exists() else path / "__init__.py"
    assert sourcefile.exists(), "%s: no module %s" % (command, modulename)
    tree = ast.parse(sourcefile.read_text(errors="ignore"))
    names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    assert function in names, "%s: no function %s in %s" % (command, function, sourcefile.name)
