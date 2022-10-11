#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'


import os

def add_extension(filename, extension, overwrite=False):

    dirname, basename = os.path.split(filename)

    parts = basename.split(".")

    if len(parts) <= 1 or overwrite:

        return os.path.join(dirname, parts[0] + "." + extension)

    return filename

def nbasename(path, n=1):

    dirname, basename = os.path.split(path)
    for i in range(n-1):
        dirname, tmp = os.path.split(dirname)
        basename = os.path.join(tmp, basename)

    return basename

class tmpCd():
    def __init__(self, dirname):
        self.dirname = dirname

    def __enter__(self):
        self.curdir = os.getcwd()

        if self.dirname is not None and self.dirname != "":
            os.chdir(self.dirname)

    def __exit__(self, type, value, traceback):
        os.chdir(self.curdir)
