#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import scipy.stats as stats


def trimmean(x, limits=0.05, axis=None):
    return stats.mstats.trimmed_mean(x, limits=limits, inclusive=(0, 0), relative=True, axis=axis)