#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import numpy as np


def closest_point(node, nodes):
    node = np.array(node)
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.sqrt(np.amin(dist_2)), int(np.argmin(dist_2))


if __name__ == "__main__":

    print(closest_point([0, 0], [[0, 1],[0, 2]]))
