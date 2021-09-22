#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import math


def solve_cos(a, b):
    return math.acos(b / a), -math.acos(b / a)


def solve_sin(a, b):
    return math.asin(b / a), math.pi - math.asin(b / a)


def solve_combi(a, b, c):
    """Solve a*sin(theta) + b*cos(theta) = c"""
    if a == 0.:

        return solve_cos(b, c)

    else:

        if b == 0.:

            return solve_sin(a, c)

        else:

            phi = 0

            if a > 0:

                phi = math.atan(b / a)

            elif a < 0:

                phi = math.atan(b / a) + math.pi

            theta1, theta2 = solve_sin(math.sqrt(a ** 2 + b ** 2), c)

            return theta1 - phi, theta2 - phi


