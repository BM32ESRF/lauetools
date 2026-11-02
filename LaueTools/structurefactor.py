# -*- coding: utf-8 -*-
"""
Systematic extinctions computed from symmetry operations: a reflection hkl is absent when the waves scattered by
all the symmetry-equivalent positions cancel, sum_j exp(2 i pi hkl.(R_j r + t_j)) = 0.

- general_allowed(): general reflection conditions of a space group (lattice centering, glide planes, screw
  axes), from the coordinates of its general Wyckoff position (wyckpos_lauetools), for any setting
- parse_symop(): symmetry operation 'x, y, z' / '-x+1/2, y, -z' (CIF, International Tables) -> (R, t)
"""
import re

import numpy as np

from . import wyckpos_lauetools as WYCK

TOLERANCE = 1e-6  # |sum of phases| / number of positions below which a reflection is absent
_SAFE_SYMOP = re.compile(r"^[xyz0-9+\-*/. ]*$")


def parse_symop(op):
    """rotation R (3x3) and translation t (3) of a symmetry operation: position -> R position + t

    :param op: str, e.g. 'x, y, z', '-x+1/2, y, -z', '(x-y, -y, -z+1/2)', '1/2+x,1/2-y,z' (implicit
        multiplication '2x' accepted)
    """
    terms = op.strip().strip("()").lower().split(",")
    if len(terms) != 3:
        raise ValueError("symmetry operation '%s' must have 3 coordinates" % op)
    terms = [re.sub(r"(\d)\s*([xyz])", r"\1*\2", term.strip()) for term in terms]
    if not all(_SAFE_SYMOP.match(term) and "**" not in term for term in terms):
        raise ValueError("cannot read symmetry operation '%s'" % op)
    code = compile("(%s)" % ", ".join(terms), "<symop>", "eval")
    evaluate = lambda x, y, z: np.array(eval(code, {"__builtins__": {}}, {"x": x, "y": y, "z": z}),
                                        dtype=float)
    t = evaluate(0.0, 0.0, 0.0)
    R = np.array([evaluate(*e) - t for e in np.eye(3)]).T
    return R, t


def spacegroup_symops(spacegroup):
    """symmetry operations (list of (R, t)) of a space group from the coordinates of its general position

    :param spacegroup: number or 'number:setting' (see wyckpos_lauetools.normalize_spacegroup())
    """
    key = WYCK.normalize_spacegroup(spacegroup)
    positions = WYCK.wp[key]
    general = sorted(positions, key=lambda label: int(label[:-1]))[-1]
    return [parse_symop(op) for op in positions[general][1]]


def _phase_sums(hkl, points, chunk=20000):
    """|sum_j exp(2 i pi hkl.point_j)| / nb points, for each hkl"""
    hkl = np.asarray(hkl, dtype=float)
    points = np.asarray(points, dtype=float)
    result = np.empty(len(hkl))
    for start in range(0, len(hkl), chunk):
        phases = 2 * np.pi * hkl[start:start + chunk] @ points.T
        result[start:start + chunk] = np.abs(np.exp(1j * phases).sum(axis=1)) / len(points)
    return result


_CACHE = {}
_CACHE_SIZE = 64


def general_allowed(hkl, spacegroup, seed=0):
    """boolean mask of reflections allowed by the general reflection conditions of the space group (all of them,
    including the conditions on reflections equivalent by symmetry, e.g. 0k0 from h00 in tetragonal groups)

    computed with the positions equivalent to 2 random (general) positions: a reflection is absent only if the
    sum of phases vanishes for both. Results are cached (same space group and same hkl array).
    """
    hkl = np.ascontiguousarray(hkl)
    key = (WYCK.normalize_spacegroup(spacegroup), hkl.shape, hash(hkl.tobytes()), seed)
    if key not in _CACHE:
        ops = spacegroup_symops(spacegroup)
        rng = np.random.default_rng(seed)
        allowed = np.zeros(len(hkl), dtype=bool)
        for r in rng.random((2, 3)):
            points = [R @ r + t for R, t in ops]
            allowed |= _phase_sums(hkl, points) > TOLERANCE
        if len(_CACHE) >= _CACHE_SIZE:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = allowed
    return _CACHE[key].copy()
