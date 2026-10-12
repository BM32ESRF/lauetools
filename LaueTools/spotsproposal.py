# -*- coding: utf-8 -*-
r"""
Proposal of Laue spots likely to have low Miller indices, to be used as central spots (spots set A)
of the indexation by recognition of angles between pairs of spots.

Two rankings of the spots of a Laue pattern:

- rank_spots_by_zones(): number of zone axes passing through each spot. In the gnomonic projection the spots
  of a zone axis are aligned: spots at the intersections of many lines have low indices.
  No lattice parameter is needed.
- rank_spots_by_triplets(): number of triplets of spots whose 3 mutual angles are those of 3 directions of low
  indices of a given unit cell. Accurate lattice parameters and a small tolerance are needed.

Lauetools project
"""
import sys

import numpy as np

if sys.version_info.major == 3:
    from . import CrystalParameters as CP
    from . import generaltools as GT
    from . import indexingImageMatching as IIM
else:
    import CrystalParameters as CP
    import generaltools as GT
    import indexingImageMatching as IIM


def gnomonic_directions(twotheta, chi):
    """gnomonic coordinates x, y of spots and corresponding unit vectors (x, y, 1) / norm

    A line of the gnomonic plane is a plane containing the origin of these vectors, so that the distance
    of a spot to a line is an angle.
    """
    gnomx, gnomy = IIM.ComputeGnomon_2((np.array(twotheta), np.array(chi)))
    vec = np.array([gnomx, gnomy, np.ones(len(gnomx))]).T
    return gnomx, gnomy, vec / np.linalg.norm(vec, axis=1)[:, None]


def find_zones(vec, tol=0.05, nmin=5, minsep=1.):
    """find zone axes = lines of the gnomonic plane containing at least nmin spots

    :param vec: array (n, 3) of unit vectors of spots, see gnomonic_directions()
    :param tol: max. angular distance (deg) of a spot to the line
    :param nmin: min. nb of spots of a zone
    :param minsep: min. angular separation (deg) of the 2 spots defining a trial line

    :return: list of (unit vector normal to the plane of the zone, boolean array of the spots of the zone),
        sorted by decreasing nb of spots
    """
    nb = len(vec)
    ii, jj = np.triu_indices(nb, 1)
    normals = np.cross(vec[ii], vec[jj])
    norms = np.linalg.norm(normals, axis=1)
    keep = norms > np.sin(np.radians(minsep))
    ii, jj, normals = ii[keep], jj[keep], normals[keep] / norms[keep][:, None]
    sintol = np.sin(np.radians(tol))
    members = np.abs(normals @ vec.T) <= sintol    # (pairs, spots)
    counts = members.sum(axis=1)
    done = np.zeros(len(ii), dtype=bool)
    zones = []
    for p in np.argsort(-counts, kind='stable'):
        if counts[p] < nmin:
            break
        if done[p]:
            continue
        normal, m = normals[p], members[p]
        for _ in range(3):                         # least squares fit of the line on its spots
            fitted = np.linalg.svd(vec[m])[2][-1]
            m2 = np.abs(vec @ fitted) <= sintol
            if m2.sum() < m.sum():
                break
            normal, m = fitted, m2
        done |= m[ii] & m[jj]
        if m.sum() >= nmin and not any((m & zm).sum() >= max(3, 0.6 * min(m.sum(), zm.sum())) for _, zm in zones):
            zones.append((normal, m))
    zones.sort(key=lambda zone: -zone[1].sum())
    return zones


def rank_spots_by_zones(twotheta, chi, tol=0.05, nmin=5):
    """rank spots by the nb of zone axes passing through them (ties: total nb of spots of these zones)

    :param twotheta, chi: scattering angles (deg) of the spots
    :param tol: max. angular distance (deg) of a spot to a zone line
    :param nmin: min. nb of spots of a zone

    :return: spot indices (best first), nb of zones of each spot (in the order of the input), list of zones
    """
    vec = gnomonic_directions(twotheta, chi)[2]
    zones = find_zones(vec, tol, nmin)
    nbzones = np.zeros(len(vec), dtype=int)
    nbzonespots = np.zeros(len(vec), dtype=int)
    for _, m in zones:
        nbzones[m] += 1
        nbzonespots[m] += m.sum()
    return np.lexsort((-nbzonespots, -nbzones)), nbzones, zones


def angle_triples(latticeparams, maxindex=2, anglemax=180.):
    """distinct triples of angles (deg) between 3 reciprocal directions of indices up to maxindex

    :param latticeparams: [a, b, c, alpha, beta, gamma] direct lattice parameters (Angstrom, deg)
    :param anglemax: triples with an angle larger than this value are left out

    :return: sorted list of (angle_ab, angle_bc, angle_ac), each sorted in increasing order
    """
    Gstar = CP.Gstar_from_directlatticeparams(*latticeparams)
    hkls = GT.threeindices_up_to(maxindex).astype(int)
    hkls = hkls[np.gcd.reduce(np.abs(hkls), axis=1) == 1]
    angles = np.round(CP.AngleBetweenNormals(hkls.astype(float), hkls.astype(float), Gstar), 3)
    ok = (angles > 0.5) & (angles < anglemax)
    triples = set()
    for a in range(len(hkls)):
        bs = np.where(ok[a])[0]
        bs = bs[bs > a]
        for b in bs:
            for c in bs[(bs > b) & ok[b, bs]]:
                triples.add(tuple(sorted((angles[a, b], angles[b, c], angles[a, c]))))
    return sorted(triples)


def rank_spots_by_triplets(twotheta, chi, latticeparams, maxindex=2, tol=0.05, triples=None):
    """rank spots by the nb of recognised triplets of spots they belong to.

    A triplet of spots is recognised if its 3 mutual angular distances are equal (within tol) to the 3 angles
    between 3 directions of indices up to maxindex of the unit cell.

    :param twotheta, chi: scattering angles (deg) of the spots
    :param latticeparams: [a, b, c, alpha, beta, gamma] direct lattice parameters (Angstrom, deg)
    :param tol: tolerance (deg) on each angle. It must be small (< 0.1 deg): lattice parameters must be accurate
    :param triples: precomputed angle_triples(latticeparams, maxindex) (to treat many patterns)

    :return: spot indices (best first), nb of triplets of each spot (in the order of the input)
    """
    thetachi = np.array([np.array(twotheta) / 2., np.array(chi)]).T
    tabdist = GT.calculdist_from_thetachi(thetachi, thetachi)
    distmax = tabdist.max() + tol
    if triples is None:
        triples = angle_triples(latticeparams, maxindex, distmax)
    cache = {}

    def recognised(angle):
        if angle not in cache:
            m = (np.abs(tabdist - angle) <= tol).astype(np.float32)
            np.fill_diagonal(m, 0)
            cache[angle] = m
        return cache[angle]

    counts = np.zeros(len(thetachi))
    for x, y, z in triples:
        if max(x, y, z) > distmax:
            continue
        mx, my, mz = recognised(x), recognised(y), recognised(z)   # spots (i, j, k): d_ij = x, d_jk = y, d_ik = z
        if not (mx.any() and my.any() and mz.any()):
            continue
        tik = (mx @ my) * mz
        if tik.any():
            counts += tik.sum(axis=1) + tik.sum(axis=0) + (mx * (mz @ my.T)).sum(axis=0)
    return np.argsort(-counts, kind='stable'), counts
