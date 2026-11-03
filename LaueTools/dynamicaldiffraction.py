# -*- coding: utf-8 -*-
"""
Back-surface Laue spots of thick perfect crystals (dynamical diffraction), e.g. silicon wafers.

In a thick perfect crystal the X-ray energy of a Bragg reflection does not flow along the incoming beam but along
the Poynting vector S of the wavefield, inside the fan between the incoming direction ui and the diffracted
direction uf. A part of it reaches the back surface of the slab, is Bragg reflected there and leaves the front
surface further along: each Laue spot has a weaker secondary spot (or a streak towards it), displaced in the
scattering plane, i.e. mainly along the 2theta direction on the detector (Y pixels in the BM32 geometry).

Reference: H. Yan and I. C. Noyan, J. Appl. Phys. 98, 073527 (2005), "Dynamical diffraction artifacts in Laue
microdiffraction images" (symmetric Bragg case, eqs. 4-7). This module generalizes their geometry to 3D:

- S(alpha) = cos(alpha) t - sin(alpha) q, with q the unit scattering vector (normal to the lattice planes) and t
  the unit vector along the lattice planes in the scattering plane: alpha = theta gives ui, alpha = -theta gives uf
- depth reached by the wavefield per unit of absorption, for a surface of outward normal n:
  (-S.n) / mu_eff(alpha), with mu_eff(alpha) / mu = (cos(alpha) - |C| eps sqrt(1 - sin^2 alpha / sin^2 theta))
  / cos(theta) (weakly absorbed branch, Borrmann effect); the energy reaching the back surface flows along
  alpha_in, the angle of maximum penetration (paper eq. 4, or eq. 6 in the symmetric case, where it does not
  depend on the absorption coefficient mu)
- the wavefield reflected by the back surface flows back to the front surface along alpha_out, the angle of the
  fan of maximum height reached per unit of absorption (S.n) / mu_eff (-alpha_in in the symmetric case)
- the exit point E on the front surface lies in the scattering plane, so the back-surface spot is the spot of a
  source displaced along the incoming beam (LaueGeometry.calc_xycam(offset)): E = offset ui + s uf

eps is |chi_h| / |chi_0| for the imaginary (absorption) parts of the susceptibility, close to the ratio of the
structure factor to its value at hkl = 000, times the Debye-Waller factor: see epsilon_from_positions().

Vectors are in the frame of LaueGeometry.calc_xycam(): y // incoming beam, z towards the detector on top
(vertical at BM32), x = y ^ z (horizontal).
"""
import numpy as np

from . import LaueGeometry as LG

DEG = np.pi / 180.0
UI = np.array([0.0, 1.0, 0.0])  # incoming beam in the frame of LaueGeometry.calc_xycam()
POLARIZATION = np.array([1.0, 0.0, 0.0])  # bending magnet / undulator: linear, horizontal, normal to the beam

# fractional positions of the 8 atoms of the diamond structure (Si, Ge, C), space group 227 origin choice 1
DIAMOND_POSITIONS = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0),
                     (0.25, 0.25, 0.25), (0.25, 0.75, 0.75), (0.75, 0.25, 0.75), (0.75, 0.75, 0.25)]
SI_LATTICE = [5.4309, 5.4309, 5.4309, 90.0, 90.0, 90.0]
# effective Debye-Waller B (Angstrom^2) of the ratio eps for silicon: 0.62 reproduces the angles alpha_m of Yan &
# Noyan Table I (Si 004 and 333, 10 to 25 keV) within 0.1 deg; the thermal B of silicon at room temperature (0.46)
# gives alpha_m 1.5 deg too small for 004 at 10 keV (absorption by thermal diffuse scattering, which adds to chi_0
# only, probably makes eps smaller)
SI_DEBYE_WALLER_B = 0.62


def surface_normal(sampletilt=40.0):
    """outward unit normal of the sample front surface, sample tilted by sampletilt (deg) from the incoming beam
    around the horizontal axis x (BM32 geometry): (0, -sin(tilt), cos(tilt))"""
    return np.array([0.0, -np.sin(sampletilt * DEG), np.cos(sampletilt * DEG)])


def scattering_frame(uf, ui=UI):
    """unit vectors of the scattering plane of the reflection of diffracted direction uf

    :return: t (along the lattice planes, (ui + uf) / |ui + uf|), q (normal to the lattice planes,
        (uf - ui) / |uf - ui|), theta (Bragg angle, rad)
    """
    uf = np.asarray(uf, dtype=float) / np.linalg.norm(uf)
    ui = np.asarray(ui, dtype=float) / np.linalg.norm(ui)
    t = (ui + uf) / np.linalg.norm(ui + uf)
    q = (uf - ui) / np.linalg.norm(uf - ui)
    theta = np.arccos(np.clip(np.dot(ui, uf), -1.0, 1.0)) / 2.0
    return t, q, theta


def flow_direction(alpha, t, q):
    """unit vector S of energy flow in the scattering plane, at angle alpha (rad) from the lattice planes
    (alpha = theta: incoming direction, alpha = -theta: diffracted direction)"""
    alpha = np.asarray(alpha, dtype=float)
    return np.multiply.outer(np.cos(alpha), t) - np.multiply.outer(np.sin(alpha), q)


def relative_absorption(alpha, theta, C=1.0, eps=1.0):
    """mu_eff(alpha) / mu along the flow direction alpha (rad), weakly absorbed branch (Yan & Noyan, after eq. 4)

    :param C: polarization factor (1 for sigma, |cos(2 theta)| for pi polarization)
    :param eps: |chi_h| / |chi_0| (imaginary parts), see epsilon_from_positions()
    """
    root = np.sqrt(np.clip(1.0 - np.sin(alpha) ** 2 / np.sin(theta) ** 2, 0.0, None))
    return (np.cos(alpha) - abs(C) * eps * root) / np.cos(theta)


def _best_alpha(theta, t, q, normal, C, eps, downwards, nb=4001):
    """angle alpha of the fan maximizing the distance travelled normal to the surface (downwards: into the crystal,
    else back to the front surface) per unit of absorption; None if no direction of the fan goes that way"""
    from scipy.optimize import minimize_scalar

    def gain(alpha):
        along_normal = -np.dot(flow_direction(alpha, t, q), normal)
        if not downwards:
            along_normal = -along_normal
        mu = relative_absorption(alpha, theta, C, eps)
        return np.where(along_normal > 0, along_normal / np.maximum(mu, 1e-12), 0.0)

    alphas = np.linspace(-theta, theta, nb)
    values = gain(alphas)
    if not np.any(values > 0):
        return None
    k = int(np.argmax(values))
    lo, hi = alphas[max(k - 1, 0)], alphas[min(k + 1, nb - 1)]
    res = minimize_scalar(lambda a: -float(gain(a)), bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-10})
    return float(res.x) if -res.fun >= values[k] else float(alphas[k])


def alpha_m_symmetric(theta, C=1.0, eps=1.0):
    """angle (rad) of maximum penetration in the symmetric Bragg case, Yan & Noyan eq. 6:
    tan(alpha_m) = sqrt(1 - C^2 eps^2) tan(theta)"""
    return np.arctan(np.sqrt(1.0 - (C * eps) ** 2) * np.tan(theta))


def separation_symmetric(thickness, theta, alpha_m):
    """distance (same unit as thickness) between the beams diffracted by the front and back surfaces, symmetric
    Bragg case, beam far from the sample edges, Yan & Noyan eq. 7 (multiple Bragg-Bragg case)"""
    return 2.0 * thickness / np.tan(alpha_m) * np.sin(theta)


def polarization_factor(uf, ui=UI, polarization=POLARIZATION, mode="auto"):
    """polarization factor C of the reflection and the component it corresponds to

    :param mode: 'sigma' (C = 1), 'pi' (C = |cos 2theta|), or 'auto': the main component of the incoming linear
        polarization for this scattering plane
    :return: C, 'sigma' or 'pi'
    """
    _, _, theta = scattering_frame(uf, ui)
    if mode == "auto":
        normal = np.cross(ui, uf)
        sigma = (np.dot(normal, polarization) / np.linalg.norm(normal)) ** 2 if np.linalg.norm(normal) > 0 else 1.0
        mode = "sigma" if sigma >= 0.5 else "pi"
    if mode == "sigma":
        return 1.0, "sigma"
    if mode == "pi":
        return abs(np.cos(2 * theta)), "pi"
    raise ValueError("polarization mode must be 'auto', 'sigma' or 'pi', not %r" % (mode,))


def epsilon_from_positions(hkl, positions, weights=None, dspacing=None, debye_waller_B=0.0):
    """eps = |chi_h| / |chi_0| for the absorption parts: |sum_j w_j exp(2 i pi hkl.r_j)| / sum_j w_j
    times the Debye-Waller factor exp(-B / (4 d^2))

    Approximation: the imaginary part f'' of the atomic scattering factors hardly depends on the scattering
    angle, so that the ratio only depends on the positions of the atoms. Exact for one element (weights = None);
    for several elements, give weights proportional to f'' of each atom at the X-ray energy.

    :param positions: fractional coordinates of all the atoms of the unit cell, e.g. DIAMOND_POSITIONS
    :param dspacing: d of the reflection hkl (Angstrom), needed if debye_waller_B != 0
    :param debye_waller_B: B (Angstrom^2), e.g. SI_DEBYE_WALLER_B
    """
    positions = np.asarray(positions, dtype=float)
    weights = np.ones(len(positions)) if weights is None else np.asarray(weights, dtype=float)
    ratio = abs(np.sum(weights * np.exp(2j * np.pi * positions @ np.asarray(hkl, dtype=float)))) / weights.sum()
    if debye_waller_B:
        if not dspacing:
            raise ValueError("dspacing is needed for the Debye-Waller factor")
        ratio *= np.exp(-debye_waller_B / (4.0 * dspacing ** 2))
    return float(ratio)


def dspacing(hkl, lattice):
    """d spacing (Angstrom) of hkl for lattice parameters [a, b, c, alpha, beta, gamma] (Angstrom, deg)"""
    from . import CrystalParameters as CP
    return 1.0 / np.linalg.norm(CP.calc_B_RR(lattice) @ np.asarray(hkl, dtype=float))


def back_surface_path(uf, thickness, normal, eps, C=1.0, ui=UI):
    """path of the energy reflected by the back surface of a perfect slab, for one reflection

    :param uf: diffracted direction (3 elements, frame of calc_xycam())
    :param thickness: slab thickness (mm, or any length unit: the results are in the same unit)
    :param normal: outward unit normal of the front surface, see surface_normal()
    :param eps: |chi_h| / |chi_0|, see epsilon_from_positions()
    :param C: polarization factor, see polarization_factor()
    :return: dict with
        alpha_in, alpha_out: flow angles (deg) towards the back surface and back to the front surface
        exit: exit point E on the front surface (vector, from the point where the beam enters)
        offset: position along the incoming beam of the equivalent source (E = offset ui + s uf), for
            LaueGeometry.calc_xycam(offset=...)
        separation: distance between the beams diffracted at the entry point and at the exit point
        absorption_length: sum of path lengths times mu_eff / mu: the intensity of the back-surface spot is
            about exp(-mu * absorption_length) times the one of the front-surface spot (mu: photoelectric
            absorption coefficient at the energy of the reflection), not counting the reflectivities
        or None if the reflection does not leave the front surface (uf.n <= 0) or no flow reaches the back surface
    """
    uf = np.asarray(uf, dtype=float) / np.linalg.norm(uf)
    ui = np.asarray(ui, dtype=float) / np.linalg.norm(ui)
    normal = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
    if np.dot(uf, normal) <= 0 or np.dot(ui, normal) >= 0:
        return None
    t, q, theta = scattering_frame(uf, ui)
    alpha_in = _best_alpha(theta, t, q, normal, C, eps, downwards=True)
    alpha_out = _best_alpha(theta, t, q, normal, C, eps, downwards=False)
    if alpha_in is None or alpha_out is None:
        return None
    s_in, s_out = flow_direction(alpha_in, t, q), flow_direction(alpha_out, t, q)
    length_in = thickness / -np.dot(s_in, normal)
    length_out = thickness / np.dot(s_out, normal)
    exit_point = length_in * s_in + length_out * s_out
    # exit point in the scattering plane (span of ui and uf): source equivalent on the incoming beam
    (offset, _), *_ = np.linalg.lstsq(np.array([ui, uf]).T, exit_point, rcond=None)
    separation = np.linalg.norm(exit_point - np.dot(exit_point, uf) * uf)
    absorption = (length_in * relative_absorption(alpha_in, theta, C, eps)
                  + length_out * relative_absorption(alpha_out, theta, C, eps))
    return {"alpha_in": alpha_in / DEG, "alpha_out": alpha_out / DEG, "exit": exit_point, "offset": float(offset),
            "separation": float(separation), "absorption_length": float(absorption)}


def back_surface_spots(twicetheta, chi, hkl, calib, thickness, sampletilt=40.0, lattice=SI_LATTICE,
                       positions=DIAMOND_POSITIONS, weights=None, debye_waller_B=SI_DEBYE_WALLER_B,
                       eps=None, polarization="auto", nb_bounces=1, pixelsize=165.0 / 2048):
    """detector positions of the back-surface spots of Laue spots of a thick perfect crystal slab

    :param twicetheta, chi: angles (deg) of the Laue spots (arrays), e.g. from a simulation or an indexation
    :param hkl: Miller indices of the reflections (array (n, 3)), used for eps
    :param calib: detector calibration (dd, xcen, ycen, xbet, xgam), point I on the front surface of the slab
    :param thickness: slab thickness (mm)
    :param sampletilt: angle (deg) between the incoming beam and the sample surface
    :param lattice, positions, weights, debye_waller_B: crystal for eps (default: silicon), see
        epsilon_from_positions()
    :param eps: |chi_h| / |chi_0|, the same for all reflections, instead of computing it from the crystal
    :param polarization: see polarization_factor()
    :param nb_bounces: spots of the 1st to nb_bounces-th round trips in the slab (multiple Bragg-Bragg case)
    :return: dict of arrays (one row per spot): xcam, ycam (spots of the front surface), offset (mm, along the beam,
        of the 1st back-surface spot), xback, yback (shape (n, nb_bounces)), alpha_in, alpha_out (deg), eps,
        C, separation (mm), absorption_length (mm, see back_surface_path()); nan where there is no
        back-surface spot
    """
    twicetheta, chi = np.atleast_1d(twicetheta).astype(float), np.atleast_1d(chi).astype(float)
    hkl = np.atleast_2d(hkl)
    normal = surface_normal(sampletilt)
    uflab = LG.uflab_from2thetachi(twicetheta, chi, verbose=0)
    xcam, ycam, _ = LG.calc_xycam(uflab.copy(), calib, pixelsize=pixelsize)
    n = len(twicetheta)
    out = {name: np.full(n, np.nan) for name in ("offset", "alpha_in", "alpha_out", "eps", "C", "separation",
                                                 "absorption_length")}
    out["xback"], out["yback"] = np.full((n, nb_bounces), np.nan), np.full((n, nb_bounces), np.nan)
    out["xcam"], out["ycam"] = xcam, ycam
    for k in range(n):
        C, _ = polarization_factor(uflab[k], mode=polarization)
        if eps is None:
            e = epsilon_from_positions(hkl[k], positions, weights, dspacing(hkl[k], lattice), debye_waller_B)
        else:
            e = eps
        path = back_surface_path(uflab[k], thickness, normal, e, C)
        if path is None:
            continue
        out["eps"][k], out["C"][k] = e, C
        for name in ("offset", "alpha_in", "alpha_out", "separation", "absorption_length"):
            out[name][k] = path[name]
        for bounce in range(nb_bounces):  # each round trip adds the same displacement (parallel slab)
            x, y, _ = LG.calc_xycam(uflab[k:k + 1].copy(), calib, offset=(bounce + 1) * path["offset"],
                                    pixelsize=pixelsize)
            out["xback"][k, bounce], out["yback"][k, bounce] = x[0], y[0]
    return out


# --- -------------------- Laue pattern of a wafer and plot on an image
def ub_from_fitfile(fitfile, grain=0):
    """UB matrix of one grain of a .fit file and the detector settings written in the file

    :return: UB (q = UB B0 G*, LaueTools frame), calib (dd, xcen, ycen, xbet, xgam), dict with the 'CCDLabel'
        and 'pixelsize' of the file when it gives them
    """
    import ast
    from . import IOLaueTools as IOLT
    result = IOLT.readfile_fit(fitfile)
    UB = np.array(result[3][grain], dtype=float).reshape(3, 3)
    calib = [float(v) for v in np.ravel(result[5][grain])[:5]]
    comments = IOLT.readfitfile_comments(fitfile)
    if "detectorparameters" in comments:  # '#DetectorParameters' block (files of DetectorCalibration and others)
        calib = [float(v) for v in ast.literal_eval(comments["detectorparameters"])][:5]
    if not any(calib):
        raise ValueError("%s: no detector calibration found" % fitfile)
    info = {}
    if "CCDLabel" in comments:
        info["CCDLabel"] = comments["CCDLabel"]
    if "pixelsize" in comments:
        info["pixelsize"] = float(comments["pixelsize"])
    return UB, calib, info


def simulate_back_surface_spots(UBmat, calib, thickness, emin=5.0, emax=22.0, material="Si", sampletilt=40.0,
                                pixelsize=None, framedim=None, ccdlabel="sCMOS", positions=DIAMOND_POSITIONS,
                                weights=None, debye_waller_B=SI_DEBYE_WALLER_B, eps=None, polarization="auto",
                                nb_bounces=1, dictmaterials=None):
    """Laue spots of a perfect crystal slab (e.g. Si wafer) and their back-surface spots

    The harmonics (n h n k n l at n times the energy) are kept as separate spots: they share the position of the
    front-surface spot but each one has its own back-surface spot.

    :param UBmat: orientation matrix of the slab (q = UB B0 G*, LaueTools frame, as in .fit files)
    :param calib: detector calibration (dd, xcen, ycen, xbet, xgam)
    :param thickness: slab thickness (mm)
    :param emin, emax: energy band (keV)
    :param material: key of the materials library (lattice parameters and extinctions)
    :param pixelsize, framedim: detector pixel size (mm) and frame dimensions, default: those of ccdlabel
    :param positions, weights, debye_waller_B, eps, polarization, nb_bounces: see back_surface_spots()
        (default positions: diamond structure, for Si, Ge, C)
    :return: dict of arrays: hkl, energy, twicetheta, chi, xcam, ycam (front-surface spots), xback, yback
        (n, nb_bounces), offset, alpha_in, alpha_out, eps, C, separation, absorption_length; spots outside the
        detector frame removed
    """
    from . import lauecore as LAUE
    from . import dict_LaueTools as DictLT
    if dictmaterials is None:
        dictmaterials = DictLT.dict_Materials
    if pixelsize is None or framedim is None:
        ccd = DictLT.dict_CCD[ccdlabel]
        pixelsize = ccd[1] if pixelsize is None else pixelsize
        framedim = ccd[0] if framedim is None else framedim
    lattice = dictmaterials[material][1]
    tth, chi, hkl, x, y, energy = LAUE.SimulateLaue_full_np([None, None, np.asarray(UBmat, dtype=float), material],
                                                            emin, emax, calib, removeharmonics=0,
                                                            pixelsize=pixelsize, dim=framedim,
                                                            dictmaterials=dictmaterials)
    hkl = np.asarray(hkl).astype(int)
    inside = (x >= 0) & (x <= framedim[1]) & (y >= 0) & (y <= framedim[0])
    tth, chi, hkl, energy = tth[inside], chi[inside], hkl[inside], np.asarray(energy)[inside]
    out = back_surface_spots(tth, chi, hkl, calib, thickness, sampletilt=sampletilt, lattice=lattice,
                             positions=positions, weights=weights, debye_waller_B=debye_waller_B, eps=eps,
                             polarization=polarization, nb_bounces=nb_bounces, pixelsize=pixelsize)
    out.update(hkl=hkl, energy=energy, twicetheta=tth, chi=chi)
    return out


def plot_back_surface_spots(image, UBmat, calib, thickness, emin=5.0, emax=22.0, material="Si",
                            ccdlabel="sCMOS", sampletilt=40.0, nb_bounces=1, pixelsize=None, framedim=None,
                            annotate=False, percentiles=(50.0, 99.8), ax=None, savefile=None, **kwargs):
    """plot an image with the Laue spots of a perfect crystal slab (circles) and the predicted elongations
    towards their back-surface spots (segments, colored by energy; one cross per round trip in the slab)

        import LaueTools.dynamicaldiffraction as DD
        UB, calib, info = DD.ub_from_fitfile("Si_wafer.fit")
        fig, ax, spots = DD.plot_back_surface_spots("image_0001.tif", UB, calib, thickness=0.725,
                                                    ccdlabel=info["CCDLabel"], pixelsize=info["pixelsize"],
                                                    emax=22, savefile="Si_backsurface.png")

    :param image: image file (read with IOimagefile.readCCDimage(ccdlabel)) or 2D array
    :param UBmat, calib, thickness, emin, emax, material, sampletilt, nb_bounces: see
        simulate_back_surface_spots(); calib: calibration of the detector for this image
    :param annotate: write hkl and energy next to the spots
    :param percentiles: intensity percentiles of the image for the color scale (log scale)
    :param ax: matplotlib axes to draw in, default: new figure
    :param savefile: save the figure to this file
    :param kwargs: other arguments of simulate_back_surface_spots() (eps, debye_waller_B, positions...)
    :return: figure, axes, dict of simulate_back_surface_spots()
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if isinstance(image, str):
        from . import IOimagefile as IOimage
        image = IOimage.readCCDimage(image, CCDLabel=ccdlabel)[0]
    image = np.asarray(image, dtype=float)
    if framedim is None:
        framedim = image.shape
    spots = simulate_back_surface_spots(UBmat, calib, thickness, emin, emax, material, sampletilt,
                                        pixelsize=pixelsize, framedim=framedim, ccdlabel=ccdlabel,
                                        nb_bounces=nb_bounces, **kwargs)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
    else:
        fig = ax.figure
    shown = np.log10(np.clip(image - np.percentile(image, percentiles[0]), 1.0, None))
    ax.imshow(shown, cmap="gray", vmin=0, vmax=max(np.percentile(shown, percentiles[1]), 1e-6))

    # LaueTools pixel positions start at 1: pixel (x, y) is drawn at (x - 1, y - 1) on the image
    x0, y0 = spots["xcam"] - 1, spots["ycam"] - 1
    ok = ~np.isnan(spots["xback"][:, -1])
    xb, yb = spots["xback"] - 1, spots["yback"] - 1
    norm = plt.Normalize(emin, emax)
    segments = [[(x0[k], y0[k]), (xb[k, -1], yb[k, -1])] for k in np.flatnonzero(ok)]
    lines = LineCollection(segments, cmap="plasma", norm=norm, linewidths=1.5, alpha=0.9)
    lines.set_array(spots["energy"][ok])
    ax.add_collection(lines)
    for bounce in range(nb_bounces):
        ax.scatter(xb[ok, bounce], yb[ok, bounce], c=spots["energy"][ok], cmap="plasma", norm=norm, marker="x",
                   s=25, linewidths=1)
    ax.scatter(x0, y0, s=80, facecolors="none", edgecolors="cyan", linewidths=0.8)
    if annotate:
        seen = set()
        for k in range(len(x0)):
            key = (round(x0[k]), round(y0[k]))
            label = "%s %.1f keV" % ("".join(str(i) for i in spots["hkl"][k]), spots["energy"][k])
            ax.annotate(label, (x0[k], y0[k]), xytext=(6, -6 - 8 * (key in seen)), textcoords="offset points",
                        color="cyan", fontsize=6)
            seen.add(key)
    fig.colorbar(lines, ax=ax, shrink=0.7, label="energy (keV)")
    ax.set_xlim(0, framedim[1])
    ax.set_ylim(framedim[0], 0)
    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("Y (pixel)")
    ax.set_title("%s slab %.3f mm: Laue spots (circles), back-surface spots (crosses)" % (material, thickness),
                 fontsize=10)
    if savefile:
        fig.savefig(savefile, dpi=150, bbox_inches="tight")
    return fig, ax, spots


def _peak_brightness(image, x, y, half=3, background=15):
    """maximum of the image around LaueTools pixel (x, y) above the median of a larger box"""
    cx, cy = int(round(x)) - 1, int(round(y)) - 1
    near = image[max(cy - half, 0):cy + half + 1, max(cx - half, 0):cx + half + 1]
    around = image[max(cy - background, 0):cy + background + 1, max(cx - background, 0):cx + background + 1]
    return float(near.max() - np.median(around)) if near.size else -np.inf


def plot_back_surface_thumbnails(image, spots, indices=None, nb=16, boxsize=40, ncols=4, ccdlabel="sCMOS",
                                 savefile=None):
    """zoomed views of the image around Laue spots, with the predicted segment towards the back-surface spots

    :param image: image file (read with IOimagefile.readCCDimage(ccdlabel)) or 2D array
    :param spots: dict returned by simulate_back_surface_spots() or plot_back_surface_spots()
    :param indices: indices of the spots to show, default: the nb brightest spots of the image having a
        back-surface spot (one per position: harmonics share the front-surface spot)
    :param boxsize: half width (pixels) of the views
    :return: figure, list of the indices shown
    """
    import matplotlib.pyplot as plt

    if isinstance(image, str):
        from . import IOimagefile as IOimage
        image = IOimage.readCCDimage(image, CCDLabel=ccdlabel)[0]
    image = np.asarray(image, dtype=float)
    if indices is None:
        ok = np.flatnonzero(~np.isnan(spots["xback"][:, 0]))
        brightness = np.array([_peak_brightness(image, spots["xcam"][k], spots["ycam"][k]) for k in ok])
        indices, seen = [], set()
        for k in ok[np.argsort(-brightness)]:
            key = (round(spots["xcam"][k]), round(spots["ycam"][k]))
            if key not in seen:
                seen.add(key)
                indices.append(int(k))
            if len(indices) == nb:
                break
    nrows = max(1, int(np.ceil(len(indices) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for ax in axes.ravel()[len(indices):]:
        ax.set_axis_off()
    for ax, k in zip(axes.ravel(), indices):
        # pixel (x, y) of LaueTools is the image element [y - 1, x - 1]
        x0, y0 = spots["xcam"][k] - 1, spots["ycam"][k] - 1
        yb = spots["yback"][k] - 1
        cx, cy = int(round(x0)), int(round(y0))
        ylo, xlo = max(cy - boxsize, 0), max(cx - boxsize, 0)
        view = image[ylo:cy + boxsize + 1, xlo:cx + boxsize + 1]
        view = np.log10(np.clip(view - np.percentile(view, 10), 1.0, None))
        ax.imshow(view, cmap="viridis", extent=(xlo - 0.5, xlo + view.shape[1] - 0.5,
                                                ylo + view.shape[0] - 0.5, ylo - 0.5))
        same = np.flatnonzero((np.abs(spots["xcam"] - spots["xcam"][k]) < 0.5)
                              & (np.abs(spots["ycam"] - spots["ycam"][k]) < 0.5)
                              & ~np.isnan(spots["xback"][:, 0]))  # harmonics of the same spot
        for j in same:
            xs, ys = spots["xback"][j] - 1, spots["yback"][j] - 1
            ax.plot([x0, xs[-1]], [y0, ys[-1]], "-", color="w", lw=0.8, alpha=0.8)
            ax.plot(xs, ys, "x", color="r" if j == k else "orange", ms=6, mew=1.2)
        ax.plot(x0, y0, "o", mfc="none", mec="w", ms=8)
        ax.set_title("%s  %.1f keV  dY %.0f px" % ("".join(str(i) for i in spots["hkl"][k]), spots["energy"][k],
                                                 yb[0] - y0), fontsize=8)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    if savefile:
        fig.savefig(savefile, dpi=150, bbox_inches="tight")
    return fig, indices
