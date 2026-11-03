"""
Back-surface Laue spots of thick perfect crystals (dynamicaldiffraction): checked against the equations and table
of Yan & Noyan, J. Appl. Phys. 98, 073527 (2005), and against a ray tracing from the exit point to the detector.
"""
import itertools

import numpy as np
import pytest

import LaueTools.dynamicaldiffraction as DD
import LaueTools.LaueGeometry as LG

DEG = np.pi / 180.0
CALIB, PIXELSIZE = [77.4, 1010.0, 1120.0, 0.4, -0.25], 0.0734  # sCMOS-like, tilted detector


def uf_vertical(theta):
    """diffracted direction for a scattering plane (y, z) containing the incoming beam y and the vertical z"""
    return np.array([0.0, np.cos(2 * theta), np.sin(2 * theta)])


@pytest.mark.parametrize("theta_deg, C, eps", [(27.167, 1, 0.918), (36.378, 1, 0.613), (40, 0.3, 0.8)])
def test_symmetric_case(theta_deg, C, eps):
    """normal = scattering vector: paper eqs. 6 (alpha_m) and 7 (separation, multiple Bragg-Bragg case)"""
    uf = uf_vertical(theta_deg * DEG)
    _, q, theta = DD.scattering_frame(uf)
    path = DD.back_surface_path(uf, 0.7, q, eps, C)
    alpha_m = DD.alpha_m_symmetric(theta, C, eps)
    assert path["alpha_in"] == pytest.approx(alpha_m / DEG, abs=1e-6)
    assert path["alpha_out"] == pytest.approx(-alpha_m / DEG, abs=1e-6)
    assert path["separation"] == pytest.approx(DD.separation_symmetric(0.7, theta, alpha_m), rel=1e-6)


def test_separation_figure7():
    """Si 004 at 12 keV, 0.7 mm thick: separation of about 3.3 mm far from the edge (paper Fig. 7a)"""
    assert DD.separation_symmetric(0.7, 22.346 * DEG, 9.2 * DEG) == pytest.approx(3.29, abs=0.01)


@pytest.mark.parametrize("theta_deg, phi_deg, eps", [(30, 5, 0.8), (30, -8, 0.9), (45, 10, 0.7)])
def test_asymmetric_case(theta_deg, phi_deg, eps):
    """surface normal in the scattering plane at phi from the scattering vector: the two roots of paper eq. 4 are
    the flow angles towards the back surface and back to the front surface"""
    uf = uf_vertical(theta_deg * DEG)
    t, q, theta = DD.scattering_frame(uf)
    phi = phi_deg * DEG
    path = DD.back_surface_path(uf, 0.5, np.cos(phi) * q - np.sin(phi) * t, eps, 1.0)
    ratio = np.tan(phi) / np.tan(theta)
    roots = [(-eps ** 2 * ratio + sign * np.sqrt(1 - eps ** 2 * (1 - ratio ** 2))) / (eps ** 2 * ratio ** 2 + 1)
             for sign in (1, -1)]
    alphas = [np.arctan(root * np.tan(theta)) / DEG for root in roots]
    assert path["alpha_in"] == pytest.approx(alphas[0], abs=1e-5)
    assert path["alpha_out"] == pytest.approx(alphas[1], abs=1e-5)


@pytest.mark.parametrize("hkl, energy, theta_table, alpha_table", [
    ((0, 0, 4), 10, 27.167, 11.5), ((0, 0, 4), 12, 22.364, 9.2), ((0, 0, 4), 15, 17.721, 7.1),
    ((0, 0, 4), 25, 10.523, 4.1), ((3, 3, 3), 10, 36.378, 30.2), ((3, 3, 3), 12, 29.621, 24.2),
    ((3, 3, 3), 15, 23.292, 18.8), ((3, 3, 3), 25, 13.724, 10.9)])
def test_table1_silicon(hkl, energy, theta_table, alpha_table):
    """Bragg angles and Poynting vector angles alpha_m of silicon (paper Table I, sigma polarization)"""
    d = DD.dspacing(hkl, DD.SI_LATTICE)
    theta = np.arcsin(12.39842 / energy / (2 * d))
    assert theta / DEG == pytest.approx(theta_table, abs=0.005)
    eps = DD.epsilon_from_positions(hkl, DD.DIAMOND_POSITIONS, dspacing=d, debye_waller_B=DD.SI_DEBYE_WALLER_B)
    assert DD.alpha_m_symmetric(theta, 1.0, eps) / DEG == pytest.approx(alpha_table, abs=0.1)


def test_epsilon_structure():
    """structure factor ratio of the diamond structure (no Debye-Waller factor)"""
    eps = lambda hkl: DD.epsilon_from_positions(hkl, DD.DIAMOND_POSITIONS)
    assert eps((0, 0, 4)) == pytest.approx(1.0)
    assert eps((2, 2, 0)) == pytest.approx(1.0)
    assert eps((1, 1, 1)) == pytest.approx(np.sqrt(0.5))
    assert eps((3, 3, 3)) == pytest.approx(np.sqrt(0.5))
    assert eps((2, 2, 2)) == pytest.approx(0.0, abs=1e-12)  # forbidden by the atom positions
    with pytest.raises(ValueError):
        DD.epsilon_from_positions((0, 0, 4), DD.DIAMOND_POSITIONS, debye_waller_B=0.5)


def test_polarization():
    """horizontal polarization: sigma for a vertical scattering plane, pi for a horizontal one"""
    assert DD.polarization_factor(uf_vertical(45 * DEG)) == (1.0, "sigma")
    C, mode = DD.polarization_factor(np.array([np.sin(60 * DEG), np.cos(60 * DEG), 0.0]))
    assert mode == "pi" and C == pytest.approx(abs(np.cos(60 * DEG)))
    with pytest.raises(ValueError):
        DD.polarization_factor(uf_vertical(45 * DEG), mode="circular")


def si001_wafer_spots(sampletilt=40.0):
    """2theta, chi, hkl of Laue spots (5-22 keV) of a Si(001) wafer, [001] along the surface normal"""
    normal = DD.surface_normal(sampletilt)
    a = np.array([1.0, 0.0, 0.0])
    R = np.array([a, np.cross(normal, a), normal]).T  # columns: crystal axes a, b, c in the lab frame
    spots = []
    for hkl in itertools.product(range(-6, 7), repeat=3):
        h = np.array(hkl)
        if not any(hkl) or DD.epsilon_from_positions(h, DD.DIAMOND_POSITIONS) < 1e-6:
            continue
        q = R @ h / np.linalg.norm(h)
        sintheta = -DD.UI @ q
        uf = DD.UI + 2 * sintheta * q
        tth = np.arccos(uf @ DD.UI) / DEG
        if sintheta <= 0 or uf[2] <= 0 or not 50 < tth < 130:
            continue
        energy = 12.39842 / (2 * DD.dspacing(h, DD.SI_LATTICE) * sintheta)
        if 5 < energy < 22:
            spots.append((tth, np.arctan2(-uf[0], uf[2]) / DEG, hkl))
    tth, chi, hkl = zip(*spots)
    return np.array(tth), np.array(chi), np.array(hkl)


def test_wafer_back_surface_spots_raytracing():
    """back-surface spots: ray from the exit point on the front surface along the diffracted direction, intersected
    with the (tilted) detector plane"""
    tth, chi, hkl = si001_wafer_spots()
    thickness = 0.5
    out = DD.back_surface_spots(tth, chi, hkl, CALIB, thickness, pixelsize=PIXELSIZE, nb_bounces=2)
    uflab = LG.uflab_from2thetachi(tth, chi, verbose=0)
    normal = DD.surface_normal(40.0)
    dd, xbet = CALIB[0], CALIB[3]
    un = np.array([0.0, np.cos(np.pi / 2 - xbet * DEG), np.sin(np.pi / 2 - xbet * DEG)])
    found = 0
    for k in range(len(tth)):
        C, _ = DD.polarization_factor(uflab[k])
        eps = DD.epsilon_from_positions(hkl[k], DD.DIAMOND_POSITIONS, dspacing=DD.dspacing(hkl[k], DD.SI_LATTICE),
                                        debye_waller_B=DD.SI_DEBYE_WALLER_B)
        path = DD.back_surface_path(uflab[k], thickness, normal, eps, C)
        if path is None:
            assert np.isnan(out["yback"][k]).all()
            assert uflab[k] @ normal <= 0  # diffracted beam does not leave the front surface
            continue
        found += 1
        exit_point = path["exit"]
        assert exit_point @ normal == pytest.approx(0.0, abs=1e-9)  # on the front surface
        # at least as far as the end of the incoming beam path in the slab
        assert path["offset"] >= thickness / np.sin(40 * DEG) - 1e-6
        for bounce in (1, 2):
            source = bounce * exit_point
            s = (dd - source @ un) / (uflab[k] @ un)
            M = source + s * uflab[k]
            x, y, _ = LG.calc_xycam(np.array([M / np.linalg.norm(M)]), CALIB, pixelsize=PIXELSIZE)
            assert out["xback"][k, bounce - 1] == pytest.approx(x[0], abs=1e-6)
            assert out["yback"][k, bounce - 1] == pytest.approx(y[0], abs=1e-6)
        # mainly along Y, towards larger Y: about offset / pixel size
        dy = out["yback"][k, 0] - out["ycam"][k]
        assert dy == pytest.approx(path["offset"] / PIXELSIZE, rel=0.05)
        assert abs(out["xback"][k, 0] - out["xcam"][k]) < 0.1 * dy
    assert found > 20


def test_fixed_eps_and_thickness_scaling():
    """offsets proportional to the thickness; eps given instead of the crystal"""
    tth, chi, hkl = si001_wafer_spots()
    thin = DD.back_surface_spots(tth, chi, hkl, CALIB, 0.25, eps=0.8, pixelsize=PIXELSIZE)
    thick = DD.back_surface_spots(tth, chi, hkl, CALIB, 0.5, eps=0.8, pixelsize=PIXELSIZE)
    ok = ~np.isnan(thick["offset"])
    assert ok.sum() > 20 and np.allclose(thin["eps"][ok], 0.8)
    assert np.allclose(thick["offset"][ok], 2 * thin["offset"][ok])
    assert np.allclose(thick["absorption_length"][ok], 2 * thin["absorption_length"][ok])


GE_IMAGE = "LaueTools/LaueImages/Ge_blanc_0000.mccd"
GE_FIT = "LaueTools/LaueImages/Ge_blanc_0000_LT_5.fit"


def test_ub_from_fitfile():
    import os
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    UB, calib, info = DD.ub_from_fitfile(os.fspath(root / GE_FIT))
    assert np.allclose(calib, [70.775, 941.736, 1082.562, 0.63, -0.682])
    assert info == {"CCDLabel": "MARCCD165", "pixelsize": 0.079142}
    assert np.allclose(UB @ UB.T, np.eye(3), atol=1e-3)  # nearly a rotation


def test_simulation_and_plots(tmp_path):
    """Laue pattern of the Ge calibration image as a 0.725 mm thick perfect slab, drawn on the image"""
    import os
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(__file__).resolve().parents[1]
    UB, calib, info = DD.ub_from_fitfile(os.fspath(root / GE_FIT))
    kwargs = dict(material="Ge", ccdlabel=info["CCDLabel"], pixelsize=info["pixelsize"], emax=22)
    spots = DD.simulate_back_surface_spots(UB, calib, 0.725, nb_bounces=2, **kwargs)
    ok = ~np.isnan(spots["offset"])
    assert len(spots["hkl"]) > 100 and ok.sum() > 100
    # harmonics kept: some positions are shared by several reflections
    positions = {(round(x, 3), round(y, 3)) for x, y in zip(spots["xcam"], spots["ycam"])}
    assert len(positions) < len(spots["hkl"])
    # on the detector: mainly along +Y, at least the end of the beam path in the slab
    dy = spots["yback"][ok, 0] - spots["ycam"][ok]
    assert (dy > 0.725 / np.sin(40 * DEG) / info["pixelsize"] * 0.9).all()
    assert (np.abs(spots["xback"][ok, 0] - spots["xcam"][ok]) < 0.05 * dy).all()
    assert np.allclose(spots["yback"][ok, 1] - spots["ycam"][ok], 2 * dy, rtol=0.05)

    fig, ax, plotted = DD.plot_back_surface_spots(os.fspath(root / GE_IMAGE), UB, calib, 0.725, annotate=True,
                                                  savefile=os.fspath(tmp_path / "pattern.png"), **kwargs)
    assert (tmp_path / "pattern.png").stat().st_size > 0 and len(plotted["hkl"]) == len(spots["hkl"])
    fig2, shown = DD.plot_back_surface_thumbnails(os.fspath(root / GE_IMAGE), plotted, nb=6,
                                                  ccdlabel=info["CCDLabel"],
                                                  savefile=os.fspath(tmp_path / "thumbs.png"))
    assert len(shown) == 6 and (tmp_path / "thumbs.png").stat().st_size > 0
    plt.close(fig)
    plt.close(fig2)
