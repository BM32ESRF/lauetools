"""
Source of scattering displaced along the incoming beam (grain in depth, parallax):
LaueGeometry.calc_uflab(offset), calc_xycam(offset) and pixelshift_from_sourceoffset()
checked against an independent ray tracing (rays from the displaced source to the detector plane).
"""
import numpy as np

import LaueTools.LaueGeometry as LG

DEG = np.pi / 180.
KI = np.array([0., 1., 0.])   # incoming beam in the frame of calc_uflab() and calc_xycam()

# detector calibrations (dd, xcen, ycen, xbet, xgam), pixelsize, frame (nb pixels X, Y)
DETECTORS = {'sCMOS': ([77.4, 1010., 1120., 0.4, -0.25], 0.0734, (2018, 2016)),
             'EIGER_4MCdTe': ([80., 1034., 1081., -0.6, 0.5], 0.075, (2068, 2162)),
             'no tilt': ([70., 1000., 1000., 0., 0.], 0.08, (2048, 2048))}
OFFSETS = (0.75, 0.05, -0.2)   # mm


def pixels_grid(frame, step=250):
    """pixels over the whole detector: 2theta from about 45 to 135 deg, chi from about -45 to 45 deg"""
    x, y = np.meshgrid(np.arange(20., frame[0], step), np.arange(20., frame[1], step))
    return x.ravel(), y.ravel()


def raytracing_pixels(source, uf, calib, pixelsize):
    """pixels hit by rays of unit vectors uf from source: intersection M with the detector plane
    (point dd * un, normal un), then pixel of the direction IM seen from the origin I (no offset)"""
    dd, xbet = calib[0], calib[3]
    un = np.array([0., np.cos(np.pi / 2 - xbet * DEG), np.sin(np.pi / 2 - xbet * DEG)])
    t = (dd - source @ un) / (uf @ un)
    M = source + t[:, None] * uf
    x, y, _ = LG.calc_xycam(M / np.linalg.norm(M, axis=1)[:, None], calib, pixelsize=pixelsize)
    return x, y


def check_offset_along_beam(detector, offset):
    calib, pixelsize, frame = DETECTORS[detector]
    x0, y0 = pixels_grid(frame)
    tth, chi = LG.calc_uflab(x0, y0, calib, pixelsize=pixelsize, kf_direction='Z>0')
    assert chi.min() < -25 and chi.max() > 25 and tth.min() < 60 and tth.max() > 120
    uf = LG.uflab_from2thetachi(tth, chi, verbose=0)
    assert np.allclose(np.arccos(uf @ KI) / DEG, tth)   # ki along y in this frame

    # same reflections from a source at offset along the beam
    xs, ys = raytracing_pixels(offset * KI, uf, calib, pixelsize)
    # positive offset (downstream): spots move towards larger Y by about offset / pixelsize
    assert np.allclose(ys - y0, offset / pixelsize, atol=0.02 * abs(offset) / pixelsize)

    x, y, _ = LG.calc_xycam(uf.copy(), calib, offset=offset, pixelsize=pixelsize)
    assert np.allclose(x, xs, atol=1e-9) and np.allclose(y, ys, atol=1e-9)
    x, y, _ = LG.calc_xycam_from2thetachi(tth, chi, calib, offset=offset, pixelsize=pixelsize)
    assert np.allclose(x, xs, atol=1e-9) and np.allclose(y, ys, atol=1e-9)

    dx, dy = LG.pixelshift_from_sourceoffset(x0, y0, calib, offset, pixelsize=pixelsize)
    assert np.allclose(dx, xs - x0, atol=1e-9) and np.allclose(dy, ys - y0, atol=1e-9)

    # angles of the pixels really hit, computed with the source offset: the true ones
    tth2, chi2 = LG.calc_uflab(xs, ys, calib, offset=offset, pixelsize=pixelsize, kf_direction='Z>0')
    assert np.allclose(tth2, tth, atol=1e-9) and np.allclose(chi2, chi, atol=1e-9)
    # ignoring a positive offset underestimates 2theta
    tth_nooffset, _ = LG.calc_uflab(xs, ys, calib, pixelsize=pixelsize, kf_direction='Z>0')
    assert np.all(np.sign(tth - tth_nooffset) == np.sign(offset))


def test_offset_along_beam():
    for detector in DETECTORS:
        for offset in OFFSETS:
            check_offset_along_beam(detector, offset)


def test_no_tilt_is_pure_y_shift():
    calib, pixelsize, frame = DETECTORS['no tilt']
    x0, y0 = pixels_grid(frame)
    dx, dy = LG.pixelshift_from_sourceoffset(x0, y0, calib, 0.75, pixelsize=pixelsize)
    assert np.allclose(dx, 0., atol=1e-9) and np.allclose(dy, 0.75 / pixelsize, atol=1e-9)


if __name__ == '__main__':
    test_offset_along_beam()
    test_no_tilt_is_pure_y_shift()
    print('test_sourceoffset: OK')
