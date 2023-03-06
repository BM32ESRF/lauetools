#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import numpy as np

import scipy.optimize as spo
import scipy.ndimage as ndimage

from skimage import filters, morphology, measure


# -------------------------------- Main functions -------------------------------
def search_blobs():
    pass


def search_peaks():
    pass


# -------------------------------- Thresholding --------------------------------
def apply_threshold(img, method='local', factor=1.0, value=100, boxsize=None):
    thr = get_threshold(img, method, boxsize)

    return img > (factor * thr + value)


def get_threshold(img, method, boxsize=None):
    if method in ('minimum', 'minimum_filter', 'min'):
        thr = minimum_filter(img, boxsize)

    elif method in ('median', 'median_filter', 'med'):
        thr = median_filter(img, boxsize)

    elif method in ('local', 'local_threshold'):
        thr = local_threshold(img, boxsize)

    else:
        thr = 0

    return thr


def local_threshold(img, block_size=101):
    if block_size is None:
        block_size = 101

    block_size = int(np.floor(block_size / 2) * 2 + 1)

    return filters.threshold_local(img, block_size=block_size, offset=0)


def minimum_filter(img, boxsize=11):
    if boxsize is None:
        boxsize = 11

    boxsize = int(np.floor(boxsize / 2) * 2 + 1)

    return ndimage.filters.minimum_filter(img, size=boxsize)


def median_filter(img, boxsize=10):
    if boxsize is None:
        boxsize = 11

    boxsize = int(np.floor(boxsize / 2) * 2 + 1)

    return ndimage.filters.median_filter(img, size=boxsize)


# -------------------------------- Bounding box and filtering -------------------------------
def draw_bbox(img, mask, min_size=3, max_size=500, dilate=6, erode=True, merge=True):
    mask = draw_bbox_dilate(mask, min_size, dilate, erode)

    props = measure.regionprops(measure.label(mask))

    props = draw_bbox_merge(mask, props, merge)

    return draw_bbox_result(props, max_size)


def draw_bbox_dilate(mask, min_size, dilate, erode=True):
    if erode:
        mask = morphology.binary_erosion(mask)

    if mask.any():
        min_area = np.round(np.pi * min_size * min_size)

        mask = morphology.remove_small_objects(mask.astype(bool), min_area, in_place=True)

        for _ in range(dilate):
            mask = morphology.binary_dilation(mask, morphology.disk(2))

    return mask


def draw_bbox_merge(mask, props, merge=True):
    if merge and len(props):
        for prop in props:
            x0, y0, x1, y1 = prop.bbox
            mask[x0:x1, y0:y1] = True

    props = measure.regionprops(measure.label(mask))
    
    return props


def draw_bbox_result(props, max_size):
    peaks_xy, peaks_hbs = [], []

    if len(props):

        peaks_xy = np.zeros((len(props), 2), dtype=int)
        peaks_hbs = np.zeros((len(props), 2), dtype=int)

        for i, prop in enumerate(props):
            x0, y0, x1, y1 = np.double(prop.bbox)
            peaks_xy[i, :] = np.floor((x0 + x1) / 2), np.floor((y0 + y1) / 2)
            peaks_hbs[i, :] = np.ceil((x1 - x0) / 2), np.ceil((y1 - y0) / 2)

        keep = np.logical_and(2 * peaks_hbs[:, 0] + 1 < max_size, 2 * peaks_hbs[:, 1] + 1 < max_size)

        if len(keep):
            peaks_xy = peaks_xy[keep, :]
            peaks_hbs = peaks_hbs[keep, :]
        else:
            peaks_xy, peaks_hbs = [], []

    return peaks_xy, peaks_hbs


# -------------------------------- Peak fitting -------------------------------
def blob_fit(img, xy, hbs, isotropic=False):
    blob, x0, y0 = roi(img, xy, hbs)

    optim_res = blob_fit_gaussian(blob, isotropic=isotropic)

    if optim_res['success']:

        params = optim_res['x']
        params = np.append(params, blob.sum())

    else:
        mu = blob_moments(blob, threshold=True)
        pedestal = blob.min()
        amplitude = mu[5] / (2 * np.pi * mu[2] * mu[3]) - pedestal

        params = mu[:4]
        params.extend([amplitude, pedestal, mu[5]])

    params[0] = params[0] + x0
    params[1] = params[1] + y0
    params[2] = params[2] * 2.355
    params[3] = params[3] * 2.355
    params[4] = params[4] % np.pi
    params[7] = params[7] - params[6] * blob.size

    return params


def blob_moments(blob, x0=0, y0=0, threshold=True):
    if threshold:
        thr = np.percentile(blob, 88)  # assuming normal distribution in a 6*std window
        blob = np.where(blob > thr, blob - thr, 0)

    # centre from raw moments
    mu = measure.moments(blob, order=2)
    area = mu[0, 0]
    crow, ccol = mu[1, 0] / area, mu[0, 1] / area

    # central moments
    mu20 = mu[2, 0] / mu[0, 0] - crow ** 2
    mu02 = mu[0, 2] / mu[0, 0] - ccol ** 2
    mu11 = mu[1, 1] / mu[0, 0] - crow * ccol

    xc = ccol
    yc = crow

    if np.abs(mu02 - mu20) > 1E-6:
        theta = 0.5 * np.arctan(2. * mu11 / (mu02 - mu20))
        major = 0.5 * (mu02 + mu20) + 0.5 * np.sqrt(4 * mu11 ** 2 + (mu02 - mu20) ** 2)
        minor = 0.5 * (mu02 + mu20) - 0.5 * np.sqrt(4 * mu11 ** 2 + (mu02 - mu20) ** 2)
    else:
        theta = 0.
        minor = 1.
        major = 1.

    if mu02 < mu20:
        theta = theta + 0.5 * np.pi

    result = [xc + x0, yc + y0, np.sqrt(major), np.sqrt(minor), theta, area]
    return [float(item) for item in result]


def blob_fit_gaussian(blob, params0=None, bounds=None, isotropic=False):

    if params0 is None:
        params0 = blob_moments(blob, threshold=True)
        params0.append(0)  # pedestal parameter
        params0[5] = params0[5] / (2 * np.pi * params0[2] * params0[3])

    # x_cen, y_cen, sig_maj ,sig_min, rot, amplitude, pedestal
    params0 = np.array(params0)

    if bounds is None:
        # x_cen, y_cen, sig_maj ,sig_min, rot, amplitude, pedestal
        params0 = np.array(params0)
        lb = [0., 0., 0.1, 0.1,
              -4. * np.pi, min(0., np.min(blob)), min(0., np.min(blob))]
        ub = [blob.shape[0] - 1., blob.shape[1] - 1., blob.shape[0], blob.shape[1],
              4. * np.pi, 2 * max(params0[5], np.max(blob)), np.max(blob)]

        # to be safe
        lb = np.minimum(lb, params0)
        ub = np.maximum(ub, 1.5*params0)
    else:
        lb, ub = bounds

    xmesh, ymesh = np.meshgrid(np.arange(blob.shape[0]), np.arange(blob.shape[1]), indexing='ij')

    if isotropic:
        idx = [0, 1, 2, 5, 6]
        params0, lb, ub = params0[idx], lb[idx], ub[idx]

        def errfun(params):
            return np.ravel(gaussian2d_iso(*params)(xmesh, ymesh) - blob)
    else:
        def errfun(params):
            return np.ravel(gaussian2d(*params)(xmesh, ymesh) - blob)

    result = spo.least_squares(errfun, params0, bounds=(lb, ub))

    if isotropic and result['success']:
        params = result['x']
        params = [params[0], params[1], params[2], params[2], 0, params[3], params[4]]
        result['x'] = params

    return result


# -------------------------------- utils -------------------------------
def roi(img, xy, hbs):
    x0 = max(xy[0] - hbs[0], 0)
    y0 = max(xy[1] - hbs[1], 0)
    x1 = min(xy[0] + hbs[0] + 1, img.shape[0])
    y1 = min(xy[1] + hbs[1] + 1, img.shape[1])

    return np.array(img[x0:x1, y0:y1], dtype=np.float), x0, y0


def gaussian2d(x_cen, y_cen, sigma_maj, sigma_min, theta, amplitude, pedestal):
    # pre-compute rotated centre
    rcen_x = x_cen * np.cos(theta) - y_cen * np.sin(theta)
    rcen_y = x_cen * np.sin(theta) + y_cen * np.cos(theta)

    def fun(x, y):
        xp = rcen_x - (x * np.cos(theta) - y * np.sin(theta))
        yp = rcen_y - (x * np.sin(theta) + y * np.cos(theta))
        return amplitude * np.exp(-0.5 * ((xp / sigma_maj) ** 2 + (yp / sigma_min) ** 2)) + pedestal

    return fun


def gaussian2d_iso(x_cen, y_cen, sigma, amplitude, pedestal):
    # pre-compute rotated centre

    def fun(x, y):
        xp = x - x_cen
        yp = y - y_cen
        return amplitude * np.exp(-0.5 * (xp**2 + yp**2)/sigma**2) + pedestal

    return fun


def vprint(msg, verbose=False):
    if verbose:
        print(msg)


# end module


if __name__ == "__main__":

    import readmccd as rimg
    import matplotlib.pyplot as mplp

    fifi = "/home/renversa/workspace/MyLaueTools/Examples/ZrO2/ZrO2_grains_1040.mccd"

    testimg, _, _ = rimg.readCCDimage(fifi, CCDLabel='MARCCD165')
    testimg = np.transpose(testimg)

    testmask = apply_threshold(testimg, method='median', factor=1.01, value=50)

    peakXY, peakHbs = draw_bbox(testimg, testmask, min_size=4, max_size=50, dilate=2, merge=False, erode=False)

    mplp.figure()
    mplp.imshow(np.transpose(testimg), vmin=100, vmax=1000)

    ang = np.radians(np.arange(361.))

    testi = 0
    for p, h in zip(peakXY, peakHbs):
        print (testi)
        testi = testi + 1
        testparams = blob_fit(testimg, p, h)
        # x0, y0, a, b, theta, A, B, area
        Xell = testparams[0] + testparams[2] / 2 * np.cos(testparams[4]) * np.cos(ang) + testparams[3] / 2 * np.sin(testparams[4]) * np.sin(
            ang)
        Yell = testparams[1] - testparams[2] / 2 * np.sin(testparams[4]) * np.cos(ang) + testparams[3] / 2 * np.cos(testparams[4]) * np.sin(
            ang)

        mplp.plot(Xell, Yell, 'r')

    mplp.show()
