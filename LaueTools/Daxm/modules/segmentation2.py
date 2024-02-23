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
from skimage.segmentation import random_walker

from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground

import photutils
print('location photutils', photutils)

# -------------------------------- Thresholding --------------------------------
def apply_threshold(img, max_size=100, min_size=3, thr=20, erode=2, dilate=2):

    bkg = calc_background(img, max_size, min_size, thr) + thr

    mask = img > bkg + thr

    mask = morphology.remove_small_objects(mask, min_size**2)#, in_place=True)

    for i in range(erode):
        mask = morphology.binary_erosion(mask)

    for i in range(dilate):
        mask = morphology.binary_dilation(mask, morphology.disk(2))

    return mask, bkg


def calc_background(img, max_size=100, min_size=3, thr=20, erode=2, dilate=2):

    mask = np.zeros(img.shape, dtype=np.int32)
    bkg = np.zeros(img.shape)

    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = MedianBackground()

    bsize = range(max_size, 10, -20)

    for i, s in enumerate(bsize):
        tmp = Background2D(img - bkg, (s, s), filter_size=(3, 3),
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, mask=mask)
        bkg = bkg + tmp.background
        mask = (img - bkg) > thr
        for i in range(erode):
            mask = morphology.binary_erosion(mask)
        mask = morphology.remove_small_objects(mask, min_size) #, in_place=True)
        for i in range(dilate):
            mask = morphology.binary_dilation(mask, morphology.disk(2))

    return bkg


# -------------------------------- Bounding box and filtering -------------------------------
def draw_bbox(img, mask, merge=True):

    props = mask_to_props(img, mask)

    props = props_merge(props, merge)

    return props_to_bbox(props)


def mask_to_props(img, mask):

    props = measure.regionprops(measure.label(mask))

    bsize = [int((prop.bbox[2] - prop.bbox[0]) * (prop.bbox[3] - prop.bbox[1])) for prop in props]

    props = [props[idx] for idx in np.argsort(bsize)]

    return props


def props_merge(props, merge=False):

    new_props = []
    for i in range(len(props)):
        test = True
        bbox0 = np.array(props[i].bbox, dtype=int)
        for j in range(i+1, len(props)):
            bbox1 = np.array(props[j].bbox, dtype=int)
            if bbox0[0] >= bbox1[0] and bbox0[1] >= bbox1[1] and bbox0[2] <= bbox1[2] and bbox0[3] <= bbox1[3]:
                test = False
                break
        if test:
            new_props.append(props[i])

    props = new_props

    if merge and len(props):
        for prop in props:
            x0, y0, x1, y1 = prop.bbox
            mask[x0:x1, y0:y1] = True

        props = mask_to_props(None, mask)

    return props


def props_to_bbox(props):

    peaks_xy, peaks_hbs = [], []

    if len(props):

        peaks_xy = np.zeros((len(props), 2), dtype=int)
        peaks_hbs = np.zeros((len(props), 2), dtype=int)

        for i, prop in enumerate(props):
            x0, y0, x1, y1 = np.double(prop.bbox)
            peaks_xy[i, :] = np.floor((x0 + x1) / 2), np.floor((y0 + y1) / 2)
            peaks_hbs[i, :] = np.ceil((x1 - x0) / 2), np.ceil((y1 - y0) / 2)

    return peaks_xy, peaks_hbs

#------------------------ Peak fitting -------------------------------
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


    fifi = "/home/renversa/Research/data/blobsearch/test.tif"

    img, _, _ = rimg.readCCDimage(fifi, CCDLabel='sCMOS')
    img = np.transpose(img)


    mask = apply_threshold(img, max_size=100, min_size=3, thr=20, erode=2, dilate=3)

    peaks_xy, peaks_hbs = draw_bbox(img, mask, False)



    mplp.figure()
    mplp.imshow( mask.transpose())

    mplp.figure()
    mplp.imshow(img.transpose(), vmin=1000, vmax=2000)
    for xy, hbs in zip(peaks_xy, peaks_hbs):
        mplp.plot([xy[0]-hbs[0],  xy[0]+hbs[0], xy[0]+hbs[0], xy[0]-hbs[0], xy[0]-hbs[0]],
                  [xy[1] - hbs[1], xy[1] - hbs[1], xy[1] + hbs[1], xy[1] + hbs[1], xy[1] - hbs[1]], 'r-')


    mplp.show()






    # #
    # # Run random walker algorithm
    #
    #
    #
    #
    # mplp.figure()
    #
    # mplp.imshow(I.transpose(), vmin=1000, vmax=2000)
    #
    # mplp.figure()
    #
    # mplp.imshow(labels.transpose())
    #
    # mplp.show()
