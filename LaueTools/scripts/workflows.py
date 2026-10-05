import numpy as np
import pandas as pd
import h5py
import fabio
import logging
import itertools
import os
import json
import sys
import time

import matplotlib.pyplot as plt
from pathlib import Path
from LaueTools import dict_LaueTools as DictLT
import LaueTools.IOimagefile as IOimage



from typing import Dict, Any, List, Optional, Tuple, Union
try:
    # Check if running in a Jupyter notebook
    get_ipython()
    from tqdm.notebook import tqdm  # Use notebook version
except NameError:
    from tqdm import tqdm  # Use standard version for scripts
#from tqdm.notebook import tqdm
import multiprocessing
from multiprocessing import Pool, cpu_count, active_children

from LaueTools.imagescollector import (
    collectpixelvalue_singlefile,
    collectroissum_singlefile,
    collectroisptp_singlefile,
    collectroiarray_singlefile)

import LaueTools.blissdatafolderstructure as bf

import os
import logging

from typing import Optional

DEFAULT_LOG_FILE = "workflow.log"

class LoggerManager:
    """Manages logging configuration for the workflows module."""

    _logger = None  # Class-level logger instance

    @classmethod
    def get_logger(cls, name: str, log_file: Optional[str] = None) -> logging.Logger:
        """
        Get or create a logger with the specified name and optional log file.

        Args:
            name: Name of the logger (typically __name__).
            log_file: Optional path to a log file. If None, logs to stderr.

        Returns:
            Configured logger instance.
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Create a formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Clear existing handlers to avoid duplicate logs
        logger.handlers.clear()

        # Add a handler (file or stderr)
        if log_file:
            # Ensure the log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir:
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except OSError as e:
                    print(f"Failed to create log directory {log_dir}: {e}")
                    log_file = None  # Fallback to stderr

            if log_file:
                try:
                    file_handler = logging.FileHandler(log_file)
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)
                except FileNotFoundError as e:
                    print(f"Failed to create log file {log_file}: {e}")
                    # Fallback to stderr
                    stderr_handler = logging.StreamHandler()
                    stderr_handler.setFormatter(formatter)
                    logger.addHandler(stderr_handler)
            else:
                # Fallback to stderr if directory creation fails
                stderr_handler = logging.StreamHandler()
                stderr_handler.setFormatter(formatter)
                logger.addHandler(stderr_handler)
        else:
            # Default: Log to stderr
            stderr_handler = logging.StreamHandler()
            stderr_handler.setFormatter(formatter)
            logger.addHandler(stderr_handler)

        return logger

# class LoggerManager:
#     """Manages logging configuration for the workflows module."""

#     _logger = None  # Class-level logger instance

#     @classmethod
#     def get_logger(cls, name: str, log_file: Optional[str] = None) -> logging.Logger:
#         """
#         Get or create a logger with the specified name and optional log file.

#         Args:
#             name: Name of the logger (typically __name__).
#             log_file: Optional path to a log file. If None, logs to stderr.

#         Returns:
#             Configured logger instance.
#         """
#         if cls._logger is None:
#             # Configure the logger
#             logger = logging.getLogger(name)
#             logger.setLevel(logging.INFO)

#             # Create a formatter
#             formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

#             # Add a handler (file or stderr)
#             if log_file:
#                 # Ensure the log directory exists
#                 log_dir = os.path.dirname(log_file)
#                 if log_dir and not os.path.exists(log_dir):
#                     try:
#                         os.makedirs(log_dir, exist_ok=True)
#                     except OSError as e:
#                         print(f"Failed to create log directory {log_dir}: {e}")
#                         log_file = None  # Fallback to stderr

#                 if log_file:
#                     file_handler = logging.FileHandler(log_file)
#                     file_handler.setFormatter(formatter)
#                     logger.addHandler(file_handler)
#                 else:
#                     # Fallback to stderr if file logging fails
#                     stderr_handler = logging.StreamHandler()
#                     stderr_handler.setFormatter(formatter)
#                     logger.addHandler(stderr_handler)
#             else:
#                 # Default: Log to stderr
#                 stderr_handler = logging.StreamHandler()
#                 stderr_handler.setFormatter(formatter)
#                 logger.addHandler(stderr_handler)

#             cls._logger = logger

#         return cls._logger

# # Define the log file path
# log_file = "pixel_monitoring.log"
# log_dir = os.path.dirname(log_file)

# # Create the directory if it doesn't exist
# if log_dir and not os.path.exists(log_dir):
#     os.makedirs(log_dir, exist_ok=True)

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     filename=log_file
# )
# logger = logging.getLogger(__name__)




# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

class ConfigManager:
    """Manages configuration parameters for the experiment."""

    def __init__(
        self,
        exp_id: str,
        data_at_esrf: bool = True,
        hdf5_logfile_exists: bool = True,
        ccd_label: str = "EIGER_4MCdTe",
        on_linux: bool = True,
        jupyter_lab: bool = True,
    ):
        self.exp_id = exp_id
        self.data_at_esrf = data_at_esrf
        self.hdf5_logfile_exists = hdf5_logfile_exists
        self.ccd_label = ccd_label
        self.on_linux = on_linux
        self.jupyter_lab = jupyter_lab
        
        #self.logger = LoggerManager.get_logger(__name__, log_file="workflow.log")
        self.experiment_folder = self._set_experiment_folder()

    def _set_experiment_folder(self) -> str:
        """Set the experiment folder path based on configuration."""
        if self.data_at_esrf:
            nice_folder = "visitor"
            experiment_folder = os.path.join("/data", nice_folder, f"{self.exp_id}/bm32/")
            if os.path.exists(experiment_folder):
                experiment_folder = bf.setExperimentFolder_with_date(experiment_folder)
                #self.logger.info(f"Experiment folder set to: {experiment_folder}")
            else:
                raise FileNotFoundError(f"Experiment folder not found: {experiment_folder}")
        else:
            experiment_folder = "/my/folder/to/data"
        return experiment_folder


def _collect_roi_array_single_file_wrapper(args):
    """Wrapper function to unpack args for _collect_roi_array_single_file."""
    return MosaicWorkflow._collect_roi_array_single_file(*args)

class MosaicWorkflow:
    """Handles mosaic-related workflows for 2D maps."""

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.roicenter = params.get("roicenter")
        self.boxsize_X = params.get("boxsize_X", 19)
        self.boxsize_Y = params.get("boxsize_Y", 19)
        self.collector = params.get("collector", "mosaic")
        #self.logger = LoggerManager.get_logger(__name__, log_file="workflow.log")

    def validate_roicenter(self) -> bool:
        """Validate that the ROI center is within detector bounds."""
        if self.roicenter is None:
            #self.logger.error("ROI center is not set.")
            return False
        x_roi, y_roi = self.roicenter
        # framedim = (nb lines, nb columns) i.e. (along Y, along X)
        dim_Y, dim_X = DictLT.dict_CCD[self.params.get("CCDLabel")][0]
        if (
            x_roi - self.boxsize_X < 0
            or x_roi + self.boxsize_X >= dim_X
            or y_roi - self.boxsize_Y < 0
            or y_roi + self.boxsize_Y >= dim_Y
        ):
            # self.logger.error(
            #     f"ROI center {self.roicenter} is too close to the detector border for boxsize: {(self.boxsize_X, self.boxsize_Y)}"
            # )
            return False
        return True

    def run_mosaic_workflow(
        self, nb_cpus: int = 64, list_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Run the mosaic workflow to collect pixel intensities.
        """
        if not self.validate_roicenter():
            raise ValueError("Invalid ROI center or boxsize.")
    
        folder = self.params.get("folder")
        prefix = self.params.get("prefix")
        ccd_label = self.params.get("CCDLabel")
    
        if list_indices is None:
            list_indices = self.params.get("listindices")
    
        nb_images = len(list_indices)
        max_nb_cpus = cpu_count()
        nb_cpus = min(nb_cpus, max_nb_cpus)
        #self.logger.info(f"Using {nb_cpus} CPUs for {nb_images} images.")
    
        # Prepare arguments for multiprocessing
        args_mosaic = zip(
            list_indices,
            itertools.repeat(self.roicenter),
            itertools.repeat(prefix),
            itertools.repeat(folder),
            itertools.repeat(self.boxsize_X),
            itertools.repeat(self.boxsize_Y),
            itertools.repeat(ccd_label),
            itertools.repeat(self.params.get("nbframes_per_file", 1)),
            itertools.repeat(self.params.get("nbimages_recorded", None)),
        )
    
        # Use the module-level wrapper function
        with Pool(nb_cpus) as pool:
            all_results = list(
                tqdm(
                    pool.imap(
                        _collect_roi_array_single_file_wrapper,  # Use the module-level wrapper
                        args_mosaic,
                    ),
                    total=nb_images,
                    desc="Mosaic collection progress",
                )
            )
    
        self.all_results = np.array(all_results)
        return self.all_results
    
    def _arrange2D(self):
        """rerrange all_results in 2D

        return: mosaic (2D array of small extracted images), bigimage (single image of the mosaic of all extracted small images)"""
        if self.params.get("scantype") == 'map':
            return arrange_mosaic2D(self.all_results, self.params.get("mapdimensions"))

    @staticmethod
    def _collect_roi_array_single_file(
        index: int,
        roicenter: Tuple[int, int],
        prefix: str,
        folder: str,
        boxsize_X: int,
        boxsize_Y: int,
        ccd_label: str,
        nbframes_per_file: int = 1,
        nbimages_recorded: Optional[int] = None,
    ) -> np.ndarray:
        """Helper function to collect ROI array for a single file
        listindices,
                   itertools.repeat(roicenter),
                   itertools.repeat(prefix),
                   listfolders,
                  itertools.repeat(boxsize_X),
                  itertools.repeat(boxsize_Y),
                   itertools.repeat(CCDLabel)"""
        # missing or unreadable image (e.g. not completed scan or file being written): blank ROI
        blank = np.full((2 * boxsize_Y + 1, 2 * boxsize_X + 1), np.nan)
        # image not yet recorded in stacked images files (full of zeros): not read
        if nbimages_recorded is not None and index >= nbimages_recorded:
            return blank
        try:
            roi = collectroiarray_singlefile(index,roicenter,prefix, folder, boxsize_X, boxsize_Y, ccd_label,
                                             nbframes_per_file=nbframes_per_file)
        except Exception as exc:
            print(f'cannot read ROI in image index {index} in {folder}: {exc}')
            return blank
        if roi is None or roi.shape != blank.shape:
            return blank
        return roi


def arrange_mosaic2D(rois: np.ndarray, mapdimensions) -> Dict[str, Any]:
    """arrange ROIs pixel intensities arrays collected over the images of a 2D map

    :param rois: array (nb images, 2*boxsize_Y+1, 2*boxsize_X+1), images in map order (line by line)
    :param mapdimensions: (nb images along fast axis, nb images along slow axis)
    :return: dict with keys 'mosaicdata' (array (dimslow, dimfast, 2*boxsize_Y+1, 2*boxsize_X+1)),
        'singleimage' (single 2D array of all up-down flipped ROIs arrays) and 'mosaic_shape'
    """
    dimfast, dimslow = mapdimensions
    _, boxY, boxX = rois.shape
    # NaN (blank) for map positions without image (e.g. not completed scan)
    mosaic = np.full((dimslow * dimfast, boxY, boxX), np.nan)
    nbimages = min(len(rois), dimslow * dimfast)
    mosaic[:nbimages] = rois[:nbimages]
    mosaic = mosaic.reshape((dimslow, dimfast, boxY, boxX))
    bigimage = mosaic[:, :, ::-1, :].transpose((0, 2, 1, 3)).reshape((dimslow * boxY, dimfast * boxX))
    return {'mosaicdata': mosaic, 'singleimage': bigimage, 'mosaic_shape': mosaic.shape}


def counter_map2D(values: np.ndarray, mapdimensions) -> np.ndarray:
    """arrange values (first axis: images in map order, line by line) in 2D map
    of shape (dimslow, dimfast) + values.shape[1:] (NaN for map positions without image)"""
    dimfast, dimslow = mapdimensions
    values = np.asarray(values, dtype=float)
    out = np.full((dimslow * dimfast,) + values.shape[1:], np.nan)
    nbimages = min(len(values), dimslow * dimfast)
    out[:nbimages] = values[:nbimages]
    return out.reshape((dimslow, dimfast) + values.shape[1:])


# ============================================================================
# ROI COUNTERS WORKFLOW (pixel intensities monitoring over a set of images)
# ============================================================================

# 2D gaussian fit results (see readmccd.fitPeakMultiROIs()). X, Y are those of mosaic.FitPeakOnMap()
# nfev: nb of function evaluations, startbaseline: minimum intensity in ROI
FIT_COLUMNS = ('background', 'amplitude', 'X', 'Y', 'sigma1', 'sigma2', 'angle', 'nfev', 'startbaseline')

# nb of values per ROI of each counter (None: ROI pixel intensities array)
ROI_COUNTERS = {
    'mosaic': None,  # ROI pixel intensities array of shape (2*boxsize_Y+1, 2*boxsize_X+1)
    'mean': 1,
    'max': 1,
    'ptp': 1,  # peak to peak (max - min)
    'sum': 1,
    'pixelval': 1,  # intensity of ROI center pixel
    'XYmax': 2,  # X, Y pixel position of highest intensity
    'XYcentroid': 2,  # X, Y center of mass of intensity above ROI minimum intensity
    'fit': len(FIT_COLUMNS),
}

# starting values of 2D peak fit (same keys as mosaic.DEFAULT_DICTfittingparameters)
DEFAULT_FIT_PARAMETERS = {'modelFunction': "gaussian", 'positionStart': "max", 'peaksizeStart': 4}

# detectors for which IOimage.readrectangle_in_image() reads the whole image: ROIs are taken in
# the image read once
FULLIMAGE_CCDLABELS = ('EIGER_4MCdTe', 'EIGER_1M', 'EIGER_4MCdTestack')


def image_filepath(params: Dict[str, Any], imageindex: int) -> Tuple[str, int]:
    """return full path of image file of image index imageindex and frame index in this file
    (-1 for single image file)

    params keys: folder, CCDLabel, filename_representative (any image filename of the series) or
    prefix and suffix, optional nbdigits (default 4), nbframes_per_file (stacked images files)
    """
    CCDLabel = params['CCDLabel']
    nbdigits = int(params.get('nbdigits', 4))
    representative = params.get('filename_representative')
    if representative is None:
        representative = f"{params['prefix']}{0:0{nbdigits}d}.{params['suffix'].lstrip('.')}"
    if CCDLabel in IOimage.STACK_CCDLABELS:
        filename, frameindex = IOimage.stack_file_and_frame(representative, int(imageindex),
                                                    int(params.get('nbframes_per_file', 1)))
    else:
        filename = IOimage.setfilename(representative, int(imageindex), CCDLabel=CCDLabel,
                                                                            nbdigits=nbdigits)
        frameindex = -1
    return os.path.join(params['folder'], filename), frameindex


def read_rois(imagepath: str, roicenters, boxsize_X: int, boxsize_Y: int, CCDLabel: str,
                                        stackimageindex: int = -1) -> List[Optional[np.ndarray]]:
    """read pixel intensities arrays of shape (2*boxsize_Y+1, 2*boxsize_X+1) centered on
    roicenters (X, Y) in an image

    :return: list of 2D arrays (None for ROI not entirely in detector frame)
    """
    framedim = DictLT.dict_CCD[CCDLabel][0]  # (nb lines, nb columns)
    shape = (2 * boxsize_Y + 1, 2 * boxsize_X + 1)
    centers = [(int(x), int(y)) for x, y in roicenters]
    inside = [boxsize_X <= x < framedim[1] - boxsize_X and boxsize_Y <= y < framedim[0] - boxsize_Y
                for x, y in centers]
    if CCDLabel in FULLIMAGE_CCDLABELS:
        data = IOimage.readCCDimage(imagepath, CCDLabel=CCDLabel, stackimageindex=stackimageindex)[0]
        crops = [data[y - boxsize_Y:y + boxsize_Y + 1, x - boxsize_X:x + boxsize_X + 1] if ok else None
                    for (x, y), ok in zip(centers, inside)]
    else:
        crops = [IOimage.readrectangle_in_image(imagepath, x, y, boxsize_X, boxsize_Y,
                                                CCDLabel=CCDLabel, stackimageindex=stackimageindex)
                    if ok else None
                    for (x, y), ok in zip(centers, inside)]
    return [np.asarray(c, dtype=np.float64) if c is not None and c.shape == shape else None
            for c in crops]


def monitor_normalization(imagepath: str, CCDLabel: str, monitoroffset: float = 0.) -> Tuple[float, float]:
    """return pedestal and monitor value to normalize image intensities as (I - pedestal) / monitor
    (as in mosaic.buildMosaic3())"""
    if CCDLabel in ("sCMOS", "sCMOS_fliplr", "sCMOS_4M", "sCMOS_9M"):
        dictMonitor = IOimage.read_header_scmos(imagepath)
        monitor = dictMonitor["mon"] - monitoroffset * dictMonitor["exposure"] / 1000.0
        return 1000.0, (monitor if monitor > 0 else 1.0)
    if CCDLabel in ("MARCCD165",):
        return 10.0, 1.0
    return 0.0, 1.0


def fit_parameters_dict(boxsize_X: int, boxsize_Y: int, framedim, fitparameters=None) -> Dict[str, Any]:
    """FittingParametersDict for readmccd.fitPeakMultiROIs() (as in mosaic.FitPeakOnMap())"""
    fitparameters = {**DEFAULT_FIT_PARAMETERS, **(fitparameters or {})}
    return {"boxsize": (2 * boxsize_X + 1, 2 * boxsize_Y + 1),
            "framedim": framedim,
            "saturation_value": 65000,
            "baseline": "auto",
            "startangles": 0,
            "position_start": fitparameters['positionStart'],
            "start_sigma1": fitparameters['peaksizeStart'],
            "start_sigma2": fitparameters['peaksizeStart'],
            "fitfunction": fitparameters['modelFunction'],
            "xtol": 0.0001,
            "offsetposition": 1}


def fit_rois(rois: List[np.ndarray], roicenters, framedim, fitparameters=None) -> np.ndarray:
    """fit 2D gaussian on ROIs pixel intensities arrays (ROIs are flipped and transposed as in
    mosaic.buildMosaic3() to get the same X, Y as mosaic.FitPeakOnMap())

    :return: array (nb rois, len(FIT_COLUMNS))
    """
    import LaueTools.readmccd as RMCCD
    boxsize_Y, boxsize_X = (np.array(rois[0].shape) - 1) // 2
    data = np.array([np.flipud(roi).T for roi in rois])
    params, _, infos, _, baseline = RMCCD.fitPeakMultiROIs(data, np.array(roicenters, dtype=float),
                                    fit_parameters_dict(boxsize_X, boxsize_Y, framedim, fitparameters),
                                    showfitresults=False)
    nfev = [info["nfev"] if info is not None else np.nan for info in infos]
    return np.column_stack((np.array(params, dtype=float), nfev, baseline))


def compute_roi_counters(rois: List[Optional[np.ndarray]], roicenters, counters, CCDLabel: str,
                                                        fitparameters=None) -> Dict[str, np.ndarray]:
    """compute counters of ROIs pixel intensities arrays of a single image

    :param rois: list of 2D arrays (None for ROI not read), see read_rois()
    :param roicenters: list of (X, Y) of ROIs centers
    :param counters: list of counters names (keys of ROI_COUNTERS)
    :return: dict counter: array of shape (nb rois, nb values) or (nb rois, ROI shape) for 'mosaic'
        (NaN for ROI not read)
    """
    valid = [k for k, roi in enumerate(rois) if roi is not None]
    if not valid:
        return {}
    shape = rois[valid[0]].shape
    boxsize_Y, boxsize_X = (shape[0] - 1) // 2, (shape[1] - 1) // 2
    res = {}
    for counter in counters:
        nbvalues = ROI_COUNTERS[counter]
        res[counter] = np.full((len(rois),) + (shape if nbvalues is None else (nbvalues,)), np.nan)
    jj, ii = np.indices(shape)  # line (Y) and column (X) indices
    for k in valid:
        roi = rois[k]
        # X, Y pixel position of roi[0, 0]
        x0, y0 = int(roicenters[k][0]) - boxsize_X, int(roicenters[k][1]) - boxsize_Y
        if 'mosaic' in res:
            res['mosaic'][k] = roi
        if 'mean' in res:
            res['mean'][k] = np.mean(roi)
        if 'max' in res:
            res['max'][k] = np.amax(roi)
        if 'ptp' in res:
            res['ptp'][k] = np.ptp(roi)
        if 'sum' in res:
            res['sum'][k] = np.sum(roi)
        if 'pixelval' in res:
            res['pixelval'][k] = roi[boxsize_Y, boxsize_X]
        if 'XYmax' in res:
            jmax, imax = np.unravel_index(np.argmax(roi), shape)
            res['XYmax'][k] = x0 + imax, y0 + jmax
        if 'XYcentroid' in res:
            weights = roi - np.amin(roi)
            total = np.sum(weights)
            if total > 0:
                res['XYcentroid'][k] = (x0 + np.sum(weights * ii) / total,
                                        y0 + np.sum(weights * jj) / total)
    if 'fit' in res:
        try:
            res['fit'][valid] = fit_rois([rois[k] for k in valid], [roicenters[k] for k in valid],
                                            DictLT.dict_CCD[CCDLabel][0], fitparameters)
        except Exception as exc:  # fit failure: NaN values
            print(f'fit failed: {exc}')
    return res


def process_single_image(imageindex: int, params: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
    """read ROIs in image of index imageindex and compute counters (see ROICountersWorkflow)

    :return: dict (see compute_roi_counters()) or None for missing or unreadable image
    """
    nbimages_recorded = params.get('nbimages_recorded')
    # image not yet recorded in stacked images files (full of zeros): not read
    if nbimages_recorded is not None and imageindex >= nbimages_recorded:
        return None
    imagepath, stackimageindex = image_filepath(params, imageindex)
    CCDLabel = params['CCDLabel']
    try:
        rois = read_rois(imagepath, params['roicenters'], params['boxsize_X'], params['boxsize_Y'],
                                                                    CCDLabel, stackimageindex)
        if params.get('NormalizeWithMonitor', False):
            pedestal, monitor = monitor_normalization(imagepath, CCDLabel,
                                                        params.get('monitoroffset', 0.))
            rois = [None if roi is None else (roi - pedestal) / monitor for roi in rois]
    except Exception as exc:  # missing or unreadable image (e.g. not completed scan, file being written)
        if params.get('verbose', 0):
            print(f'cannot read image index {imageindex} ({imagepath}): {exc}')
        return None
    return compute_roi_counters(rois, params['roicenters'], params['counters'], CCDLabel,
                                params.get('fitparameters')) or None


# parameters of workflow in each process of the pool (set once by _init_worker())
_WORKER_PARAMS = None


def _init_worker(params):
    global _WORKER_PARAMS
    _WORKER_PARAMS = params


def _process_image_in_worker(imageindex):
    return process_single_image(imageindex, _WORKER_PARAMS)


def available_cpus() -> int:
    """nb of cpus that can be used by this process (cpus allocated by SLURM on a cluster node)"""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not on linux
        return cpu_count()


def to_jsonable(obj):
    """convert numpy arrays and scalars (in dict, list, tuple) to json serializable objects"""
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, range)):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, bytes):
        return obj.decode(errors='replace')
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)  # e.g. dates of BLISS scan dict


class WorkflowCancelled(Exception):
    pass


def session_mp_context():
    """multiprocessing context to process images from a long running python session (GUI, notebook),
    possibly from a thread: 'fork' on linux, workers being copies of the session process with
    LaueTools already imported (fast start; 'forkserver' and 'spawn' would import the main module of
    the session, e.g. the whole GUI, in each worker). 'spawn' if fork is not available."""
    if sys.platform.startswith('linux'):
        return multiprocessing.get_context('fork')
    return multiprocessing.get_context('spawn')


class ROICountersWorkflow:
    """Compute counters (see ROI_COUNTERS) in ROIs centered on detector pixels over a set of
    images (e.g. 2D map), images being processed in parallel.

    params keys:
        folder, CCDLabel, filename_representative or (prefix, suffix), nbdigits: images files
            (see image_filepath())
        listindices: images indices (in map order, line by line for a 2D map)
        roicenters: list of (X, Y) pixel positions of ROIs centers (or roicenter for a single ROI)
        boxsize_X, boxsize_Y: ROI half sizes along X and Y (ROI shape is (2*boxsize_Y+1, 2*boxsize_X+1))
        counters: list of counters names (keys of ROI_COUNTERS)
    optional keys:
        mapdimensions: (nb images along fast axis, nb images along slow axis)
        nbframes_per_file, nbimages_recorded: for stacked images files
        NormalizeWithMonitor, monitoroffset: intensity normalization by monitor
        fitparameters: starting values of 2D peak fit for 'fit' counter (see DEFAULT_FIT_PARAMETERS)
        verbose: print reading errors
    """

    def __init__(self, params: Dict[str, Any]):
        params = dict(params)
        if params.get('roicenters') is None:
            params['roicenters'] = [params['roicenter']]
        params['roicenters'] = [[int(x), int(y)] for x, y in params['roicenters']]
        params['listindices'] = [int(index) for index in np.ravel(params['listindices'])]
        params['boxsize_X'], params['boxsize_Y'] = int(params['boxsize_X']), int(params['boxsize_Y'])
        unknown = set(params['counters']) - set(ROI_COUNTERS)
        if unknown:
            raise ValueError(f"unknown counters {unknown}. Possible counters: {list(ROI_COUNTERS)}")
        self.params = params
        self.results = None
        # set to True (e.g. from another thread) to stop run_roi_counters_workflow()
        self.cancel_requested = False

    def run_roi_counters_workflow(self, nb_cpus: Optional[int] = None, progressbar: bool = False,
                                    progress_callback=None, mp_context=None) -> Dict[str, Any]:
        """process all images in parallel with nb_cpus processes (None: all available cpus)

        :param progressbar: show tqdm progress bar (else print progress)
        :param progress_callback: function called with (nb processed images, nb images)
        :param mp_context: multiprocessing context of the pool of processes (default: multiprocessing
            default, see session_mp_context())
        :raise WorkflowCancelled: if self.cancel_requested is set to True during processing
        :return: results dict with keys 'imageindices', 'missing' (bool array, True for missing or
            unreadable image), 'params' and counters names: arrays of shape
            (nb images, nb rois, nb values) or (nb images, nb rois, ROI shape) for 'mosaic'
        """
        params = self.params
        listindices = params['listindices']
        nbimages, nbrois = len(listindices), len(params['roicenters'])
        roishape = (2 * params['boxsize_Y'] + 1, 2 * params['boxsize_X'] + 1)
        results = {'imageindices': np.array(listindices), 'missing': np.ones(nbimages, dtype=bool)}
        for counter in params['counters']:
            nbvalues = ROI_COUNTERS[counter]
            # float32 for ROIs arrays: mosaic may be large
            dtype = np.float32 if nbvalues is None else np.float64
            results[counter] = np.full((nbimages, nbrois) + (roishape if nbvalues is None else (nbvalues,)),
                                        np.nan, dtype=dtype)

        nb_cpus = max(1, min(nb_cpus or available_cpus(), available_cpus(), nbimages))
        print(f"Processing {nbimages} images, {nbrois} ROI(s), counters {params['counters']} "
                                                            f"with {nb_cpus} cpu(s)", flush=True)
        t0 = time.time()
        if nb_cpus == 1:
            iterresults = (process_single_image(index, params) for index in listindices)
            pool = None
        else:
            pool = (mp_context or multiprocessing).Pool(nb_cpus, initializer=_init_worker, initargs=(params,))
            chunksize = max(1, min(20, nbimages // (4 * nb_cpus)))
            iterresults = pool.imap(_process_image_in_worker, listindices, chunksize=chunksize)
        if progressbar:
            iterresults = tqdm(iterresults, total=nbimages, desc="ROI counters", unit="image")
        reportstep = max(1, nbimages // 100)
        try:
            for k, res in enumerate(iterresults):
                if self.cancel_requested:
                    raise WorkflowCancelled(f'workflow cancelled after {k} images')
                if res is not None:
                    results['missing'][k] = False
                    for counter, values in res.items():
                        results[counter][k] = values
                if (k + 1) % reportstep == 0 or k + 1 == nbimages:
                    if not progressbar:
                        print(f"processed {k + 1}/{nbimages} images", flush=True)
                    if progress_callback is not None:
                        progress_callback(k + 1, nbimages)
        finally:
            if pool is not None:
                pool.terminate()
        nbmissing = int(np.sum(results['missing']))
        print(f"ROI counters collected in {time.time() - t0:.1f} s"
                + (f" ({nbmissing} missing or unreadable images)" if nbmissing else ""), flush=True)
        results['params'] = params
        self.results = results
        return results

    def map2D(self, counter: str, roiindex: int = 0, column: Optional[int] = None) -> np.ndarray:
        """2D map (dimslow, dimfast) of counter values in ROI roiindex
        (column: index of value for counter with several values e.g. 0 for X of 'XYmax')"""
        values = self.results[counter][:, roiindex]
        if column is None and ROI_COUNTERS[counter] == 1:
            column = 0
        if column is not None:
            values = values[:, column]
        return counter_map2D(values, self.params['mapdimensions'])


def save_results(results: Dict[str, Any], path: str) -> str:
    """save results of ROICountersWorkflow in hdf5 file (params as json in attribute 'params')"""
    with h5py.File(path, 'w') as f:
        f.attrs['params'] = json.dumps(to_jsonable(results.get('params', {})))
        f.attrs['fit_columns'] = json.dumps(FIT_COLUMNS)
        for key, values in results.items():
            if key == 'params':
                continue
            compression = 'gzip' if key == 'mosaic' else None
            f.create_dataset(key, data=values, compression=compression)
    return path


def load_results(path: str) -> Dict[str, Any]:
    """load results of ROICountersWorkflow saved by save_results()"""
    with h5py.File(path, 'r') as f:
        results = {key: f[key][()] for key in f.keys()}
        results['params'] = json.loads(f.attrs['params'])
    return results


# ============================================================================
# USE CASE HANDLER
# ============================================================================

class UseCaseHandler:
    """Handles execution of use cases based on experiment parameters."""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.use_cases = {
            "mosaic_2d_map": self._run_mosaic_2d_map,
            "roi_counters_daxm": self._run_roi_counters_daxm,
            "roi_counters_ascan": self._run_roi_counters_ascan,
            "roi_counters": self._run_roi_counters,
        }
        #self.logger = LoggerManager.get_logger(__name__, log_file="workflow.log")

    def _run_mosaic(self, params: Dict[str, Any]) -> None:
        """Run mosaic workflow for a scans."""
        mosaic_workflow = MosaicWorkflow(params)
        results = mosaic_workflow.run_mosaic_workflow()
        #self.logger.info("Mosaic workflow completed.")
        return results

    def _run_mosaic_2d_map(self, params: Dict[str, Any]) -> None:
        """Run mosaic workflow for a 2D map scans."""
        mosaic_workflow = MosaicWorkflow(params)
        results = mosaic_workflow.run_mosaic_workflow()
        print('results',results)
        resdict = mosaic_workflow._arrange2D()
        #singleimage_mosaic = resdict.get('singleimage')
        #mosaic_shape = resdict.get('mosaic_shape')
        #self.logger.info("Mosaic workflow completed.")
        return resdict

    def _run_roi_counters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run ROI counters workflow (params: see ROICountersWorkflow, optional 'nb_cpus')"""
        return ROICountersWorkflow(params).run_roi_counters_workflow(params.get('nb_cpus'))

    def _run_roi_counters_daxm(self, params: Dict[str, Any]) -> None:
        """Run ROI counters workflow for DAXM scans."""
        roi_workflow = ROICountersWorkflow(params)
        results = roi_workflow.run_roi_counters_workflow()
        #self.logger.info("ROI counters workflow for DAXM completed.")
        return results

    def _run_roi_counters_ascan(self, params: Dict[str, Any]) -> None:
        """Run ROI counters workflow for ascan scans."""
        roi_workflow = ROICountersWorkflow(params)
        results = roi_workflow.run_roi_counters_workflow()
        #self.logger.info("ROI counters workflow for ascan completed.")
        return results

    def execute_use_case(self, use_case_name: str, params: Dict[str, Any]) -> Any:
        """
        Execute a use case by name.

        Args:
            use_case_name: Name of the use case to execute.
            params: Dictionary of parameters for the workflow.

        Returns:
            Results of the workflow execution.
        """
        if use_case_name not in self.use_cases:
            raise ValueError(f"Use case '{use_case_name}' not found.")
        return self.use_cases[use_case_name](params)



# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def plot_mosaic(mosaic_dict_results: np.ndarray, dict_scan_exp= None, vmin=0, vmax=2000) -> None:
    """Plot mosaic data with ROI center highlighted."""

    singleimage_mosaic = mosaic_dict_results.get('singleimage')
    mosaic_shape = mosaic_dict_results.get('mosaic_shape')
    def format_coord(x, y):
        col = int(x)
        row = int(y)
        if col >= 0 and col < singleimage_mosaic.shape[1] and row >= 0 and row < singleimage_mosaic.shape[0]:
            #cnt_idx = fig.gca()
            i, j = col//mosaic_shape.shape[3], row//mosaic_shape.shape[2]
            img_idx = 0+ mosaic_shape.shape[1]*j+i
            return "x=%1.4f, y=%1.4f, imageid=%d" % (x, y,img_idx)
        else:
            return "x=%1.4f, y=%1.4f" % (x, y)


    d = dict_scan_exp
    
    roicenter = d['roicenter']
    print(singleimage_mosaic.shape, )
    figmosaic, axmosaic = plt.subplots(figsize=(8,8))
    axmosaic.format_coord = format_coord
    axmosaic.imshow(singleimage_mosaic, origin='lower', vmax=vmax)
    axmosaic.set_xlabel(f"fastaxis {d['fastaxis']} // pixelX")  # ok for fdscan2d
    axmosaic.set_ylabel(f"slowaxis {d['slowaxis']} // pixelY")
    title = ''
    title += '%s\n'%d['imagefolder']
    title += 'roicenter Det. pixel X,Y = (%d,%d)'%(roicenter[0],roicenter[1])
    axmosaic.set_title(title)
    axmosaic.format_coord = format_coord
    plt.show()
