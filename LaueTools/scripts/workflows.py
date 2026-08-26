import numpy as np
import pandas as pd
import h5py
import fabio
import logging
import itertools
import os

import matplotlib.pyplot as plt
from pathlib import Path
from LaueTools import dict_LaueTools as DictLT



from typing import Dict, Any, List, Optional, Tuple, Union

from tqdm.notebook import tqdm
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
        max_dimension = 2015  # Adjust based on detector specs
        if (
            x_roi - self.boxsize_X < 0
            or x_roi + self.boxsize_X > max_dimension
            or y_roi - self.boxsize_Y < 0
            or y_roi + self.boxsize_Y > max_dimension
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
        scantype = self.params.get("scantype")
        mapdimensions = self.params.get("mapdimensions")
        listindices = self.params.get("listindices")
        all_results= self.all_results

        boxsize_X = self.boxsize_X
        boxsize_Y = self.boxsize_Y
        
        if scantype=='map':
            # processing and rearranging collected ROIs imagelets
            dimfast, dimslow = mapdimensions
            #print('axis dimensions: dimslow, dimfast',dimslow, dimfast)
            mosaic = np.zeros((dimslow, dimfast, 2*boxsize_Y+1, 2*boxsize_X+1))
            #print('mosaic.shape', mosaic.shape)
            dict_map_imageindex ={}
        
            #print(d['listindices'])
            
            sm = mosaic.shape
            bigimage = np.zeros((sm[0]*sm[2],sm[1]*sm[3]))
            
            
            if dimfast > 0:
                for map_imageindex, absolute_imageindex in enumerate(listindices):
                    
                    imap, jmap = map_imageindex // dimfast, map_imageindex % dimfast
            
                    dict_map_imageindex[map_imageindex] = [absolute_imageindex,
                                                            map_imageindex,
                                                            imap, jmap]
                    
                    raw = all_results[map_imageindex,:,:]
                    
                    #datcrop = np.flipud(raw).T
                    datcrop = raw
                    
                    mosaic[imap,jmap] = datcrop #datcrop.T #np.flipud(datcrop).T
                    # for 2D map
                    bigimage[imap*sm[2]:(imap+1)*sm[2], jmap*sm[3]:(jmap+1)*sm[3]] = np.flipud(datcrop)
            
            if 0:
                mosaictranspose = mosaic.transpose((0, 3, 1, 2))
                mosaicflat = mosaictranspose.reshape((dimfast * (2 * boxsize_X + 1), dimslow * (2 * boxsize_Y + 1)))

            resdict = {}
            resdict['mosaicdata'] = mosaic
            resdict['singleimage'] = bigimage
            resdict['mosaic_shape'] = mosaic.shape
            return resdict

    @staticmethod
    def _collect_roi_array_single_file(
        index: int,
        roicenter: Tuple[int, int],
        prefix: str,
        folder: str,
        boxsize_X: int,
        boxsize_Y: int,
        ccd_label: str,
    ) -> np.ndarray:
        """Helper function to collect ROI array for a single file
        listindices,
                   itertools.repeat(roicenter),
                   itertools.repeat(prefix),
                   listfolders,
                  itertools.repeat(boxsize_X),
                  itertools.repeat(boxsize_Y),
                   itertools.repeat(CCDLabel)"""
        # Placeholder for actual implementation
        
        return collectroiarray_singlefile(index,roicenter,prefix, folder, boxsize_X, boxsize_Y, ccd_label)




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
