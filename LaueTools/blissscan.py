from pandas import Timestamp
import numpy as np


class BlissScan:
    """
    A class to represent and launch workflows based on experimental parameters.
    Attributes are dynamically created from the input dictionary.
    """

    def __init__(self, params: dict):
        """
        Initialize the BlissScan with a dictionary of parameters.
        Dynamically sets attributes based on the dictionary keys.
        """
        for key, value in params.items():
            setattr(self, key, value)

        # Validate required attributes based on scan type
        self._validate_parameters()

    def _validate_parameters(self):
        """
        Validate that required parameters are present for the scan type.
        Raises an AttributeError if required parameters are missing.
        """
        if not hasattr(self, 'scantype'):
            raise AttributeError("Missing required parameter: 'scantype'")

        if self.scantype == 'map':
            required_params = [
                'motors', 'mapdimensions', 'fastaxis', 'slowaxis',
                'localhdf5file', 'imagefolder', 'scanindex', 'prefix', 'suffix',
                'CCDLabel', 'fullcommand'
            ]
            for param in required_params:
                if not hasattr(self, param):
                    raise AttributeError(f"Missing required parameter for 'map' scan: '{param}'")

        elif self.scantype == 'daxm':
            required_params = [
                'localhdf5file', 'imagefolder', 'scanindex', 'prefix', 'suffix',
                'CCDLabel', 'fullcommand'
            ]
            for param in required_params:
                if not hasattr(self, param):
                    raise AttributeError(f"Missing required parameter for 'daxm' scan: '{param}'")

        elif self.scantype in ['ascan', 'a2scan']:
            required_params = [
                'motors', 'localhdf5file', 'imagefolder', 'scanindex',
                'prefix', 'suffix', 'CCDLabel', 'fullcommand'
            ]
            for param in required_params:
                if not hasattr(self, param):
                    raise AttributeError(f"Missing required parameter for '{self.scantype}' scan: '{param}'")

    def possible_workflows(self):
        """
        Return a list of possible workflows based on the scan type.
        """
        workflows = {
            'map': ['mosaic', 'grainimaging'],
            'daxm': ['quickdaxm', 'calibratewire', 'reconstruct_one_spot', 'reconstruct_several_spots'],
            'ascan': ['mosaic', 'grainprofile'],
            'a2scan': ['mosaic', 'grainprofile']
        }
        return workflows.get(self.scantype, [])

    def launch_workflow(self, workflow_name: str):
        """
        Launch a workflow based on the scan type and workflow name.
        """
        possible_workflows = self.possible_workflows()
        if workflow_name not in possible_workflows:
            raise ValueError(
                f"Workflow '{workflow_name}' is not available for scan type '{self.scantype}'. "
                f"Available workflows: {possible_workflows}"
            )

        print(f"Launching {workflow_name} workflow for {self.scantype} scan: {self.scanindex}")
        # Placeholder for actual workflow logic
        if self.scantype == 'map':
            if workflow_name == 'mosaic':
                self._launch_mosaic_workflow()
            elif workflow_name == 'grainimaging':
                self._launch_grainimaging_workflow()

        elif self.scantype == 'daxm':
            if workflow_name == 'quickdaxm':
                self._launch_quickdaxm_workflow()
            elif workflow_name == 'calibratewire':
                self._launch_calibratewire_workflow()
            elif workflow_name == 'reconstruct_one_spot':
                self._launch_reconstruct_one_spot_workflow()
            elif workflow_name == 'reconstruct_several_spots':
                self._launch_reconstruct_several_spots_workflow()

        elif self.scantype in ['ascan', 'a2scan']:
            if workflow_name == 'mosaic':
                self._launch_mosaic_workflow()
            elif workflow_name == 'grainprofile':
                self._launch_grainprofile_workflow()

    def _launch_mosaic_workflow(self):
        """Placeholder for mosaic workflow logic."""
        print(f"Running MOSAIC workflow for scan: {self.scanindex}")
        print(f"Motors: {self.motors}")
        print(f"HDF5 file: {self.localhdf5file}")

    def _launch_grainimaging_workflow(self):
        """Placeholder for grain imaging workflow logic."""
        print(f"Running GRAIN IMAGING workflow for scan: {self.scanindex}")
        print(f"Dimensions: {self.mapdimensions}")

    def _launch_quickdaxm_workflow(self):
        """Placeholder for quick daxm workflow logic."""
        print(f"Running QUICK DAXM workflow for scan: {self.scanindex}")

    def _launch_calibratewire_workflow(self):
        """Placeholder for calibrate wire workflow logic."""
        print(f"Running CALIBRATE WIRE workflow for scan: {self.scanindex}")

    def _launch_reconstruct_one_spot_workflow(self):
        """Placeholder for reconstruct one spot workflow logic."""
        print(f"Running RECONSTRUCT ONE SPOT workflow for scan: {self.scanindex}")

    def _launch_reconstruct_several_spots_workflow(self):
        """Placeholder for reconstruct several spots workflow logic."""
        print(f"Running RECONSTRUCT SEVERAL SPOTS workflow for scan: {self.scanindex}")

    def _launch_grainprofile_workflow(self):
        """Placeholder for grain profile workflow logic."""
        print(f"Running GRAIN PROFILE workflow for scan: {self.scanindex}")

    def __str__(self):
        """String representation of the BlissScan object."""
        attrs = vars(self)
        return "\n".join(f"{key}: {value}" for key, value in attrs.items())


# Example usage
if __name__ == "__main__":
    # Example for 'map' scan
    params_map = {
        'scantype': 'map',
        'start_time': Timestamp('2026-07-08 17:07:22'),
        'end_time': Timestamp('2026-07-08 20:05:31'),
        'sample_dataset_scanindex': 'CrZr_map2D_3',
        'fullcommand': 'amesh yech -3.65030 -3.53030 120 xech -2.51804 -2.39804 120 0.3',
        'scanindex': '3',
        'motors': 'xech yech',
        'localhdf5file': '/data/visitor/a321217/bm32/20260707/RAW_DATA/CrZr/CrZr_map2D/CrZr_map2D.h5',
        'imagefolder': '/data/visitor/a321217/bm32/20260707/RAW_DATA/CrZr/CrZr_map2D/scan0003',
        'endreason': 'SUCCESS',
        'samplename': b'CrZr',
        'folder': '/data/visitor/a321217/bm32/20260707/RAW_DATA/CrZr/CrZr_map2D/scan0003',
        'nodeinhdf5file': 'CrZr/CrZr_map2D/3.1',
        'prefix': 'eiger4m_',
        'suffix': 'h5',
        'listindices': np.array([0, 1, 2, 14638, 14639, 14640]),
        'nbimagesperline': 121,
        'mapdimensions': (121, 121),
        'peaklistfile': None,
        'fastaxis': 'yech',
        'slowaxis': 'xech',
        'collector': 'pixelval',
        'CCDLabel': 'EIGER_4MCdTe'
    }

    scan_map = BlissScan(params_map)
    print(f"Possible workflows for 'map' scan: {scan_map.possible_workflows()}")
    scan_map.launch_workflow('mosaic')

    # Example for 'daxm' scan
    params_daxm = {
        'scantype': 'daxm',
        'scanindex': '4',
        'localhdf5file': '/data/visitor/a321217/bm32/20260707/RAW_DATA/CrZr/CrZr_daxm/CrZr_daxm.h5',
        'imagefolder': '/data/visitor/a321217/bm32/20260707/RAW_DATA/CrZr/CrZr_daxm/scan0004',
        'prefix': 'eiger4m_',
        'suffix': 'h5',
        'CCDLabel': 'EIGER_4MCdTe',
        'fullcommand': 'daxm yech -3.65030 -3.53030 120'
    }

    scan_daxm = BlissScan(params_daxm)
    print(f"\nPossible workflows for 'daxm' scan: {scan_daxm.possible_workflows()}")
    scan_daxm.launch_workflow('quickdaxm')