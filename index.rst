.. include::LaueTools/Documentation/source/index.rst

PPPPPPPPPPPPPPPP

Welcome to LaueTools's documentation!
=====================================

Synopsis
*****************

LaueTools is a package of python modules (with scripts and wxpython-based Graphical interface) for Laue pattern analysis, and particularly dedicated to synchrotron Laue microdiffraction data analysis from CRG-IF BM32 beamline @ ESRF, Grenoble, France

1- Download LaueTools code
***************************

- the very last version of the code at gitlab.esrf.fr (you are also welcome to fork this project):

	https://gitlab.esrf.fr/micha/lauetools

- last (or older releases) with repository on pypi

	https://pypi.org/project/LaueTools/

	if pip is installed:

	.. code-block:: python

	   pip install lauetools 



2a- Launch Graphical User Interfaces of LaueTools
*************************************************
- start Lauetools GUIs from command line :

To deal with relative import, the package name ‘LaueTools’ must be specified to the python interpreter as following

	Examples:

	- python -m LaueTools.LaueToolsGUI

	- python -m LaueTools.LaueSimulatorGUI

	- python -m LaueTools.PeaksearchGUI

The two last GUIs (LaueSimulatorGUI, PeaksearchGUI) can be accessed by the first main one, LaueToolsGUI

There are additional basic GUIs for batch processing located in FileSeries folder:

	- python -m LaueTools.FileSeries.Peak_Search
	- python -m LaueTools.FileSeries.Index_Refine
	- python -m LaueTools.FileSeries.Build_summary
	- python -m LaueTools.FileSeries.Plot_Maps2

- within interactive python (say, ipython -i), GUI can be started thanks to a start() function:

	- In [1] : import LaueTools.LaueToolsGUI as LTGUI

	- In [2] : LTGUI.start()

.. note::
	in the LaueTools folder :

	- neither > python LaueToolsGUI

	- nor in >ipython -i :  > run LaueToolsGUI  will work…


2b- Use LaueTools module as a library
**************************************

With pip installation, LaueTools package will be included to python packages. Therefore any module will be callable as the following:
 
	-In [1] : import LaueTools.readmccd as rmccd

	-In [2] : rmccd.readCCDimage(‘myimage.tif’)

In jupyter-notebook, it is also simple:

	.. image:: notebook0.jpg

3- Mathematics and Conventions
**************************************


4- Graphical User Interfaces
**************************************

The main steps of analysis are Laue peaks search, Laue Pattern indexation and unit Cell Refinement. Detector geometry calibration (DetectorCalibrationBoard) and Laue Pattern of Polycrystals (LaueSimulatorGUI) are also available.

4a- Peak Search (PeaksearchGUI)
----------------------------------

4b- Indexation (LaueToolsGUI)
---------------------------------------

4c- Crystal unit cell refinement (LaueToolsGUI)
---------------------------------------------------

4d- Detector Geometry Calibration (DetectorCalibrationBoard)
-----------------------------------------------------------------

4e- Laue pattern simulation of assembly of crystals (LaueSimulatorGUI)
---------------------------------------------------------------------------



.. toctree::
   :maxdepth: 3
   :caption: Contents:

   Simulation_Module.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
