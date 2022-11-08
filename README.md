
![til](https://github.com/BM32ESRF/lauetools/blob/master/animation_Si.gif)

[![Conda](https://img.shields.io/conda/pn/bm32esrf/lauetools?color=green&label=supported%20platform)](https://anaconda.org/bm32esrf/lauetools)

[![Lint, test, build, and publish (PYPI, CONDA)](https://github.com/BM32ESRF/lauetools/actions/workflows/complete_workflow.yml/badge.svg)](https://github.com/BM32ESRF/lauetools/actions/workflows/complete_workflow.yml)
[![PyPI](https://img.shields.io/pypi/v/LaueTools)](https://pypi.python.org/pypi/LaueTools/)
[![Conda](https://img.shields.io/conda/v/bm32esrf/lauetools?style=flat-square)](https://anaconda.org/bm32esrf/lauetools)


[![PyPI pyversions](https://img.shields.io/pypi/pyversions/LaueTools.svg)](https://pypi.python.org/pypi/LaueTools/)
[![Anaconda-Server Badge](https://anaconda.org/bm32esrf/lauetools/badges/license.svg)](https://anaconda.org/bm32esrf/lauetools)


Welcome to LaueTools's DOCUMENTATION!
=====================================

Last revision (Sept 2022)

LaueTools information on BM32 beamline website:

https://www.esrf.fr/UsersAndScience/Experiments/CRG/BM32/Microdiffraction


1- Download LaueTools code
***************************

- the very last version of the code running with python3 is now on github (you are also welcome to fork this project):

	https://github.com/BM32ESRF/lauetools

- last (or older releases) with repository on pypi

	https://pypi.org/project/LaueTools/

	if pip is installed:

	.. code-block:: python

	   pip install lauetools

- Former LaueTools package written for python 2.7 only (up to June 2019) is no longer maintained and can be found on sourceforge:
	
	https://sourceforge.net/projects/lauetools/version


But it is highly recommended to use python 3 to take benefit from all capabilities

2a- Launch Graphical User Interfaces of LaueTools
*************************************************
- start Lauetools GUIs from command line :

Normally, in a command window (if environment variables are well set) 3 main GUIs can be launched:

	> lauetools   for the main GUI
	> peaksearch  for batch Laue pattern peak search processing
	> indexrefine   for batch Laue pattern indexing and unit cell (strain) refinement
	> buildsummary    to compile all results from indexrefine analysis
	> plotmap    to plot 2D map of structural quantities from file built previously
	> plotmeshgui   to plot 2D map from counters values in (spec) logfile

To deal with relative import, the package name ‘LaueTools’ must be specified to the python interpreter with -m option as following

	if LaueTools is a current subfolder:

	- python -m LaueTools.LaueToolsGUI

	- python -m LaueTools.GUI.LaueSimulatorGUI

	- python -m LaueTools.GUI.PeaksearchGUI

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

In jupyter-notebook, it is also simple in the same manner:

	.. image:: /images/notebook0.jpg




