Welcome to LaueTools's DOCUMENTATION!
=====================================

Last revision (March 2020)

LaueTools code and documentation at:

https://lauetools.readthedocs.io/en/latest/index.html

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

Normally, in a command window (if environment variables are well set) 3 main GUIs can be launched:

	- lauetools   for the main GUI

	- peaksearch  for batch Laue pattern peak search processing
	
	- indexrefine   for batch Laue pattern indexing and unit cell (strain) refinement

To deal with relative import, from the parent folder of LaueTools, the package name ‘LaueTools’ must be specified to the python interpreter as following

	Examples:

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

	.. image:: Images/notebook0.jpg

