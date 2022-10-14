#########################
LaueTools Modules
#########################

Browse Modules and Functions
==============================

* :ref:`genindex`
* :ref:`modindex`

.. * :ref:`search`


Modules for Laue Pattern Simulation
======================================

The following modules are used to compute Laue pattern from grain (or crystal) structural parameters and 2D plane detector geometry:

   - `CrystalParameters.py` defines structural parameters describing the crystal. It includes orientation matrix and strain operators.

   - `lauecore.py` contains the core procedures to compute all Laue spots properties.

   - `LaueGeometry.py` handles the 2D plane geometry set by the detector position and orientation with respect to sample and incoming direction.

   - `multigrainsSimulator.py` allows to simulate an assembly of grains, some of them according to a distribution of grains. This module is called by the graphical user interface (LaueSimulatorGUI) which provides all arguments in an intuitive way.    

.. toctree::
    Simulation_Module.rst


Modules for Digital Image processing, Peak Search & Fitting
=============================================================

.. toctree::
    PeakSearchGUI.rst
    PeakSearch.rst
    
Modules for Laue Pattern Indexation
=====================================================


Modules for Crystal unit cell refinement
=====================================================


Modules for batch processing
=====================================================
