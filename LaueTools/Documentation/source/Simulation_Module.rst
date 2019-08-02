.. _Simulation:


CrystalParameters
=======================

.. automodule:: LaueTools.CrystalParameters
    :members: GrainParameter_from_Material, Prepare_Grain, AngleBetweenNormals, FilterHarmonics_2,
        calc_B_RR, VolumeCell, VolumeCell, DeviatoricStrain_LatticeParams, matrix_to_rlat

Laue Pattern Simulation
============================

.. automodule:: LaueTools.lauecore
    :members: SimulateLaue, SimulateLaue_full_np, SimulateResult, Quicklist, genHKL_np,
	ApplyExtinctionrules, getLaueSpots, create_spot,
	create_spot_np, filterLaueSpots, get2ThetaChi_geometry, calcSpots_fromHKLlist

2D Detection Geometry
===============================

.. automodule:: LaueTools.LaueGeometry
   :members: calc_uflab, calc_uflab_trans, calc_xycam, calc_xycam_transmission,
       uflab_from2thetachi, from_twchi_to_qunit,from_twchi_to_q,
       from_qunit_to_twchi, qvector_from_xy_E, unit_q, Compute_data2thetachi,
       convert2corfile, convert2corfile_fileseries, convert2corfile_multiprocessing,
       vec_normalTosurface, vec_onsurface_alongys, convert_xycam_from_sourceshift,
       lengthInSample

Multiple Grains and Strain/orientation Distribution 
=========================================================
.. automodule:: LaueTools.multigrainsSimulator
    :members: Read_GrainListparameter, Construct_GrainsParameters_parametric, dosimulation_parametric


