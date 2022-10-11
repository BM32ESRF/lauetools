
Indexation of spots data set.
=============================

**Using Class spotsSet and play with spots considered for indexation and
refinement**

This Notebook is a part of Tutorials on LaueTools Suite. Author:J.-S. Micha Date: July 2019
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %matplotlib inline
    
    import matplotlib
    import numpy as np
    import matplotlib.pyplot as plt
    import time,copy,os
    
    # third party LaueTools import
    import LaueTools.readmccd as RMCCD
    import LaueTools.LaueGeometry as F2TC
    import LaueTools.indexingSpotsSet as ISS
    import LaueTools.IOLaueTools as RWASCII


.. parsed-literal::

    /home/micha/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters


.. parsed-literal::

    module Image / PIL is not installed
    LaueToolsProjectFolder /home/micha/LaueToolsPy3/LaueTools
    Cython compiled module 'gaussian2D' for fast computation is not installed!
    module Image / PIL is not installed
    Entering CrystalParameters ******---***************************
    
    
    Cython compiled module for fast computation of Laue spots is not installed!
    Cython compiled 'angulardist' module for fast computation of angular distance is not installed!
    Using default module
    Cython compiled module for fast computation of angular distance is not installed!
    module Image / PIL is not installed


refinement from guessed solutions with two materials (see script
IndexingTwinsSeries)

Let's take a simple example of a single Laue Pattern. From the peak search we get 83 spots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    folder= '../Examples/Ge/'
    datfilename='Ge0001.dat'

.. code:: ipython3

    key_material='Ge'
    emin, emax= 5,23

.. code:: ipython3

    # detector geometry and parameters as read from Ge0001.det
    calibration_parameters = [69.179, 1050.81, 1115.59, 0.104, -0.273]
    CCDCalibdict = {}
    CCDCalibdict['CCDCalibParameters'] = calibration_parameters
    CCDCalibdict['framedim'] = (2048, 2048)
    CCDCalibdict['detectordiameter'] = 165.
    CCDCalibdict['kf_direction'] = 'Z>0'
    CCDCalibdict['xpixelsize'] = 0.08057
    # CCDCalibdict can also be simply build by reading the proper .det file
    print("reading geometry calibration file")
    CCDCalibdict=RWASCII.readCalib_det_file(os.path.join(folder,'Ge0001.det'))
    CCDCalibdict['kf_direction'] = 'Z>0'


.. parsed-literal::

    reading geometry calibration file
    calib =  [ 6.91790e+01  1.05081e+03  1.11559e+03  1.04000e-01 -2.73000e-01
      8.05700e-02  2.04800e+03  2.04800e+03]
    matrix =  [-0.211596  0.092178 -0.973001 -0.77574   0.589743  0.224568  0.594521
      0.802313 -0.053281]


Compute scattering angles from spots pixel positions and detector geometry. Write a .cor file from .dat including these new infos
                                                                                                                                 

.. code:: ipython3

    F2TC.convert2corfile(datfilename,
                             calibration_parameters,
                             dirname_in=folder,
                            dirname_out=folder,
                            CCDCalibdict=CCDCalibdict)
    corfilename = datfilename.split('.')[0] + '.cor'
    fullpathcorfile = os.path.join(folder,corfilename)


.. parsed-literal::

    nb of spots and columns in .dat file (83, 3)
    file :../Examples/Ge/Ge0001.dat
    containing 83 peaks
    (2theta chi X Y I) written in ../Examples/Ge/Ge0001.cor


Create an instance of the class spotset. Initialize spots properties to data contained in .cor file
                                                                                                   

.. code:: ipython3

    DataSet = ISS.spotsset()
    
    DataSet.importdatafromfile(fullpathcorfile)


.. parsed-literal::

    CCDcalib in readfile_cor {'dd': 69.179, 'xcen': 1050.81, 'ycen': 1115.59, 'xbet': 0.104, 'xgam': -0.273, 'xpixelsize': 0.08057, 'ypixelsize': 0.08057, 'CCDLabel': 'sCMOS', 'framedim': [2048.0, 2048.0], 'detectordiameter': 165.00736, 'kf_direction': 'Z>0', 'pixelsize': 0.08057}
    CCD Detector parameters read from .cor file
    CCDcalibdict {'dd': 69.179, 'xcen': 1050.81, 'ycen': 1115.59, 'xbet': 0.104, 'xgam': -0.273, 'xpixelsize': 0.08057, 'ypixelsize': 0.08057, 'CCDLabel': 'sCMOS', 'framedim': [2048.0, 2048.0], 'detectordiameter': 165.00736, 'kf_direction': 'Z>0', 'pixelsize': 0.08057}




.. parsed-literal::

    True



Class methods and attributes rely on a dictionnary of spots properties. key = exprimental spot index, val = spots properties
                                                                                                                            

.. code:: ipython3

    [DataSet.indexed_spots_dict[k] for k in range(10)]




.. parsed-literal::

    [[0, 78.215821, 1.638153, 1027.11, 1293.28, 70931.27, 0],
     [1, 64.329767, -20.824155, 1379.17, 1553.58, 51933.84, 0],
     [2, 68.680451, -15.358122, 1288.11, 1460.16, 22795.07, 0],
     [3, 105.61498, 8.176187, 926.22, 872.06, 19489.69, 0],
     [4, 103.859791, 27.866566, 595.46, 876.44, 19058.79, 0],
     [5, 120.59561, -8.92066, 1183.27, 598.92, 17182.88, 0],
     [6, 60.359458, 26.483191, 626.12, 1661.28, 15825.39, 0],
     [7, 56.269853, 12.967153, 856.14, 1702.52, 15486.2, 0],
     [8, 82.072076, -35.89243, 1672.67, 1258.62, 13318.81, 0],
     [9, 83.349535, -27.458061, 1497.4, 1224.7, 13145.99, 0]]



.. code:: ipython3

    DataSet.getUnIndexedSpotsallData()[:3]




.. parsed-literal::

    array([[ 0.0000000e+00,  7.8215821e+01,  1.6381530e+00,  1.0271100e+03,
             1.2932800e+03,  7.0931270e+04],
           [ 1.0000000e+00,  6.4329767e+01, -2.0824155e+01,  1.3791700e+03,
             1.5535800e+03,  5.1933840e+04],
           [ 2.0000000e+00,  6.8680451e+01, -1.5358122e+01,  1.2881100e+03,
             1.4601600e+03,  2.2795070e+04]])



.. code:: ipython3

    dict_loop = {'MATCHINGRATE_THRESHOLD_IAL': 100,
                       'MATCHINGRATE_ANGLE_TOL': 0.2,
                       'NBMAXPROBED': 6,
                       'central spots indices': [0,],
                       'AngleTolLUT': 0.5,
                       'UseIntensityWeights': False,
                       'nbSpotsToIndex':10000,
                       'list matching tol angles':[0.5,0.5,0.2,0.2],
                       'nlutmax':3,
                       'MinimumNumberMatches': 3,
                       'MinimumMatchingRate':3
                       }
    grainindex=0
    DataSet = ISS.spotsset()
        
    DataSet.pixelsize = CCDCalibdict['xpixelsize']
    DataSet.dim = CCDCalibdict['framedim']
    DataSet.detectordiameter = CCDCalibdict['detectordiameter']
    DataSet.kf_direction = CCDCalibdict['kf_direction']
    DataSet.key_material = key_material
    DataSet.emin = emin
    DataSet.emax = emax


Normally we read all spots data from a .cor file
''''''''''''''''''''''''''''''''''''''''''''''''

.. code:: ipython3

    DataSet.importdatafromfile(fullpathcorfile)
    DataSet.emin


.. parsed-literal::

    CCDcalib in readfile_cor {'dd': 69.179, 'xcen': 1050.81, 'ycen': 1115.59, 'xbet': 0.104, 'xgam': -0.273, 'xpixelsize': 0.08057, 'ypixelsize': 0.08057, 'CCDLabel': 'sCMOS', 'framedim': [2048.0, 2048.0], 'detectordiameter': 165.00736, 'kf_direction': 'Z>0', 'pixelsize': 0.08057}
    CCD Detector parameters read from .cor file
    CCDcalibdict {'dd': 69.179, 'xcen': 1050.81, 'ycen': 1115.59, 'xbet': 0.104, 'xgam': -0.273, 'xpixelsize': 0.08057, 'ypixelsize': 0.08057, 'CCDLabel': 'sCMOS', 'framedim': [2048.0, 2048.0], 'detectordiameter': 165.00736, 'kf_direction': 'Z>0', 'pixelsize': 0.08057}




.. parsed-literal::

    5



but we can import a custom list of spots. For example, starting from spots a the previous .cor file
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

.. code:: ipython3

    Gespots = RWASCII.readfile_cor(fullpathcorfile)[0]


.. parsed-literal::

    CCDcalib in readfile_cor {'dd': 69.179, 'xcen': 1050.81, 'ycen': 1115.59, 'xbet': 0.104, 'xgam': -0.273, 'xpixelsize': 0.08057, 'ypixelsize': 0.08057, 'CCDLabel': 'sCMOS', 'framedim': [2048.0, 2048.0], 'detectordiameter': 165.00736, 'kf_direction': 'Z>0', 'pixelsize': 0.08057}
    CCD Detector parameters read from .cor file


.. code:: ipython3

    # 2theta chi X, Y Intensity of the first 7 spots
    Gespots[:7,:5]




.. parsed-literal::

    array([[ 7.82158210e+01,  1.63815300e+00,  1.02711000e+03,
             1.29328000e+03,  7.09312700e+04],
           [ 6.43297670e+01, -2.08241550e+01,  1.37917000e+03,
             1.55358000e+03,  5.19338400e+04],
           [ 6.86804510e+01, -1.53581220e+01,  1.28811000e+03,
             1.46016000e+03,  2.27950700e+04],
           [ 1.05614980e+02,  8.17618700e+00,  9.26220000e+02,
             8.72060000e+02,  1.94896900e+04],
           [ 1.03859791e+02,  2.78665660e+01,  5.95460000e+02,
             8.76440000e+02,  1.90587900e+04],
           [ 1.20595610e+02, -8.92066000e+00,  1.18327000e+03,
             5.98920000e+02,  1.71828800e+04],
           [ 6.03594580e+01,  2.64831910e+01,  6.26120000e+02,
             1.66128000e+03,  1.58253900e+04]])



.. code:: ipython3

    tth,chi,X,Y,I=Gespots[:,:5].T
    exp_data_all=np.array([tth,chi,I,X,Y])
    exp_data_all.shape




.. parsed-literal::

    (5, 83)



.. code:: ipython3

    #select some exp spots from absolute index   (6,0,2,30,9,8,20,10,5,1,7,14)
    tth_e,chi_e,X_e,Y_e,I_e = (np.take(Gespots[:,:5],(6,0,2,30,9,8,20,10,5,1,7,14),axis=0)).T
    exp_data=np.array([tth_e,chi_e,I_e,X_e,Y_e])

spots data must be imported as an array of 5 elements: 2theta, chi, Intensity, pixelX, pixelY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    DataSet.importdata(exp_data)
    DataSet.detectorparameters = calibration_parameters
    DataSet.nbspots = len(exp_data[0])
    DataSet.filename = 'short_'+corfilename
    #DataSet.setSelectedExpSpotsData(0)
    DataSet.getSelectedExpSpotsData(0)




.. parsed-literal::

    (array([[ 6.03594580e+01,  7.82158210e+01,  6.86804510e+01,
              1.08452917e+02,  8.33495350e+01,  8.20720760e+01,
              8.17712570e+01,  9.17982210e+01,  1.20595610e+02,
              6.43297670e+01,  5.62698530e+01,  1.14942090e+02],
            [ 2.64831910e+01,  1.63815300e+00, -1.53581220e+01,
              3.77494610e+01, -2.74580610e+01, -3.58924300e+01,
              3.03824700e+01, -8.30994100e+00, -8.92066000e+00,
             -2.08241550e+01,  1.29671530e+01,  1.15295700e+01],
            [ 1.58253900e+04,  7.09312700e+04,  2.27950700e+04,
              4.40061000e+03,  1.31459900e+04,  1.33188100e+04,
              6.13787000e+03,  1.17999300e+04,  1.71828800e+04,
              5.19338400e+04,  1.54862000e+04,  1.00105200e+04]]),
     array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]))



core function to index a set of spots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

by defaut DataSet.getUnIndexedSpotsallData() is called

if use\_file = 0, then current non indexed exp. spots will be considered
for indexation

if use\_file = 1, reimport data from file and reset also spots
properties dictionary (i.e. with status unindexed)

.. code:: ipython3

    DataSet.IndexSpotsSet(fullpathcorfile, key_material, emin, emax, dict_loop, None,
                             use_file=0, # if 1, reimport data from file and reset also spots properties dictionary
                             IMM=False,LUT=None,n_LUT=dict_loop['nlutmax'],angletol_list=dict_loop['list matching tol angles'],
                            nbGrainstoFind=1,
                          starting_grainindex=0,
                          MatchingRate_List=[1, 1, 1,1,1,1,1,1],
                            verbose=0, previousResults=None,
                            corfilename=corfilename)


.. parsed-literal::

    self.pixelsize in IndexSpotsSet 0.08057
    ResolutionAngstromLUT in IndexSpotsSet False
    
     Remaining nb of spots to index for grain #0 : 12
    
    
     ******
    start to index grain #0 of Material: Ge 
    
    ******
    
    providing new set of matrices Using Angles LUT template matching
    nbspots 12
    NBMAXPROBED 6
    nbspots 12
    set_central_spots_hkl None
    Computing LUT from material data
    Compute LUT for indexing Ge spots in LauePattern 
    Build angles LUT with latticeparameters
    [ 5.657499999999999  5.657499999999999  5.657499999999999
     90.                90.                90.               ]
    and n=3
    MaxRadiusHKL False
    cubicSymmetry True
    Central set of exp. spotDistances from spot_index_central_list probed
    self.absolute_index [ 0  1  2  3  4  5  6  7  8  9 10 11]
    spot_index_central_list [0]
    [0]
    LUT is not None when entering getOrientMatrices()
    set_central_spots_hkl None
    set_central_spots_hkl_list [None]
    cubicSymmetry True
    LUT_tol_angle 0.5
    *---****------------------------------------------------*
    Calculating all possible matrices from exp spot #0 and the 5 other(s)
    hkl in getOrientMatrices None <class 'NoneType'>
    using LUTcubic
    LUTcubic is None for k_centspot_index 0 in getOrientMatrices()
    hkl1 in matrices_from_onespot_hkl() [[1 0 0]
     [1 1 0]
     [1 1 1]
     [2 1 0]
     [2 1 1]
     [2 2 1]
     [3 1 0]
     [3 1 1]
     [3 2 1]
     [3 2 2]
     [3 3 1]
     [3 3 2]]
    Computing hkl2 list for specific or cubic LUT in matrices_from_onespot_hkl()
    Calculating LUT in PlanePairs_from2sets()
    Looking up planes pairs in LUT from exp. spots (0, 1): 
    Looking up planes pairs in LUT from exp. spots (0, 2): 
    Looking up planes pairs in LUT from exp. spots (0, 3): 
    Looking up planes pairs in LUT from exp. spots (0, 4): 
    Looking up planes pairs in LUT from exp. spots (0, 5): 
    calculating matching rates of solutions for exp. spots [0, 1]
    calculating matching rates of solutions for exp. spots [0, 2]
    calculating matching rates of solutions for exp. spots [0, 3]
    calculating matching rates of solutions for exp. spots [0, 4]
    
    
    return best matrix and matching scores for the one central_spot
    
    -----------------------------------------
    results:
    matrix:                                         matching results
    [-0.211852735694566  0.092255643652867 -0.972937466948891]        res: [20.0, 162.0] 0.014 12.35
    [-0.775856536468367  0.58951816141498   0.22475536073965 ]        spot indices [0 1]
    [ 0.594300563948835  0.802473664571131 -0.053318452339475]        planes [[1.0, 3.0, 2.0], [1.0, 1.0, 1.0]]
    
    Number of matrices found (nb_sol):  1
    set_central_spots_hkl in FindOrientMatrices None
    
    -----------------------------------------
    results:
    matrix:                                         matching results
    [-0.211852735694566  0.092255643652867 -0.972937466948891]        res: [20.0, 162.0] 0.014 12.35
    [-0.775856536468367  0.58951816141498   0.22475536073965 ]        spot indices [0 1]
    [ 0.594300563948835  0.802473664571131 -0.053318452339475]        planes [[1.0, 3.0, 2.0], [1.0, 1.0, 1.0]]
    
    Nb of potential orientation matrice(s) UB found: 1 
    [array([[-0.211852735694566,  0.092255643652867, -0.972937466948891],
           [-0.775856536468367,  0.58951816141498 ,  0.22475536073965 ],
           [ 0.594300563948835,  0.802473664571131, -0.053318452339475]])]
    Nb of potential UBs  1
    
    Working with a new stack of orientation matrices
    MATCHINGRATE_THRESHOLD_IAL= 100.0
    has not been reached! All potential solutions have been calculated
    taking the first one only.
    bestUB object <LaueTools.indexingSpotsSet.OrientMatrix object at 0x7fb8ec7ddc50>
    
    
    ---------------refining grain orientation and strain #0-----------------
    
    
     refining grain #0 step -----0
    
    bestUB <LaueTools.indexingSpotsSet.OrientMatrix object at 0x7fb8ec7ddc50>
    True it is an OrientMatrix object
    Orientation <LaueTools.indexingSpotsSet.OrientMatrix object at 0x7fb8ec7ddc50>
    matrix [[-0.211852735694566  0.092255643652867 -0.972937466948891]
     [-0.775856536468367  0.58951816141498   0.22475536073965 ]
     [ 0.594300563948835  0.802473664571131 -0.053318452339475]]
    ***nb of selected spots in AssignHKL***** 12
    UBOrientMatrix [[-0.211852735694566  0.092255643652867 -0.972937466948891]
     [-0.775856536468367  0.58951816141498   0.22475536073965 ]
     [ 0.594300563948835  0.802473664571131 -0.053318452339475]]
    For angular tolerance 0.50 deg
    Nb of pairs found / nb total of expected spots: 12/176
    Matching Rate : 6.82
    Nb missing reflections: 164
    
    grain #0 : 12 links to simulated spots have been found 
    ***********mean pixel deviation    0.560750282710606     ********
    Initial residues [0.053680370172309 0.013739858524874 0.921977335411896 0.403270956234836
     0.919825854310187 0.785969463406447 0.565019172757509 1.127873079813964
     0.363514793614926 0.412635402450867 0.711008521607465 0.450488584221994]
    ---------------------------------------------------
    
    
    
    ***************************
    first error with initial values of: ['b/a', 'c/a', 'a12', 'a13', 'a23', 'theta1', 'theta2', 'theta3']  
    
    ***************************
    
    ***********mean pixel deviation    0.560750282710606     ********
    
    
    ***************************
    Fitting parameters:   ['b/a', 'c/a', 'a12', 'a13', 'a23', 'theta1', 'theta2', 'theta3'] 
    
    ***************************
    
    With initial values [1. 1. 0. 0. 0. 0. 0. 0.]
    code results 1
    nb iterations 1767
    mesg Both actual and predicted relative reductions in the sum of squares
      are at most 0.000000
    strain_sol [ 1.001128981010799e+00  9.993806401299155e-01  8.449040381845989e-04
     -8.486913595751131e-04  3.520626401662759e-04 -2.714612741167435e-02
      3.054889720130747e-02  5.311773668297186e-02]
    
    
     **************  End of Fitting  -  Final errors  ****************** 
    
    
    ***********mean pixel deviation    0.3158807195732847     ********
    devstrain, lattice_parameter_direct_strain [[ 0.000159671589578 -0.000413843247724  0.00039782267455 ]
     [-0.000413843247724 -0.00095830941256  -0.000151781897557]
     [ 0.00039782267455  -0.000151781897557  0.000798637822982]] [ 5.657424223234738  5.651101185787196  5.661104877237695
     90.01741959362538  89.9544419036642   90.04747664598754 ]
    For comparison: a,b,c are rescaled with respect to the reference value of a = 5.657500 Angstroms
    lattice_parameter_direct_strain [ 5.657499999999999  5.651176877860324  5.661180703302433
     90.01741959362538  89.9544419036642   90.04747664598754 ]
    devstrain1, lattice_parameter_direct_strain1 [[ 0.000159671589578 -0.000413843247724  0.00039782267455 ]
     [-0.000413843247724 -0.00095830941256  -0.000151781897557]
     [ 0.00039782267455  -0.000151781897557  0.000798637822982]] [ 5.657499999999999  5.651176877860324  5.661180703302433
     90.01741959362538  89.9544419036642   90.04747664598754 ]
    new UBs matrix in q= UBs G (s for strain)
    strain_direct [[-1.339403716515974e-05 -4.138432477238585e-04  3.978226745502020e-04]
     [-4.138432477238585e-04 -1.131375039302607e-03 -1.517818975573709e-04]
     [ 3.978226745502020e-04 -1.517818975573709e-04  6.255721962393768e-04]]
    deviatoric strain [[ 0.000159671589578 -0.000413843247724  0.00039782267455 ]
     [-0.000413843247724 -0.00095830941256  -0.000151781897557]
     [ 0.00039782267455  -0.000151781897557  0.000798637822982]]
    new UBs matrix in q= UBs G (s for strain)
    strain_direct [[-1.339403716515974e-05 -4.138432477238585e-04  3.978226745502020e-04]
     [-4.138432477238585e-04 -1.131375039302607e-03 -1.517818975573709e-04]
     [ 3.978226745502020e-04 -1.517818975573709e-04  6.255721962393768e-04]]
    deviatoric strain [[ 0.000159671589578 -0.000413843247724  0.00039782267455 ]
     [-0.000413843247724 -0.00095830941256  -0.000151781897557]
     [ 0.00039782267455  -0.000151781897557  0.000798637822982]]
    For comparison: a,b,c are rescaled with respect to the reference value of a = 5.657500 Angstroms
    lattice_parameter_direct_strain [ 5.657499999999999  5.651176877860324  5.661180703302433
     90.01741959362538  89.9544419036642   90.04747664598754 ]
    final lattice_parameters [ 5.657499999999999  5.651176877860324  5.661180703302433
     90.01741959362538  89.9544419036642   90.04747664598754 ]
    UB and strain refinement completed
    True it is an OrientMatrix object
    Orientation <LaueTools.indexingSpotsSet.OrientMatrix object at 0x7fb8c98baa20>
    matrix [[-0.211415207301911  0.091252946262469 -0.97230572627369 ]
     [-0.77573594328912   0.590041596352251  0.224552051799525]
     [ 0.594613709497768  0.803610914885283 -0.054088075690982]]
    ***nb of selected spots in AssignHKL***** 12
    UBOrientMatrix [[-0.211415207301911  0.091252946262469 -0.97230572627369 ]
     [-0.77573594328912   0.590041596352251  0.224552051799525]
     [ 0.594613709497768  0.803610914885283 -0.054088075690982]]
    For angular tolerance 0.50 deg
    Nb of pairs found / nb total of expected spots: 12/177
    Matching Rate : 6.78
    Nb missing reflections: 165
    
    grain #0 : 12 links to simulated spots have been found 
    GoodRefinement condition is  True
    nb_updates 12 compared to 6
    
    
     refining grain #0 step -----1
    
    bestUB <LaueTools.indexingSpotsSet.OrientMatrix object at 0x7fb8ec7ddc50>
    True it is an OrientMatrix object
    Orientation <LaueTools.indexingSpotsSet.OrientMatrix object at 0x7fb8ec7ddc50>
    matrix [[-0.211852735694566  0.092255643652867 -0.972937466948891]
     [-0.775856536468367  0.58951816141498   0.22475536073965 ]
     [ 0.594300563948835  0.802473664571131 -0.053318452339475]]
    ***nb of selected spots in AssignHKL***** 12
    UBOrientMatrix [[-0.211852735694566  0.092255643652867 -0.972937466948891]
     [-0.775856536468367  0.58951816141498   0.22475536073965 ]
     [ 0.594300563948835  0.802473664571131 -0.053318452339475]]
    For angular tolerance 0.50 deg
    Nb of pairs found / nb total of expected spots: 12/176
    Matching Rate : 6.82
    Nb missing reflections: 164
    
    grain #0 : 12 links to simulated spots have been found 
    ***********mean pixel deviation    0.560750282710606     ********
    Initial residues [0.053680370172309 0.013739858524874 0.921977335411896 0.403270956234836
     0.919825854310187 0.785969463406447 0.565019172757509 1.127873079813964
     0.363514793614926 0.412635402450867 0.711008521607465 0.450488584221994]
    ---------------------------------------------------
    
    
    
    ***************************
    first error with initial values of: ['b/a', 'c/a', 'a12', 'a13', 'a23', 'theta1', 'theta2', 'theta3']  
    
    ***************************
    
    ***********mean pixel deviation    0.560750282710606     ********
    
    
    ***************************
    Fitting parameters:   ['b/a', 'c/a', 'a12', 'a13', 'a23', 'theta1', 'theta2', 'theta3'] 
    
    ***************************
    
    With initial values [1. 1. 0. 0. 0. 0. 0. 0.]
    code results 1
    nb iterations 1767
    mesg Both actual and predicted relative reductions in the sum of squares
      are at most 0.000000
    strain_sol [ 1.001128981010799e+00  9.993806401299155e-01  8.449040381845989e-04
     -8.486913595751131e-04  3.520626401662759e-04 -2.714612741167435e-02
      3.054889720130747e-02  5.311773668297186e-02]
    
    
     **************  End of Fitting  -  Final errors  ****************** 
    
    
    ***********mean pixel deviation    0.3158807195732847     ********
    devstrain, lattice_parameter_direct_strain [[ 0.000159671589578 -0.000413843247724  0.00039782267455 ]
     [-0.000413843247724 -0.00095830941256  -0.000151781897557]
     [ 0.00039782267455  -0.000151781897557  0.000798637822982]] [ 5.657424223234738  5.651101185787196  5.661104877237695
     90.01741959362538  89.9544419036642   90.04747664598754 ]
    For comparison: a,b,c are rescaled with respect to the reference value of a = 5.657500 Angstroms
    lattice_parameter_direct_strain [ 5.657499999999999  5.651176877860324  5.661180703302433
     90.01741959362538  89.9544419036642   90.04747664598754 ]
    devstrain1, lattice_parameter_direct_strain1 [[ 0.000159671589578 -0.000413843247724  0.00039782267455 ]
     [-0.000413843247724 -0.00095830941256  -0.000151781897557]
     [ 0.00039782267455  -0.000151781897557  0.000798637822982]] [ 5.657499999999999  5.651176877860324  5.661180703302433
     90.01741959362538  89.9544419036642   90.04747664598754 ]
    new UBs matrix in q= UBs G (s for strain)
    strain_direct [[-1.339403716515974e-05 -4.138432477238585e-04  3.978226745502020e-04]
     [-4.138432477238585e-04 -1.131375039302607e-03 -1.517818975573709e-04]
     [ 3.978226745502020e-04 -1.517818975573709e-04  6.255721962393768e-04]]
    deviatoric strain [[ 0.000159671589578 -0.000413843247724  0.00039782267455 ]
     [-0.000413843247724 -0.00095830941256  -0.000151781897557]
     [ 0.00039782267455  -0.000151781897557  0.000798637822982]]
    new UBs matrix in q= UBs G (s for strain)
    strain_direct [[-1.339403716515974e-05 -4.138432477238585e-04  3.978226745502020e-04]
     [-4.138432477238585e-04 -1.131375039302607e-03 -1.517818975573709e-04]
     [ 3.978226745502020e-04 -1.517818975573709e-04  6.255721962393768e-04]]
    deviatoric strain [[ 0.000159671589578 -0.000413843247724  0.00039782267455 ]
     [-0.000413843247724 -0.00095830941256  -0.000151781897557]
     [ 0.00039782267455  -0.000151781897557  0.000798637822982]]
    For comparison: a,b,c are rescaled with respect to the reference value of a = 5.657500 Angstroms
    lattice_parameter_direct_strain [ 5.657499999999999  5.651176877860324  5.661180703302433
     90.01741959362538  89.9544419036642   90.04747664598754 ]
    final lattice_parameters [ 5.657499999999999  5.651176877860324  5.661180703302433
     90.01741959362538  89.9544419036642   90.04747664598754 ]
    UB and strain refinement completed
    True it is an OrientMatrix object
    Orientation <LaueTools.indexingSpotsSet.OrientMatrix object at 0x7fb8c98ba9b0>
    matrix [[-0.211415207301911  0.091252946262469 -0.97230572627369 ]
     [-0.77573594328912   0.590041596352251  0.224552051799525]
     [ 0.594613709497768  0.803610914885283 -0.054088075690982]]
    ***nb of selected spots in AssignHKL***** 12
    UBOrientMatrix [[-0.211415207301911  0.091252946262469 -0.97230572627369 ]
     [-0.77573594328912   0.590041596352251  0.224552051799525]
     [ 0.594613709497768  0.803610914885283 -0.054088075690982]]
    For angular tolerance 0.50 deg
    Nb of pairs found / nb total of expected spots: 12/177
    Matching Rate : 6.78
    Nb missing reflections: 165
    
    grain #0 : 12 links to simulated spots have been found 
    GoodRefinement condition is  True
    nb_updates 12 compared to 6
    
    
     refining grain #0 step -----2
    
    bestUB <LaueTools.indexingSpotsSet.OrientMatrix object at 0x7fb8ec7ddc50>
    True it is an OrientMatrix object
    Orientation <LaueTools.indexingSpotsSet.OrientMatrix object at 0x7fb8ec7ddc50>
    matrix [[-0.211852735694566  0.092255643652867 -0.972937466948891]
     [-0.775856536468367  0.58951816141498   0.22475536073965 ]
     [ 0.594300563948835  0.802473664571131 -0.053318452339475]]
    ***nb of selected spots in AssignHKL***** 12
    UBOrientMatrix [[-0.211852735694566  0.092255643652867 -0.972937466948891]
     [-0.775856536468367  0.58951816141498   0.22475536073965 ]
     [ 0.594300563948835  0.802473664571131 -0.053318452339475]]
    For angular tolerance 0.20 deg
    Nb of pairs found / nb total of expected spots: 12/176
    Matching Rate : 6.82
    Nb missing reflections: 164
    
    grain #0 : 12 links to simulated spots have been found 
    ***********mean pixel deviation    0.560750282710606     ********
    Initial residues [0.053680370172309 0.013739858524874 0.921977335411896 0.403270956234836
     0.919825854310187 0.785969463406447 0.565019172757509 1.127873079813964
     0.363514793614926 0.412635402450867 0.711008521607465 0.450488584221994]
    ---------------------------------------------------
    
    
    
    ***************************
    first error with initial values of: ['b/a', 'c/a', 'a12', 'a13', 'a23', 'theta1', 'theta2', 'theta3']  
    
    ***************************
    
    ***********mean pixel deviation    0.560750282710606     ********
    
    
    ***************************
    Fitting parameters:   ['b/a', 'c/a', 'a12', 'a13', 'a23', 'theta1', 'theta2', 'theta3'] 
    
    ***************************
    
    With initial values [1. 1. 0. 0. 0. 0. 0. 0.]
    code results 1
    nb iterations 1767
    mesg Both actual and predicted relative reductions in the sum of squares
      are at most 0.000000
    strain_sol [ 1.001128981010799e+00  9.993806401299155e-01  8.449040381845989e-04
     -8.486913595751131e-04  3.520626401662759e-04 -2.714612741167435e-02
      3.054889720130747e-02  5.311773668297186e-02]
    
    
     **************  End of Fitting  -  Final errors  ****************** 
    
    
    ***********mean pixel deviation    0.3158807195732847     ********
    devstrain, lattice_parameter_direct_strain [[ 0.000159671589578 -0.000413843247724  0.00039782267455 ]
     [-0.000413843247724 -0.00095830941256  -0.000151781897557]
     [ 0.00039782267455  -0.000151781897557  0.000798637822982]] [ 5.657424223234738  5.651101185787196  5.661104877237695
     90.01741959362538  89.9544419036642   90.04747664598754 ]
    For comparison: a,b,c are rescaled with respect to the reference value of a = 5.657500 Angstroms
    lattice_parameter_direct_strain [ 5.657499999999999  5.651176877860324  5.661180703302433
     90.01741959362538  89.9544419036642   90.04747664598754 ]
    devstrain1, lattice_parameter_direct_strain1 [[ 0.000159671589578 -0.000413843247724  0.00039782267455 ]
     [-0.000413843247724 -0.00095830941256  -0.000151781897557]
     [ 0.00039782267455  -0.000151781897557  0.000798637822982]] [ 5.657499999999999  5.651176877860324  5.661180703302433
     90.01741959362538  89.9544419036642   90.04747664598754 ]
    new UBs matrix in q= UBs G (s for strain)
    strain_direct [[-1.339403716515974e-05 -4.138432477238585e-04  3.978226745502020e-04]
     [-4.138432477238585e-04 -1.131375039302607e-03 -1.517818975573709e-04]
     [ 3.978226745502020e-04 -1.517818975573709e-04  6.255721962393768e-04]]
    deviatoric strain [[ 0.000159671589578 -0.000413843247724  0.00039782267455 ]
     [-0.000413843247724 -0.00095830941256  -0.000151781897557]
     [ 0.00039782267455  -0.000151781897557  0.000798637822982]]
    new UBs matrix in q= UBs G (s for strain)
    strain_direct [[-1.339403716515974e-05 -4.138432477238585e-04  3.978226745502020e-04]
     [-4.138432477238585e-04 -1.131375039302607e-03 -1.517818975573709e-04]
     [ 3.978226745502020e-04 -1.517818975573709e-04  6.255721962393768e-04]]
    deviatoric strain [[ 0.000159671589578 -0.000413843247724  0.00039782267455 ]
     [-0.000413843247724 -0.00095830941256  -0.000151781897557]
     [ 0.00039782267455  -0.000151781897557  0.000798637822982]]
    For comparison: a,b,c are rescaled with respect to the reference value of a = 5.657500 Angstroms
    lattice_parameter_direct_strain [ 5.657499999999999  5.651176877860324  5.661180703302433
     90.01741959362538  89.9544419036642   90.04747664598754 ]
    final lattice_parameters [ 5.657499999999999  5.651176877860324  5.661180703302433
     90.01741959362538  89.9544419036642   90.04747664598754 ]
    UB and strain refinement completed
    True it is an OrientMatrix object
    Orientation <LaueTools.indexingSpotsSet.OrientMatrix object at 0x7fb8cf573a90>
    matrix [[-0.211415207301911  0.091252946262469 -0.97230572627369 ]
     [-0.77573594328912   0.590041596352251  0.224552051799525]
     [ 0.594613709497768  0.803610914885283 -0.054088075690982]]
    ***nb of selected spots in AssignHKL***** 12
    UBOrientMatrix [[-0.211415207301911  0.091252946262469 -0.97230572627369 ]
     [-0.77573594328912   0.590041596352251  0.224552051799525]
     [ 0.594613709497768  0.803610914885283 -0.054088075690982]]
    For angular tolerance 0.20 deg
    Nb of pairs found / nb total of expected spots: 12/177
    Matching Rate : 6.78
    Nb missing reflections: 165
    
    grain #0 : 12 links to simulated spots have been found 
    GoodRefinement condition is  True
    nb_updates 12 compared to 6
    
    
     refining grain #0 step -----3
    
    bestUB <LaueTools.indexingSpotsSet.OrientMatrix object at 0x7fb8ec7ddc50>
    True it is an OrientMatrix object
    Orientation <LaueTools.indexingSpotsSet.OrientMatrix object at 0x7fb8ec7ddc50>
    matrix [[-0.211852735694566  0.092255643652867 -0.972937466948891]
     [-0.775856536468367  0.58951816141498   0.22475536073965 ]
     [ 0.594300563948835  0.802473664571131 -0.053318452339475]]
    ***nb of selected spots in AssignHKL***** 12
    UBOrientMatrix [[-0.211852735694566  0.092255643652867 -0.972937466948891]
     [-0.775856536468367  0.58951816141498   0.22475536073965 ]
     [ 0.594300563948835  0.802473664571131 -0.053318452339475]]
    For angular tolerance 0.20 deg
    Nb of pairs found / nb total of expected spots: 12/176
    Matching Rate : 6.82
    Nb missing reflections: 164
    
    grain #0 : 12 links to simulated spots have been found 
    ***********mean pixel deviation    0.560750282710606     ********
    Initial residues [0.053680370172309 0.013739858524874 0.921977335411896 0.403270956234836
     0.919825854310187 0.785969463406447 0.565019172757509 1.127873079813964
     0.363514793614926 0.412635402450867 0.711008521607465 0.450488584221994]
    ---------------------------------------------------
    
    
    
    ***************************
    first error with initial values of: ['b/a', 'c/a', 'a12', 'a13', 'a23', 'theta1', 'theta2', 'theta3']  
    
    ***************************
    
    ***********mean pixel deviation    0.560750282710606     ********
    
    
    ***************************
    Fitting parameters:   ['b/a', 'c/a', 'a12', 'a13', 'a23', 'theta1', 'theta2', 'theta3'] 
    
    ***************************
    
    With initial values [1. 1. 0. 0. 0. 0. 0. 0.]
    code results 1
    nb iterations 1767
    mesg Both actual and predicted relative reductions in the sum of squares
      are at most 0.000000
    strain_sol [ 1.001128981010799e+00  9.993806401299155e-01  8.449040381845989e-04
     -8.486913595751131e-04  3.520626401662759e-04 -2.714612741167435e-02
      3.054889720130747e-02  5.311773668297186e-02]
    
    
     **************  End of Fitting  -  Final errors  ****************** 
    
    
    ***********mean pixel deviation    0.3158807195732847     ********
    devstrain, lattice_parameter_direct_strain [[ 0.000159671589578 -0.000413843247724  0.00039782267455 ]
     [-0.000413843247724 -0.00095830941256  -0.000151781897557]
     [ 0.00039782267455  -0.000151781897557  0.000798637822982]] [ 5.657424223234738  5.651101185787196  5.661104877237695
     90.01741959362538  89.9544419036642   90.04747664598754 ]
    For comparison: a,b,c are rescaled with respect to the reference value of a = 5.657500 Angstroms
    lattice_parameter_direct_strain [ 5.657499999999999  5.651176877860324  5.661180703302433
     90.01741959362538  89.9544419036642   90.04747664598754 ]
    devstrain1, lattice_parameter_direct_strain1 [[ 0.000159671589578 -0.000413843247724  0.00039782267455 ]
     [-0.000413843247724 -0.00095830941256  -0.000151781897557]
     [ 0.00039782267455  -0.000151781897557  0.000798637822982]] [ 5.657499999999999  5.651176877860324  5.661180703302433
     90.01741959362538  89.9544419036642   90.04747664598754 ]
    new UBs matrix in q= UBs G (s for strain)
    strain_direct [[-1.339403716515974e-05 -4.138432477238585e-04  3.978226745502020e-04]
     [-4.138432477238585e-04 -1.131375039302607e-03 -1.517818975573709e-04]
     [ 3.978226745502020e-04 -1.517818975573709e-04  6.255721962393768e-04]]
    deviatoric strain [[ 0.000159671589578 -0.000413843247724  0.00039782267455 ]
     [-0.000413843247724 -0.00095830941256  -0.000151781897557]
     [ 0.00039782267455  -0.000151781897557  0.000798637822982]]
    new UBs matrix in q= UBs G (s for strain)
    strain_direct [[-1.339403716515974e-05 -4.138432477238585e-04  3.978226745502020e-04]
     [-4.138432477238585e-04 -1.131375039302607e-03 -1.517818975573709e-04]
     [ 3.978226745502020e-04 -1.517818975573709e-04  6.255721962393768e-04]]
    deviatoric strain [[ 0.000159671589578 -0.000413843247724  0.00039782267455 ]
     [-0.000413843247724 -0.00095830941256  -0.000151781897557]
     [ 0.00039782267455  -0.000151781897557  0.000798637822982]]
    For comparison: a,b,c are rescaled with respect to the reference value of a = 5.657500 Angstroms
    lattice_parameter_direct_strain [ 5.657499999999999  5.651176877860324  5.661180703302433
     90.01741959362538  89.9544419036642   90.04747664598754 ]
    final lattice_parameters [ 5.657499999999999  5.651176877860324  5.661180703302433
     90.01741959362538  89.9544419036642   90.04747664598754 ]
    UB and strain refinement completed
    True it is an OrientMatrix object
    Orientation <LaueTools.indexingSpotsSet.OrientMatrix object at 0x7fb8ec7bc128>
    matrix [[-0.211415207301911  0.091252946262469 -0.97230572627369 ]
     [-0.77573594328912   0.590041596352251  0.224552051799525]
     [ 0.594613709497768  0.803610914885283 -0.054088075690982]]
    ***nb of selected spots in AssignHKL***** 12
    UBOrientMatrix [[-0.211415207301911  0.091252946262469 -0.97230572627369 ]
     [-0.77573594328912   0.590041596352251  0.224552051799525]
     [ 0.594613709497768  0.803610914885283 -0.054088075690982]]
    For angular tolerance 0.20 deg
    Nb of pairs found / nb total of expected spots: 12/177
    Matching Rate : 6.78
    Nb missing reflections: 165
    
    grain #0 : 12 links to simulated spots have been found 
    GoodRefinement condition is  True
    nb_updates 12 compared to 6
    
    ---------------------------------------------
    indexing completed for grain #0 with matching rate 6.78 
    ---------------------------------------------
    
    transform matrix to matrix with lowest Euler Angles
    start 
     [[-0.211415207301911  0.091252946262469 -0.97230572627369 ]
     [-0.77573594328912   0.590041596352251  0.224552051799525]
     [ 0.594613709497768  0.803610914885283 -0.054088075690982]]
    final 
     [[ 0.97230572627369   0.211415207301911  0.091252946262469]
     [-0.224552051799525  0.77573594328912   0.590041596352251]
     [ 0.054088075690982 -0.594613709497768  0.803610914885283]]
    hkl [[2. 6. 4.]
     [3. 3. 3.]
     [5. 3. 3.]
     [1. 3. 5.]
     [6. 2. 4.]
     [5. 1. 3.]
     [1. 3. 3.]
     [6. 4. 6.]
     [4. 2. 6.]
     [4. 2. 2.]
     [3. 5. 3.]
     [2. 2. 4.]]
    new hkl (min euler angles) [[-4. -2.  6.]
     [-3. -3.  3.]
     [-3. -5.  3.]
     [-5. -1.  3.]
     [-4. -6.  2.]
     [-3. -5.  1.]
     [-3. -1.  3.]
     [-6. -6.  4.]
     [-6. -4.  2.]
     [-2. -4.  2.]
     [-3. -3.  5.]
     [-4. -2.  2.]]
    UB before [[-0.211415207301911  0.091252946262469 -0.97230572627369 ]
     [-0.77573594328912   0.590041596352251  0.224552051799525]
     [ 0.594613709497768  0.803610914885283 -0.054088075690982]]
    new UB (min euler angles) [[ 0.97230572627369   0.211415207301911  0.091252946262469]
     [-0.224552051799525  0.77573594328912   0.590041596352251]
     [ 0.054088075690982 -0.594613709497768  0.803610914885283]]
    writing fit file -------------------------
    for grainindex= 0
    self.dict_grain_matrix[grain_index] [[ 0.97230572627369   0.211415207301911  0.091252946262469]
     [-0.224552051799525  0.77573594328912   0.590041596352251]
     [ 0.054088075690982 -0.594613709497768  0.803610914885283]]
    self.refinedUBmatrix [[-0.211415207301911  0.091252946262469 -0.97230572627369 ]
     [-0.77573594328912   0.590041596352251  0.224552051799525]
     [ 0.594613709497768  0.803610914885283 -0.054088075690982]]
    new UBs matrix in q= UBs G (s for strain)
    strain_direct [[-1.339403716515974e-05 -4.138432477238585e-04  3.978226745502020e-04]
     [-4.138432477238585e-04 -1.131375039302607e-03 -1.517818975573709e-04]
     [ 3.978226745502020e-04 -1.517818975573709e-04  6.255721962393768e-04]]
    deviatoric strain [[ 0.000159671589578 -0.000413843247724  0.00039782267455 ]
     [-0.000413843247724 -0.00095830941256  -0.000151781897557]
     [ 0.00039782267455  -0.000151781897557  0.000798637822982]]
    new UBs matrix in q= UBs G (s for strain)
    strain_direct [[-1.339403716515974e-05 -4.138432477238585e-04  3.978226745502020e-04]
     [-4.138432477238585e-04 -1.131375039302607e-03 -1.517818975573709e-04]
     [ 3.978226745502020e-04 -1.517818975573709e-04  6.255721962393768e-04]]
    deviatoric strain [[ 0.000159671589578 -0.000413843247724  0.00039782267455 ]
     [-0.000413843247724 -0.00095830941256  -0.000151781897557]
     [ 0.00039782267455  -0.000151781897557  0.000798637822982]]
    For comparison: a,b,c are rescaled with respect to the reference value of a = 5.657500 Angstroms
    lattice_parameter_direct_strain [ 5.657499999999999  5.651176877860324  5.661180703302433
     90.01741959362538  89.9544419036642   90.04747664598754 ]
    final lattice_parameters [ 5.657499999999999  5.651176877860324  5.661180703302433
     90.01741959362538  89.9544419036642   90.04747664598754 ]
    File : Ge0001_g0.fit written in /home/micha/LaueToolsPy3/LaueTools/notebooks
    Experimental experimental spots indices which are not indexed []
    Missing reflections grainindex is -100 for indexed grainindex 0
    within angular tolerance 0.500
    
     Remaining nb of spots to index for grain #1 : 0
    
    12 spots have been indexed over 12
    indexing rate is --- : 100.0 percents
    indexation of short_Ge0001.cor is completed
    for the 1 grain(s) that has(ve) been indexed as requested
    Leaving Index and Refine procedures...


.. code:: ipython3

    index_grain_retrieve=0
    print("number of indexed spots", len(DataSet.getallIndexedSpotsallData()[index_grain_retrieve]))


.. parsed-literal::

    number of indexed spots 12


Results of indexation can be found in attributes or through methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    spotsdata=DataSet.getSummaryallData()
    print("first 2 indexed spots properties\n")
    print('#spot  #grain 2theta chi X Y I h k l Energy')
    print(spotsdata[:2])


.. parsed-literal::

    first 2 indexed spots properties
    
    #spot  #grain 2theta chi X Y I h k l Energy
    [[ 0.000000000000000e+00  0.000000000000000e+00  6.035945800000000e+01
       2.648319100000000e+01  6.261200000000000e+02  1.661280000000000e+03
       1.582539000000000e+04 -4.000000000000000e+00 -2.000000000000000e+00
       6.000000000000000e+00  1.632366988464985e+01]
     [ 1.000000000000000e+00  0.000000000000000e+00  7.821582100000001e+01
       1.638153000000000e+00  1.027110000000000e+03  1.293280000000000e+03
       7.093127000000000e+04 -3.000000000000000e+00 -3.000000000000000e+00
       3.000000000000000e+00  9.031851548397370e+00]]


.. code:: ipython3

    print('#grain  : [Npairs = Nb pairs with tolerance angle %.4f, 100*Npairs/Ndirections theo.]'%dict_loop['list matching tol angles'][-1])
    DataSet.dict_grain_matching_rate



.. parsed-literal::

    #grain  : [Npairs = Nb pairs with tolerance angle 0.2000, 100*Npairs/Ndirections theo.]




.. parsed-literal::

    {0: [12, 6.779661016949152]}



.. code:: ipython3

    print("#grain  : deviatoric strain")
    DataSet.dict_grain_devstrain


.. parsed-literal::

    #grain  : deviatoric strain




.. parsed-literal::

    {0: array([[ 0.000159671589578, -0.000413843247724,  0.00039782267455 ],
            [-0.000413843247724, -0.00095830941256 , -0.000151781897557],
            [ 0.00039782267455 , -0.000151781897557,  0.000798637822982]])}



.. code:: ipython3

    #RefinedUB= DataSet.dict_grain_matrix
    print("#grain  : refined UB matrix")
    DataSet.dict_grain_matrix
    



.. parsed-literal::

    #grain  : refined UB matrix




.. parsed-literal::

    {0: array([[ 0.97230572627369 ,  0.211415207301911,  0.091252946262469],
            [-0.224552051799525,  0.77573594328912 ,  0.590041596352251],
            [ 0.054088075690982, -0.594613709497768,  0.803610914885283]])}



.. code:: ipython3

    print([DataSet.indexed_spots_dict[k] for k in range(10)])


.. parsed-literal::

    [[0, 60.359458, 26.483191, 626.12, 1661.28, 15825.39, array([-4., -2.,  6.]), 16.323669884649853, 0, 1], [1, 78.215821, 1.638153, 1027.11, 1293.28, 70931.27, array([-3., -3.,  3.]), 9.03185154839737, 0, 1], [2, 68.680451, -15.358122, 1288.11, 1460.16, 22795.07, array([-3., -5.,  3.]), 12.73794897550435, 0, 1], [3, 108.452917, 37.749461, 383.77, 754.58, 4400.61, array([-5., -1.,  3.]), 7.989738078276174, 0, 1], [4, 83.349535, -27.458061, 1497.4, 1224.7, 13145.99, array([-4., -6.,  2.]), 12.327936233758937, 0, 1], [5, 82.072076, -35.89243, 1672.67, 1258.62, 13318.81, array([-3., -5.,  1.]), 9.870774398536632, 0, 1], [6, 81.771257, 30.38247, 548.25, 1260.32, 6137.87, array([-3., -1.,  3.]), 7.2986772558942405, 0, 1], [7, 91.798221, -8.309941, 1176.09, 1086.19, 11799.93, array([-6., -6.,  4.]), 14.309911950813975, 0, 1], [8, 120.59561, -8.92066, 1183.27, 598.92, 17182.88, array([-6., -4.,  2.]), 9.435284238042113, 0, 1], [9, 64.329767, -20.824155, 1379.17, 1553.58, 51933.84, array([-2., -4.,  2.]), 10.087272916663885, 0, 1]]

