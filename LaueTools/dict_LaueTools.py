# -*- coding: utf-8 -*-
"""
Dictionary of several parameters concerning Detectors, Materials, Transforms etc
that are used in LaueTools and in LaueToolsGUI module

Lauetools project
2009 - October 2024

"""
__author__ = "Jean-Sebastien Micha, CRG-IF BM32 @ ESRF"

import copy
import os
import re

import numpy as np

LAUETOOLSFOLDER = os.path.split(__file__)[0]
# print("LaueToolsProjectFolder", LAUETOOLSFOLDER)
# WRITEFOLDER = os.path.join(LAUETOOLSFOLDER, "laueanalysis")

##------------------------------------------------------------------------
# --- -----------  Element-materials library
#-------------------------------------------------------------------------
# label, a,b,c,alpha, beta, gamma  in real space lattice, extinction rules label
dict_Materials = {
    "AmbiguousTriclinic": ["AmbiguousTriclinic", [3.9, 4, 4.1, 89, 90, 91], "no"],  # confirmed by IM2NP
    "Ag": ["Ag", [4.085, 4.085, 4.085, 90, 90, 90], "fcc"],  # confirmed by IM2NP
    "Al2O3": ["Al2O3", [4.785, 4.785, 12.991, 90, 90, 120], "Al2O3"],  # 293 K
    "Al2O3_600K": ["Al2O3_600K", [4.7711, 4.7711, 13.0285, 90, 90, 120], "Al2O3"],  # 600 K
    "Al2O3_1000K": ["Al2O3_1000K", [4.7852, 4.7852, 13.07092, 90, 90, 120], "Al2O3"],  # 1000 K
    "Al2O3_1500K": ["Al2O3_1500K", [4.8051, 4.8051, 13.1308, 90, 90, 120], "Al2O3"],  # 1500 K
    "Al2O3_all": ["Al2O3_all", [4.785, 4.785, 12.991, 90, 90, 120], "no"],
    "Al": ["Al", [4.05, 4.05, 4.05, 90, 90, 90], "fcc"],
    "Al2Cu": ["Al2Cu", [6.063, 6.063, 4.872, 90, 90, 90], "no"],
    "AlN": ["AlN", [3.11, 3.11, 4.98, 90.0, 90.0, 120.0], "wurtzite"],
    "Fe": ["Fe", [2.856, 2.856, 2.856, 90, 90, 90], "bcc"],
    "FeAl": ["FeAl", [5.871, 5.871, 5.871, 90, 90, 90], "fcc"],
    "Fe2Ta": ["Fe2Ta", [4.83, 4.83, 0.788, 90, 90, 120], "no"],
    "Si": ["Si", [5.4309, 5.4309, 5.4309, 90, 90, 90], "dia"],
    "3H-SiC": ["3H-SiC", [4.3596, 4.3596, 4.3596, 90, 90, 90], "dia"],  # zinc blende  SG 216
    "4H-SiC": ["4H-SiC", [3.073, 3.073, 10.053, 90, 90, 120], "wurtzite"],  # wurtzite  = 6H-SiC also  SG 186
    "Cazadero": ["Cazadero", [11.637, 11.637, 11.6443, 90, 90, 90], "SG141"], # normally SG142
    "EastAlps": ["EastAlps", [11.5935, 11.5935, 11.5987, 90, 90, 90], "SG141"], # normally SG142
    "CubicGarnet": ["CubicGarnet", [11.6,11.6,11.6, 90, 90, 90], "no"], # fictius garnet
    "CdHgTe": ["CdHgTe", [6.46678, 6.46678, 6.46678, 90, 90, 90], "dia"],
    "CdHgTe_fcc": ["CdHgTe_fcc", [6.46678, 6.46678, 6.46678, 90, 90, 90], "fcc"],
    "Cr": ["Cr", [2.87, 2.87, 2.87, 90, 90, 90], "bcc"],  # 229
    "Ge": ["Ge", [5.6575, 5.6575, 5.6575, 90, 90, 90], "dia"],
    "Getest": ["Getest", [5.6575, 5.6575, 5.6574, 90, 90, 90], "dia", ],  # c is slightly lower
    "Au": ["Au", [4.078, 4.078, 4.078, 90, 90, 90], "fcc"],
    "Ge_s": ["Ge_s", [5.6575, 5.6575, 5.6575, 90, 90, 89.5], "dia", ],  # Ge a bit strained
    "Ge_compressedhydro": ["Ge_compressedhydro", [5.64, 5.64, 5.64, 90, 90, 90.0], "dia", ],  # Ge compressed hydrostatically
    "GaAs": ["GaAs", [5.65325, 5.65325, 5.65325, 90, 90, 90], "dia"],  # AsGa
    "GaAs_wurtz": ["GaAs_wurtz", [5.65325, 5.65325, 5.9, 90, 90, 90], "wurtzite"],  # AsGa
    "ZrUO2_corium": ["ZrUO2_corium", [5.47, 5.47, 5.47, 90, 90, 90], "fcc"],
    "Cu": ["Cu", [3.6, 3.6, 3.6, 90, 90, 90], "fcc"],
    "Crocidolite": ["Crocidolite", [9.811, 18.013, 5.326, 90, 103.68, 90], "no", ],  # a= 9.811, b=18.013, c= 5.326A, beta=103,68°
    "Crocidolite_2": ["Crocidolite_2", [9.76, 17.93, 5.35, 90, 103.6, 90], "no", ],  # a= 9.811, b=18.013, c= 5.326A, beta=103,68°
    "Crocidolite_2_72deg": ["Crocidolite_2", [9.76, 17.93, 5.35, 90, 76.4, 90], "no", ],  # a= 9.811, b=18.013, c= 5.326A, beta=103,68°
    "Crocidolite_whittaker_1949": ["Crocidolite_whittaker_1949", [9.89, 17.85, 5.31, 90, 180 - 72.5, 90], "no"],
    "CCDL1949": ["CCDL1949", [9.89, 17.85, 5.31, 90, 180 - 72.5, 90], "h+k=2n"],
    "Crocidolite_small": ["Crocidolite_small", [9.76 / 3, 17.93 / 3, 5.35 / 3, 90, 103.6, 90], "no", ],  # a= 9.811, b=18.013, c= 5.326A, beta=103,68°
    "Hematite": ["Hematite", [5.03459, 5.03459, 13.7533, 90, 90, 120], "no", ],  # extinction for h+k+l=3n and always les l=2n
    "CuZnAl_betabrass": ["CuZnAl_betabrass", [4.40, 5.3, 3.8, 90, 88, 90], "no"],# 
    "CuZnAl_betabrass_light": ["CuZnAl_betabrass_light", [4.40, 5.3, 3.8, 90, 88, 90], "VO2_mono"],# 
    "CuZnAl_tetra": ["CuZnAl_tetra", [4.1, 4.1, 5.8, 90, 90, 90], "no"],# 
    "CuZnAl_6M1": ["CuZnAl_6M1", [4.5, 5.45, 1.3, 94.2, 90, 90], "no"],# 
    "Mg":["Mg",[3.2095,3.2095,5.2104,90,90,120],"no"],
    "CaF2": ["CaF2", [5.463, 5.463, 5.463, 90, 90, 90], "fcc"],  # GS 225 fcc 
    "Magnetite_fcc": ["Magnetite_fcc", [8.391, 8.391, 8.391, 90, 90, 90], "fcc"],  # GS 225 fcc extinction
    "Magnetite": ["Magnetite", [8.391, 8.391, 8.391, 90, 90, 90], "dia"],  # GS 227
    "Magnetite_sc": ["Magnetite_sc", [8.391, 8.391, 8.391, 90, 90, 90], "no"],  # no extinction
    "MgO": ["MgO", [4.211, 4.211, 4.211, 90, 90, 90], "fcc"],
    "MoO3": ["alpha_MoO3", [3.76, 3.97, 14.432, 90, 90, 90], "no"],  # SG62
    "literature_MoO3": ["literature_MoO3", [3.76, 3.97, 14.43, 90.0, 90.0, 90.0], "no"], #sg62
    "PMNPT": ["PMNPT", [3.9969, 3.9969, 4.0457, 90, 90, 90], "no"],# sg99
    "NiTi": ["NiTi", [3.5506, 3.5506, 3.5506, 90, 90, 90], "fcc"],
    "Ni": ["Ni", [3.5238, 3.5238, 3.5238, 90, 90, 90], "fcc"],
    "NiO": ["NiO", [2.96, 2.96, 7.23, 90, 90, 120], "no"],
    "NiMnIn_modulated": ["NiMnIn_modulated", [4.3919, 5.6202, 4.3315, 90, 93.044, 90], "h+k=2n, modulated"],  # average I2/m SG 12 # modulated structure Yan et al Acta Mat 88 (2015) 375
    "NiMnIn": ["NiMnIn", [4.3919, 5.6202, 4.3315, 90, 93.044, 90], "h+k=2n"],  # I2/m SG 12 
    "NiMnIn_2ac": ["NiMnIn_2ac", [2*4.3919, 5.6202, 2*4.3315, 90, 93.044, 90], "2h+k=2n"],  # I2/m SG 12 
    "Nb14W3O44": ["Nb14W3O44", [21.002, 21.002, 3.82, 90, 90, 90], "no"],  #space group 82 to be checked
    "Olivine_forsterite": ["Olivine_forsterite", [4.754,10.1971,5.9806, 90, 90, 90], "no"],
    "Olivine_fayalite": ["Olivine_fayalite", [4.8211,10.4779,6.0889, 90, 90, 90], "no"],
    "Olivine_mantle": ["Olivine_mantle", [4.7646,10.2296,5.9942, 90, 90, 90], "no"],
    "dummy": ["dummy", [4.0, 8.0, 2.0, 90, 90, 90], "no"],
    "CdTe": ["CdTe", [6.487, 6.487, 6.487, 90, 90, 90], "fcc"],
    "CdTeDiagB": ["CdTeDiagB", [4.5721, 7.9191, 11.1993, 90, 90, 90], "no"],
    "DarinaMolecule": ["DarinaMolecule", [9.4254, 13.5004, 13.8241, 61.83, 84.555, 75.231], "no", ],
    # 'NbSe3' :['NbSe3',[10.006, 3.48, 15.629],'cubic'], # monoclinic structure, angle beta = 109.5 must be input in grain definition
    "UO2": ["UO2", [5.47, 5.47, 5.47, 90, 90, 90], "fcc"],
    "ZrO2Y2O3": ["ZrO2Y2O3", [5.1378, 5.1378, 5.1378, 90, 90, 90], "fcc"],
    "ZrO2": ["ZrO2", [5.1505, 5.2116, 5.3173, 90, 99.23, 90], "VO2_mono"],
    "ZrO2fake1": ["ZrO2fake1", [5.1505, 5.048116, 4.988933, 90, 99.23, 90], "VO2_mono"],
    "ZrO2swapac": ["ZrO2swapac", [5.3173, 5.2116, 5.1505, 90, 99.23, 90], "VO2_mono"],
    "ZrO2_1200C": ["ZrO2_1200C", [3.6406,3.6406,5.278,90,90,90], "h+k+l=2n"],
    "DIA": ["DIA", [5.0, 5.0, 5.0, 90, 90, 90], "dia", ],  #  small lattice Diamond like Structure
    "DIAs": ["DIAs", [3.56683, 3.56683, 3.56683, 90, 90, 90], "dia", ],  #  small lattice Diamond material Structure
    "FCC": ["FCC", [5.0, 5.0, 5.0, 90, 90, 90], "fcc"],  # small lattice fcc Structure
    "SC": ["SC", [1.0, 1.0, 1.0, 90, 90, 90], "no"],  # 1Ang simple cubic Structure
    "SC5": ["SC5", [5.0, 5.0, 5.0, 90, 90, 90], "no"],  # 5Ang simple cubic Structure
    "SC7": ["SC7", [7.0, 7.0, 7.0, 90, 90, 90], "no"],  # 7Ang simple cubic Structure
    "W": ["W", [3.1652, 3.1652, 3.1652, 90, 90, 90], "bcc"],
    "testindex": ["testindex", [2.0, 1.0, 4.0, 90, 90, 90], "no"],
    "testindex2": ["testindex2", [2.0, 1.0, 4.0, 75, 90, 120], "no"],
    "Ti": ["Ti", [2.95, 2.95, 4.68, 90, 90, 120], "no"],
    "Ti2AlN_w": ["Ti2AlN_w", [2.989, 2.989, 13.624, 90, 90, 120], "wurtzite"],
    "Ti2AlN": ["Ti2AlN", [2.989, 2.989, 13.624, 90, 90, 120], "Ti2AlN"],
    "Ti_beta": ["Ti_beta", [3.2587, 3.2587, 3.2587, 90, 90, 90], "bcc"],
    "Ti_omega": ["Ti_omega", [4.6085, 4.6085, 2.8221, 90, 90, 120], "no"],
    "alphaQuartz": ["alphaQuartz", [4.9, 4.9, 5.4, 90, 90, 120], "no"],
    "betaQuartznew": ["betaQuartznew", [4.9, 4.9, 6.685, 90, 90, 120], "no"],
    "GaN": ["GaN", [3.189, 3.189, 5.185, 90, 90, 120], "wurtzite"],
    "GaN_strained_test": ["GaN_strained_test", [3.185, 3.2, 5.17, 90.2, 89.85, 120.3], "wurtzite"],
    "GaN_all": ["GaN_all", [3.189, 3.189, 5.185, 90, 90, 120], "no"],
    "GaN_PG_sub": ["GaN_PG_sub", [3.185, 3.185, 5.189, 90, 90, 120], "wurtzite"],
    "InGaN_PG": ["InGaN_PG", [3.218, 3.218, 5.255, 90, 90, 120], "wurtzite"],
    "In": ["In", [3.2517, 3.2517, 4.9459, 90, 90, 90], "h+k+l=2n"],
    "In_distorted": ["In_distorted", [3.251700, 3.251133, 4.818608, 89.982926, 90.007213, 95.379102],"h+k+l=2n"],
    "InN": ["InN", [3.533, 3.533, 5.693, 90, 90, 120], "wurtzite"],
    "In2Bi": ["In2Bi", [5.496, 5.496, 6.585, 90, 90, 120], "no"],  # GS 194
    "In_epsilon": ["In_epsilon", [3.47, 3.47, 4.49, 90, 90, 90], "no"], #"h+k+l=2n"],  # GS 
    "InGaN": ["InGaN", [(3.533 + 3.189) / 2.0, (3.533 + 3.189) / 2.0, (5.693 + 5.185) / 2.0, 90, 90, 120, ], "wurtzite", ],  # wegard's law
    "Ti_s": ["Ti_s", [3.0, 3.0, 4.7, 90.5, 89.5, 120.5], "no"],  # Ti strained
    "inputB": ["inputB", [1.0, 1.0, 1.0, 90, 90, 90], "no"],
    "bigpro": ["bigpro", [112.0, 112.0, 136.0, 90, 90, 90], "no"],  # big protein
    "smallpro": ["smallpro", [20.0, 4.8, 49.0, 90, 90, 90], "no"],  # small protein
    "Nd45": ["Nd45", [5.4884, 5.4884, 5.4884, 90, 90, 90], "fcc"],
    "YAG": ["YAG", [9.2, 9.2, 9.2, 90, 90, 90], "no"],
    "Cu6Sn5_tetra": ["Cu6Sn5_tetra", [3.608, 3.608, 5.037, 90, 90, 90], "no"],
    "Cu6Sn5_monoclinic": ["Cu6Sn5_monoclinic", [11.02, 7.28, 9.827, 90, 98.84, 90], "no", ],
    "Sn_beta": ["Sn_beta", [5.83, 5.83, 3.18, 90, 90, 90], "SG141"],
    "Sn_beta_all": ["Sn_all", [5.83, 5.83, 3.18, 90, 90, 90], "no"], 
    "Sb": ["Sb", [4.3, 4.3, 11.3, 90, 90, 120], "no"],
    "STO": ["STO", [3.95, 3.95, 3.95, 90, 90, 90], "no"],  # SrTIO3
    "SRO": ["STO", [3.99, 3.99, 3.99, 90, 90, 90], "no"],  # 
    "quartz_alpha": ["quartz_alpha", [4.913, 4.913, 5.404, 90, 90, 120], "no"],
    "ferrydrite": ["ferrydrite", [2.96, 2.96, 9.4, 90, 90, 120], "no"],
    "feldspath": ["feldspath", [8.59, 12.985, 7.213, 90, 116., 90], 'no'],
    "hexagonal": ["hexagonal", [1.0, 1.0, 3.0, 90, 90, 120.0], "no"],
    "ZnO": ["ZnO", [3.252, 3.252, 5.213, 90, 90, 120], "wurtzite"],
    "test_reference": ["test_reference", [3.2, 4.5, 5.2, 83, 92.0, 122], "wurtzite"],
    "test_solution": ["test_solution", [3.252, 4.48, 5.213, 83.2569, 92.125478, 122.364], "wurtzite",],
    "Y2SiO5": ["Y2SiO5", [10.34, 6.689, 12.38, 90.0, 102.5, 90.0], "no", ],  # SG 15  I2/a
    "VO2M1": ["VO2M1", [5.75175, 4.52596, 5.38326, 90.0, 122.6148, 90.0], "VO2_mono", ],  # SG 14
    "VO2M2": ["VO2M2", [4.5546, 4.5546, 2.8514, 90.0, 90, 90.0], "no"], # SG 136 (87 deg Celsius)  Rutile
    "VO2R": ["VO2R", [4.5546, 4.5546, 2.8514, 90.0, 90, 90.0], "rutile"], # SG 136 (87 deg Celsius)  Rutile
    "ZnCuOCl": ["ZnCuOCl", [6.839, 6.839, 14.08, 90.0, 90, 120.0], "SG166"],
    "ZnCuOCl_all": ["ZnCuOCl_all", [6.839, 6.839, 14.08, 90.0, 90, 120.0], "no"],
    "FePS3": ["FePS3", [5, 10, 7, 90, 107, 90], "no"],
    "Zr": ["Zr", [3.24, 3.24, 5.17, 90, 90, 120], "no"],  # 194 
    "Nb14W3O44": ["Nb14W3O44", [21.03,21.03,3.93,90, 90, 90], "h+k+l=2n"], #SG 82
    "tourmaline" : ["tourmaline", [15.963, 15.963 , 7.148, 90, 90, 120], "R3m_sg160"],
    "CaWO4": ["CaWO4", [5.26, 5.26, 11.35, 90, 90, 90], "h+k+l=2n"], # SG88
    "CaWO4_distorted": ["CaWO4_distorted", [5.26, 5.26, 11.4, 90, 90, 90], "h+k+l=2n"],
    "LTO_a321204": ["LTO_a321204", [5.13, 5.45, 6.35, 87.20, 89.38, 118.09], "R3m_sg160"],
    "LTO_1": ["LTO_1", [5.13, 5.13, 13.82, 90, 90, 120], "R3m_sg160"],
    "LTO_half": ["LTO_half", [5.13, 5.13, 6.91, 90, 90, 120], "R3m_sg160"],
    "LTO_all": ["LTO_all", [5.13, 5.13, 13.82, 90, 90, 120], "no"],
    "fcc_mystere": ["fcc_mystere", [5.33, 5.33, 5.23, 90, 91, 90], "fcc"],
    "AAA_mystere": ["AAA_mystere", [7.89, 7.89, 7.89 , 93.5, 93.5, 86.5], "fcc"],
    "LTO_distor": ["LTO_distor", [10.869, 13.655,  8.459, 92.653, 63.159, 86.865], "fcc"],
    "LTO_distor_all": ["LTO_distor_all", [10.869, 13.655,  8.459, 92.653, 63.159, 86.865], "no"],
    "LTO_distor_al2o3_ext": ["LTO_distor_al2o3_ext", [10.869, 13.655,  8.459, 92.653, 63.159, 86.865], "Al2O3"],
    "LTO_distor_bulk": ["LTO_distor_bulk", [5.130, 4.830,  11.958, 90.00, 92.80, 118.08], "R3m_sg160"],
    "LTO_distor_bulk_fit": ["LTO_distor_bulk_fit", [5.13, 4.8301684, 5.9791912, 89.9987506, 92.8008151, 118.0844543], "R3m_sg160"]
}

dict_Materials_short = {
    "Al2O3": ["Al2O3", [4.785, 4.785, 12.991, 90, 90, 120], "Al2O3"],
    "Al2O3_all": ["Al2O3_all", [4.785, 4.785, 12.991, 90, 90, 120], "no"],
    "Al": ["Al", [4.05, 4.05, 4.05, 90, 90, 90], "fcc"],
    "Al13Fe4": ["Al13Fe4", [15.492, 8.078,12.471, 90, 107.69, 90], "no"], 
    "Al2Cu": ["Al2Cu", [6.063, 6.063, 4.872, 90, 90, 90], "no"],
    "AlN": ["AlN", [3.11, 3.11, 4.98, 90.0, 90.0, 120.0], "wurtzite"],
    "Fe": ["Fe", [2.856, 2.856, 2.856, 90, 90, 90], "bcc"],
    "Si": ["Si", [5.4309, 5.4309, 5.4309, 90, 90, 90], "dia"],
    "CdHgTe": ["CdHgTe", [6.46678, 6.46678, 6.46678, 90, 90, 90], "dia"],
    "CdHgTe_fcc": ["CdHgTe_fcc", [6.46678, 6.46678, 6.46678, 90, 90, 90], "fcc"],
    "Ge": ["Ge", [5.6575, 5.6575, 5.6575, 90, 90, 90], "dia"],
    "Au": ["Au", [4.078, 4.078, 4.078, 90, 90, 90], "fcc"],
    "GaAs": ["GaAs", [5.65325, 5.65325, 5.65325, 90, 90, 90], "dia"],  # AsGa
    "Cu": ["Cu", [3.6, 3.6, 3.6, 90, 90, 90], "fcc"],
    "Crocidolite_whittaker_1949": ["Crocidolite_whittaker_1949", [9.89, 17.85, 5.31, 90, 180 - 72.5, 90], "no", ],
    "Hematite": ["Hematite", [5.03459, 5.03459, 13.7533, 90, 90, 120], "no", ],  # extinction for h+k+l=3n and always les l=2n
    "Magnetite_fcc": ["Magnetite_fcc", [8.391, 8.391, 8.391, 90, 90, 90], "fcc", ],  # GS 225 fcc extinction
    "Magnetite": ["Magnetite", [8.391, 8.391, 8.391, 90, 90, 90], "dia"],  # GS 227
    "Magnetite_sc": ["Magnetite_sc", [8.391, 8.391, 8.391, 90, 90, 90], "no", ],  # no extinction
    "NiTi": ["NiTi", [3.5506, 3.5506, 3.5506, 90, 90, 90], "fcc"],
    "Ni": ["Ni", [3.5238, 3.5238, 3.5238, 90, 90, 90], "fcc"],
    "NiO": ["NiO", [2.96, 2.96, 7.23, 90, 90, 120], "no"],
    "CdTe": ["CdTe", [6.487, 6.487, 6.487, 90, 90, 90], "fcc"],
    'NbSe3' :['NbSe3', [10.006, 3.48, 15.629], 'cubic'], # monoclinic structure, angle beta = 109.5 must be input in grain definition
    "UO2": ["UO2", [5.47, 5.47, 5.47, 90, 90, 90], "fcc"],
    "ZrO2Y2O3": ["ZrO2Y2O3", [5.1378, 5.1378, 5.1378, 90, 90, 90], "fcc"],
    "ZrO2": ["ZrO2", [5.1505, 5.2116, 5.3173, 90, 99.23, 90], "VO2_mono"],
    "DIAs": ["DIAs", [3.56683, 3.56683, 3.56683, 90, 90, 90], "dia"],  #  small lattice Diamond material Structure
    "W": ["W", [3.1652, 3.1652, 3.1652, 90, 90, 90], "bcc"],
    "Ti": ["Ti", [2.95, 2.95, 4.68, 90, 90, 120], "no"],
    "Ti2AlN_w": ["Ti2AlN_w", [2.989, 2.989, 13.624, 90, 90, 120], "wurtzite"],
    "Ti2AlN": ["Ti2AlN", [2.989, 2.989, 13.624, 90, 90, 120], "Ti2AlN"],
    "Ti_beta": ["Ti_beta", [3.2587, 3.2587, 3.2587, 90, 90, 90], "bcc"],
    "Ti_omega": ["Ti_omega", [4.6085, 4.6085, 2.8221, 90, 90, 120], "no"],
    "alphaQuartz": ["alphaQuartz", [4.9, 4.9, 5.4, 90, 90, 120], "no"],
    "betaQuartznew": ["betaQuartznew", [4.9, 4.9, 6.685, 90, 90, 120], "no"],
    "GaN": ["GaN", [3.189, 3.189, 5.185, 90, 90, 120], "wurtzite"],
    "GaN_all": ["GaN_all", [3.189, 3.189, 5.185, 90, 90, 120], "no"],
    "In": ["In", [3.2517, 3.2517, 4.9459, 90, 90, 90], "h+k+l=2n"],
    "InN": ["InN", [3.533, 3.533, 5.693, 90, 90, 120], "wurtzite"],
    "InGaN": ["InGaN", [(3.533 + 3.189) / 2.0, (3.533 + 3.189) / 2.0, (5.693 + 5.185) / 2.0, 90, 90, 120, ], "wurtzite"],  # wegard's law
    "bigpro": ["bigpro", [112.0, 112.0, 136.0, 90, 90, 90], "no"],  # big protein
    "smallpro": ["smallpro", [20.0, 4.8, 49.0, 90, 90, 90], "no"],  # small protein
    "Nd45": ["Nd45", [5.4884, 5.4884, 5.4884, 90, 90, 90], "fcc"],
    "YAG": ["YAG", [9.2, 9.2, 9.2, 90, 90, 90], "no"],
    "Sn": ["Sn", [5.83, 5.83, 3.18, 90, 90, 90], "h+k+l=2n"], # we may add hhl: 2h+l=4n
    "Sb": ["Sb", [4.3, 4.3, 11.3, 90, 90, 120], "no"],
    "quartz_alpha": ["quartz_alpha", [4.913, 4.913, 5.404, 90, 90, 120], "no"],
    "ferrydrite": ["ferrydrite", [2.96, 2.96, 9.4, 90, 90, 120], "no"],
    "ZnO": ["ZnO", [3.252, 3.252, 5.213, 90, 90, 120], "wurtzite"],
    "FePS3": ["FePS3", [5,10,7, 90, 107, 90], "no"],
    "VO2M1": ["VO2M1", [5.75175, 4.52596, 5.38326, 90.0, 122.6148, 90.0], "VO2_mono", ],  # SG 14
    "VO2M2": ["VO2M2", [4.5546, 4.5546, 2.8514, 90.0, 90, 90.0], "no", ],  # SG 136 (87 deg Celsius)  Rutile
    "VO2R": ["VO2R", [4.5546, 4.5546, 2.8514, 90.0, 90, 90.0], "rutile", ],  # SG 136 (87 deg Celsius)  Rutile
}

# --- -------------- Extinction Rules
dict_Extinc = {
    "NoExtinction": "no",
    "FaceCenteredCubic": "fcc",
    "Diamond": "dia",
    "BodyCentered": "bcc",
    "VO2_mono": "VO2_mono",
    "wurtzite": "wurtzite",
    "magnetite": "magnetite",
    "Al2O3": "Al2O3",
    "SG166": "SG166",
    "h+k=2n": "h+k=2n",
    "h+l=2n": "h+l=2n",
    "h+k+l=2n": "h+k+l=2n",
    "Ti2AlN": "Ti2AlN",
    "Al2O3_rhombo":"Al2O3_rhombo",
    "VO2_mono2":"VO2_mono2",
    "rutile": "rutile",
    "SG227": "SG227",
    "SG141": "SG141",
    "137": "137",
    "R3m_sg160": "R3m_sg160",
    "h+k=2n, modulated":"h+k=2n, modulated"
}

dict_Extinc_inv = {
    "no": "NoExtinction",
    "fcc": "FaceCenteredCubic",
    "dia": "Diamond",
    "bcc": "BodyCentered",
    "VO2_mono": "VO2_mono",
    "wurtzite": "wurtzite",
    "magnetite": "magnetite",
    "Al2O3": "Al2O3",
    "SG166": "SG166",
    "h+k=2n": "h+k=2n",
    "h+l=2n": "h+l=2n",
    "h+k+l=2n": "h+k+l=2n",
    "Ti2AlN": "Ti2AlN",
    "Al2O3_rhombo":"Al2O3_rhombo",
    "VO2_mono2":"VO2_mono2",
    "rutile":"rutile",
    "SG227": "SG227",
    "SG141": "SG141",
    "137": "137",
    "R3m_sg160":"R3m_sg160",
    "h+k=2n, modulated":"h+k=2n, modulated"
}

def writeDict(_dict, filename, writemode='a', sep=', '):
    """write an ascii file of dict_Materials

    :param dict: dict_Materials (see below)
    :type dict: dict
    :param filename: output file path
    :type filename: str
    :param writemode: 'a', to append existing file, 'w' to overwrite or create, defaults to 'a'
    :type writemode: str, optional
    :param sep: separator, defaults to ', '
    :type sep: str, optional
    """
    with open(filename, writemode) as f:
        for i in _dict.keys():
            f.write(i + ": " + sep.join([str(x) for x in _dict[i]]) + "\n")

def readDict(filename):
    """read ascii dict written by writeDict()

    :param filename: file path
    :type filename: str
    :raises ValueError: if an ascii line cannot be parsed
    :return: dict of Materials parameters
    :rtype: dict
    """
    with open(filename, "r") as f:
        _dict = {}
        k = 1
        for line in f:
            _key, val = readsinglelinedictfile(line)
            if _key in _dict:
                txt = 'In Materials file: %s line %d'%(filename, k)
                txt += '\n'+line
                txt += '\nkey_material : "%s" already exists!!!'%_key
                txt += '\nwith parameters: %s'%str(_dict[_key])
                txt += '\nPlease find an other name or choose one of them!'
                raise ValueError(txt)
            _dict[_key] = val
            k += 1
        return _dict

def readsinglelinedictfile(line):
    """parse a line of ascii file for dict Materials and return a single material key, value.
    Value is a list of: key_material (str), list of 6 lattice parameters, extinction rules label (str)

    :param line: line
    :type line: str
    :raises ValueError: if an ascii line cannot be parsed
    :return: key, value
    :rtype: tuple of 2 elements
    """
    listval = re.split("[ ()\[\)\;\:\,\]\n\t\a\b\f\r\v]", line)
    #             print "listval", listval
    liststring = []
    listfloat = []
    for elem in listval:
        if len(elem) == 0:
            continue
        try:
            val = float(elem)
            listfloat.append(val)
        except ValueError:
            liststring.append(elem)

    nbstrings = len(liststring)
    nbval = len(listfloat)

    if nbval != 6 or nbstrings != 3:
        txt = "Something wrong, I can't parse the line %s!!\n"%line
        txt += "Each line must look like :\n Cu: Cu, [3.6, 3.6, 3.6, 90, 90, 90], fcc"
        raise ValueError(txt)
    keydict = liststring[0]
    valdict = [liststring[1], listfloat, liststring[2]]
    return keydict, valdict

dict_Stiffness = {"Ge": ["Ge", [126, 44, 67.7], "cubic"]}


######## Geometrey Default  ##############
# Default constant
DEFAULT_DETECTOR_DISTANCE = 70.0  # mm
DEFAULT_DETECTOR_DIAMETER = 165.0  # mm
DEFAULT_TOP_GEOMETRY = "Z>0"

#############   2D DETECTOR ##############

LAUEIMAGING_DATA_FORMAT = "uint16"  # 'uint8'
LAUEIMAGING_FRAME_DIM = (1290, 1970)  # (645, 985)

CCD_CALIBRATION_PARAMETERS = ["dd", "xcen", "ycen", "xbet", "xgam", "pixelsize",
                                "xpixelsize", "ypixelsize", "CCDLabel",
                                "framedim", "detectordiameter", "kf_direction"]

# --- ---  CCD Read Image Parameters
# CCDlabel,
# framedim=(dim1, dim2),
# pixelsize,
# saturation value,
# geometrical operator key,
# header size in byte,
# binary encoding format,
# description,
# file extension
dict_CCD = {
    "MARCCD165": ((2048, 2048), 0.079142, 65535, "no", 4096, "uint16", "MAR Research 165 mm now rayonix", "mccd", ),
    "sCMOS": [(2018, 2016), 0.0734, 65535, "no", 3828, "uint16", "file as produced by sCMOS camera with checked fliplr transform. CCD parameters read from tif header by fabio", "tif"],

    "IMSTAR_bin2": [(3092, 3035), 0.0504, 65535, "no", 3828, "uint16", "", "tif"],
    "IMSTAR_bin1": [(6185, 6070), 0.0252, 65535, "no", 3828, "uint16", "", "tif"],
    "sCMOS_fliplr": [(2018, 2016), 0.0734, 65535, "sCMOS_fliplr", 3828, "uint16", "binned 2x2, CCD parameters read from tif header by fabio", "tif"],
    "sCMOS_fliplr_16M": [(2 * 2018, 2 * 2016), 0.0734 / 2.0, 65535, "sCMOS_fliplr", 3828, "uint16", "binned 1x1, CCD parameters binned 1x1 read from tif header by fabio ", "tif"],
    "sCMOS_16M": [(2 * 2018, 2 * 2016), 0.0734 / 2.0, 65535, "no", 3828, "uint16", "binned 1x1, CCD parameters binned 1x1 read from tif header by fabio ", "tif"],
    "sCMOS_9M": [(3072, 3072), 0.0504 , 65535, "no", 3828, "uint16", "binned 2x2, CCD parameters binned 2x2 from  36M sCMOS", "tif"],

    "psl_IN_bmp": [(425, 640), 10/43., 256, "no", -1, "uint16", "camera from inst Neel ", "bmp", ],
    "psl_IN_tif": [(425, 640), 10/43., 65535, "no", -1, "uint16", "camera from inst Neel ", "tif", ],
    "psl_weiwei": [(1247, 1960), 0.075000001, 65535, "no", -1, "uint16", "camera from desy photonics science 1247*1960 ", "tif"],
    "Alexiane": [(1078, 886), 0.139, 65535, "Alexiane", -1, "uint16", "camera from Alexiane's lab reflection geometry", "tif"],

    "VHR_full": ((2671, 4008), 0.031, 10000, "vhr", 4096, "uint16", "NOT USED: very basic vhr settings, the largest frame available without grid correction", "tiff"),
    "VHR_diamond": ((2594, 3764), 0.031, 10000, "vhr", 4096, "uint16", "first vhr settings of Jun 12 close to diamond 2theta axis displayed is vertical, still problem with fit from PeakSearchGUI", "tiff"),
    "VHR_small": ((2594, 2748), 0.031, 10000, "vhr", 4096, "uint16", "vhr close to diamond Nov12 frame size is lower than VHR_diamond", "tiff"),
    "ImageStar": ((1500, 1500), 0.044, 65535, "vhr", 4096, "uint16", "Imagestar photonics Science close to diamond March13  extension is mar.tiff", "tiff"),
    "ImageStar_raw_3056x3056": ((3056, 3056), 0.022, 64000, "vhr", 872, "uint16", "raw image Apr 2018 imagestar for diamond binning 1x1  .tif (could be read by VHR_DLS?)", "tif"),
    "ImageStar_1528x1528": ((1528, 1528), 0.044, 65535, "vhr", 4096, "uint16", "OLD: Imagestar photonics Science close to diamond May 2018  extension is mar.tiff  non remapping 1528x1528", "tiff"),
    "ImageDeathStar": ((1500, 1500), 0.044, 65535, "VHR_Feb13", 4096, "uint16", "OLD: Imagestar photonics Science close to sample, mounting similar to MARCCD, Sep14", "tif"),
    "ImageStar_raw": ((1500, 1500), 0.044, 64000, "vhr", 872, "uint16", "OLD: raw image GISAXS BM32 November 2014 .tif", "tif"),
    "ImageStar_dia_2021": ((3056, 3056), 0.022, 65535, "ImageStar_dia_2021", 4096, "uint16", "VALID 2024: Imagestar photonics Science close to diamond for rainbow technique", "tif"),
    "ImageStar_dia_2021_2x2": ((1528, 1528), 0.044, 64000, "ImageStar_dia_2021", 4096, "uint16", "VALID 2024: Imagestar photonics Science close to diamond since Feb21", "tif"),
    "VHR_diamond_Mar13": ((2594, 2774), 0.031, 10000, "vhr", 4096, "uint16", "vhr close to diamond Mar13 frame size is lower than VHR_diamond", "tiff"),
    "VHR": ((2594, 3764), 0.031, 10000, "VHR_Feb13", 4096, "uint16", "OLD: vhr settings of Jun 12 2theta axis displayed is horizontal, no problem with fit from PeakSearchGUI", "tiff"),
    "VHR_Feb13": ((2594, 2774), 0.031, 10000, "VHR_Feb13", 4096, "uint16", "OLD: vhr settings of Feb13 close to sample 2theta axis displayed is vertical, no problem with fit from PeakSearchGUI", "tiff"),
    "VHR_Feb13_rawtiff": ((2594, 2774), 0.031, 10000, "OLD: VHR_Feb13", 110, "uint16", " ", "tiff"),
    "VHR_PSI": ((2615, 3891), 0.0312, 65000, "no", 4096, "uint16", "OLD: vhr at psi actually read by libtiff (variable header size and compressed data)", "tif"),
    "VHR_DLS": ((3056, 3056), 0.0312, 65000, "no", 4096, "uint16", "OLD: vhr at dls actually read by libtiff (variable header size and compressed data)", "tif"),

    "Alban": [(1504, 1499), 0.2, 65535*65535, "no", -1, "uint32", "Alban de vaucorbeil detector australia", "tif"],
    "PRINCETON": ((2048, 2048), 0.079, 57000, "no", 4096, "uint16", "OLD: ROPER Princeton Quadro 2048x2048 pixels converted from .spe to .mccd", "mccd"),  # 2X2, saturation value depends on gain and DAC
    "FRELON": ((2048, 2048), 0.048, 65000, "frelon2", 1024, "uint16", "FRELON camera 2048x2048 pixels, 2theta axis is horizontal (edf format)", "edf"),
    "TIFF Format": (-1, -1, "", "", "", "" "CCD parameters read from tiff header", "tiff", ),
    "EDF": ((1065, 1030), 0.075, 650000000000, "no", 0, "uint32", "CCD parameters read from edf header EIGER", "edf", ),
    "pnCCD": ((384, 384), 0.075, 65000, "no", 1024, "uint16", "pnCCD from SIEGEN only: pixel size and frame dimensions OK", "tiff"),
    "pnCCD_Tuba": ((384, 384), 0.075, 10000000, "no", 258, "uint16", "pnCCD from Tuba only: pixel size and frame dimensions OK", "tiff"),
    
    "EIGER_4Mstack": ((2167, 2070), 0.075, 4294967295, "no", 0, "uint32", "detector parameters read hdf5 EIGER4M stack used at SLS", "h5"),
    "EIGER_4Munstacked": ((2167, 2070), 0.075, 4294967295, "no", 0, "uint32", "unstacked hdf5 EIGER4M  used at SLS", "unstacked"),
    "EIGER_4MCdTe": ((2162, 2068), 0.075000002, 4294967295, "no", 0, "uint32", "single image in hdf5 EIGER4M of CdTe  used at BM32", "h5"),
    "EIGER_4MCdTestack": ((2162, 2068), 0.075000003, 4294967295, "no", 0, "uint32", "stacked images in hdf5 EIGER4M of CdTe used at BM32", "h5"),
    "EIGER_1M": ((1065, 1030), 0.075000004, 4294967295, "no", 0, "uint32", "CCD parameters read from edf header EIGER1M Silicon at BM32 ESRF", "edf"),
    "EIGER_4M": ((2167, 2070), 0.075000005, 4294967295, "spe", 0, "uint32", "detector parameters read from tif header EIGER4M used at ALS or SLS   (fliprot = spe for SLS)", "tif"),
    "RXO": ((512, 512), 0.4, 255, "no", 0 , "uint8", "RXO industrial detector", "tif"),
    "MaxiPIXCdTe": ((516, 516), 0.055, 4294967295, "MaxiPIXCdTe", 0, "int32", "MaxiPIX Laue CdTe BM32 test", "h5"),
}


def getwildcardstring(CCDlabel):
    r"""
    return smart wildcard to open binary CCD image file with priority of CCD type of CCDlabel

    Parameters
    ----------
    CCDlabel : string
        label defining the CCD type

    Returns
    -------
    wildcard_extensions : string
        string from concatenated strings to be used in wxpython open file dialog box

    See Also
    ----------

    :func:`getIndex_fromfilename`

    LaueToolsGUI.AskUserfilename

    wx.FileDialog

    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.

    >>> from dict_LaueTools import getwildcardstring
    >>> getwildcardstring('MARCCD165')
    'MARCCD, ROPER(*.mccd)|*mccd|mar tif(*.tif)|*_mar.tiff|tiff(*.tiff)|*tiff|Princeton(*.spe)|*spe|Frelon(*.edf)|*edf|tif(*.tif)|*tif|All files(*)|*'
    """
    ALL_EXTENSIONS = ["mccd", "tif", "_mar.tiff", "tiff", "spe", "edf", "tif", "tif.gz", "h5", "", ]
    INFO_EXTENSIONS = ["MARCCD, ROPER(*.mccd)", "sCMOS sCMOS_fliplr", "mar tif(*.tif)", "tiff(*.tiff)",
                        "Princeton(*.spe)", "Frelon(*.edf)", "tif(*.tif)", "tif.gz(*.tif.gz)",
                        "hdf5(*.h5)", "All files(*)", ]

    extensions = copy.copy(ALL_EXTENSIONS)
    infos = copy.copy(INFO_EXTENSIONS)
    if CCDlabel != '*':
        chosen_extension = dict_CCD[CCDlabel][7]
    else:
        chosen_extension = ''

    if chosen_extension in ALL_EXTENSIONS:
        index = ALL_EXTENSIONS.index(chosen_extension)
        ce = extensions.pop(index)
        extensions.insert(0, ce)

        inf = infos.pop(index)
        infos.insert(0, inf)

    wcd = ""
    for inf, ext in zip(infos, extensions):
        wcd += "%s|*%s|" % (inf, ext)

    wildcard_extensions = wcd[:-1]

    return wildcard_extensions


# ------------------------------------------------------
# CCD pixels skewness:
# #RECTPIX = 0.0 : square pixels
# #RECTPIX = -1.0e-4
# define rectangular pixels with
# xpixsize = pixelsize
# ypixsize = xpixsize*(1.0+RECTPIX)
# ---------------------------------------------
RECTPIX = 0  # CCD pixel skewness   see find2thetachi

list_CCD_file_extensions = []
for key in list(dict_CCD.keys()):
    list_CCD_file_extensions.append(dict_CCD[key][-1])
list_CCD_file_extensions.append("tif.gz")
# print list_CCD_file_extensions

# ---   ---  general geometry of detector CCD position
DICT_LAUE_GEOMETRIES = {"Z>0": "Top Reflection (2theta=90)",
                        "X>0": "Transmission (2theta=0)",
                        "X<0": "Back Reflection (2theta=180)"}

DICT_LAUE_GEOMETRIES_INFO = {
    "Top Reflection (2theta=90)": ["Z>0", "top reflection geometry camera on top (2theta=90)"],
    "Transmission (2theta=0)": ["X>0", "Transmission geometry, camera in direct beam (2theta=0)"],
    "Back Reflection (2theta=180)": ["X<0", "Back reflection geometry, camera is upstream (2theta=180)"]
}


# --- -------------- History of Calibration Parameters
dict_calib = {
    "ZrO2 Sep08": [69.8076, 878.438, 1034.46, 0.54925, 0.18722],  # as first trial of zrO2 sicardy sep 08
    "Sep09": [68.0195, 934.94, 1033.6, 0.73674, -0.74386],  # Sep09
    "ZrO2 Dec09": [69.66221, 895.29492, 960.78674, 0.84324, -0.32201],  # Dec09 Julie Ge_run41_1_0003.mccd
    "Basic": [68, 930, 1100, 0.0, 0.0],
}

# --- -------------- Transforms 3x3 Matrix
dict_Vect = {
    "Identity": [[1.0, 0, 0], [0, 1, 0], [0, 0, 1]],
    "Default": [[1.0, 0, 0], [0, 1, 0], [0, 0, 1]],
    "sigma3_1": [
        [1.0 / 3, -2.0 / 3, -2.0 / 3],
        [-2.0 / 3, 1.0 / 3, -2.0 / 3],
        [-2.0 / 3, -2.0 / 3, 1.0 / 3],
    ],
    "sigma3_2": [
        [1.0 / 3, -2.0 / 3, -2.0 / 3],
        [2.0 / 3, -1.0 / 3, 2.0 / 3],
        [2.0 / 3, 2.0 / 3, -1.0 / 3],
    ],
    "sigma3_3": [
        [-1.0 / 3, 2.0 / 3, 2.0 / 3],
        [-2.0 / 3, 1.0 / 3, -2.0 / 3],
        [2.0 / 3, 2.0 / 3, -1.0 / 3],
    ],
    "sigma3_4": [
        [-1.0 / 3, 2.0 / 3, 2.0 / 3],
        [2.0 / 3, -1.0 / 3, 2.0 / 3],
        [-2.0 / 3, -2.0 / 3, 1.0 / 3],
    ],
    # "sigma3_1": [
    #     [-1.0 / 3, 2.0 / 3, 2.0 / 3],
    #     [2.0 / 3, -1.0 / 3, 2.0 / 3],
    #     [2.0 / 3, 2.0 / 3, -1.0 / 3],
    # ],
    # "sigma3_2": [
    #     [-1.0 / 3, -2.0 / 3, 2.0 / 3],
    #     [-2.0 / 3, -1.0 / 3, -2.0 / 3],
    #     [2.0 / 3, -2.0 / 3, -1.0 / 3],
    # ],
    # "sigma3_3": [
    #     [-1.0 / 3, 2.0 / 3, -2.0 / 3],
    #     [2.0 / 3, -1.0 / 3, -2.0 / 3],
    #     [-2.0 / 3, -2.0 / 3, -1.0 / 3],
    # ],
    # "sigma3_4": [
    #     [-1.0 / 3, -2.0 / 3, -2.0 / 3],
    #     [-2.0 / 3, -1.0 / 3, 2.0 / 3],
    #     [-2.0 / 3, 2.0 / 3, -1.0 / 3],
    # ],
    "shear1": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.008, 0.0, 1.0]],
    "shear2": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.016, 0.0, 1.0]],
    "shear3": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.024, 0.0, 1.0]],
    "shear4": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.032, 0.0, 1.0]],
    "JSMtest": [[1.0, 1.02, 0.98], [-0.1, 1.1, 0.2], [-0.032, 0.05, 1.15]],
    "JSMtest2": [[1.0, 1.02, 0.98], [-0.1, 1.1, 0.2], [-0.032, 0.05, 1.15]],
    "543_909075": [[0.20705523608201659, -0.066987298107780757, 2.0410779985789219e-17],
        [0.0, 0.24999999999999994, -2.0410779985789219e-17],
        [0.0, 0.0, 0.33333333333333331]]}

dict_Transforms = {
    "Identity": [[1.0, 0, 0], [0, 1, 0], [0, 0, 1]],
    "Default": [[1.0, 0, 0], [0, 1, 0], [0, 0, 1]],
    "sigma3_1": [
        [1.0 / 3, -2.0 / 3, -2.0 / 3],
        [-2.0 / 3, 1.0 / 3, -2.0 / 3],
        [-2.0 / 3, -2.0 / 3, 1.0 / 3],
    ],
    "sigma3_2": [
        [1.0 / 3, -2.0 / 3, -2.0 / 3],
        [2.0 / 3, -1.0 / 3, 2.0 / 3],
        [2.0 / 3, 2.0 / 3, -1.0 / 3],
    ],
    "sigma3_3": [
        [-1.0 / 3, 2.0 / 3, 2.0 / 3],
        [-2.0 / 3, 1.0 / 3, -2.0 / 3],
        [2.0 / 3, 2.0 / 3, -1.0 / 3],
    ],
    "sigma3_4": [
        [-1.0 / 3, 2.0 / 3, 2.0 / 3],
        [2.0 / 3, -1.0 / 3, 2.0 / 3],
        [-2.0 / 3, -2.0 / 3, 1.0 / 3],
    ],
    "shear1": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.008, 0.0, 1.0]],
    "shear2": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.016, 0.0, 1.0]],
    "shear3": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.024, 0.0, 1.0]],
    "shear4": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.032, 0.0, 1.0]],
    "stretch_axe1_0p01": [[1.01, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "stretch_axe2_0p01": [[1.0, 0.0, 0.0], [0.0, 1.01, 0.0], [0.0, 0.0, 1.0]],
    "stretch_axe3_0p01": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.01]],
    "JSMtest": [[1.0, 1.02, 0.98], [-0.1, 1.1, 0.2], [-0.032, 0.05, 1.15]],
    "JSMtest2": [[1.0, 1.02, 0.98], [-0.1, 1.1, 0.2], [-0.032, 0.05, 1.15]],
    "twin100": [[1, 0, 0], [0, -1, 0], [0, 0, -1.0]],
    "twin010": [[-1, 0, 0], [0, 1, 0], [0, 0, -1.0]],
    "twin001": [[-1, 0, 0], [0, -1, 0], [0, 0, 1.0]],
}

sq3 = np.sqrt(3)
sq2 = np.sqrt(2)
sq6 = np.sqrt(6)
# --- --------------  (almost) Rotation Matrices

dict_Rot = {
    "GesCMOS_sept2024": [[0.6497410927187327, 0.5764646261387358, 0.49550484078704454],
[-0.0119670971887755, 0.6595228823484899, -0.7515892204147262],
[-0.7600613797697796, 0.482408646760677, 0.4354177264570093]],
    "mat203": [[ 0.280833091148746,  0.               , -0.959756622751748],
       [ 0.               ,  1.               ,  0.               ],
       [ 0.959756622751748,  0.               ,  0.280833091148746]],
    "Si_001": [
        [-0.55064559927923185, 0.54129537384117710, -0.63548417889810638],
        [-0.70646885891790212, -0.70765027148876392, 0.00938623461321308],
        [-0.44458921907077736, 0.45412248299046004, 0.77207936061212878],
    ],
    "Ge_23Feb09": [
        [-0.48273739999999998, 0.22555140000000001, -0.84622169999999997],
        [-0.80558430000000003, 0.2646423, 0.53009289999999998],
        [0.34350920000000001, 0.9375985, 0.0539479],
    ],
    "matTEST": [
        [-0.069614099999999998, -0.097000299999999998, -0.99284680000000003],
        [-0.71808340000000004, 0.69573390000000002, -0.017623799999999998],
        [0.69246669999999999, 0.71171989999999996, -0.1180872],
    ],
    "OrientSurf001": [
        [0.76604444311897801, 0.0, -0.64278760968653925],
        [0.0, 1.0, 0.0],
        [0.64278760968653925, 0.0, 0.76604444311897801],
    ],
    "OrientSurf101": [
        [0.087155742747658138, 0.0, -0.99619469809174555],
        [0.0, 1.0, 0.0],
        [0.99619469809174555, 0.0, 0.087155742747658138],
    ],
    "matsolCu": [
        [-0.87097440000000004, -0.0123476, -0.49117309999999997],
        [-0.24467739999999999, 0.87780820000000004, 0.4118078],
        [0.42607099999999998, 0.47885309999999998, -0.76756970000000002],
    ],
    "mat112": [[ 0.363055725859447, -0.69392925376254 , -0.621814064406101],
       [ 0.408248290463863,  0.718350341907227, -0.563299316185545],
       [ 0.83757021989443 , -0.049345486666304,  0.544096636353931]],
    "mat113": [[ 0.499105513678866, -0.492912416425547, -0.712692806157662],
       [ 0.301511344577764,  0.869848865542223, -0.390453403373329],
       [ 0.812394359474673, -0.020007519797189,  0.582765050294972]],
    "Identity": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "mat111": [[ 0.071162365961679, -0.975273803766748, -0.20922936064777 ],
       [ 0.577350269189626,  0.211324865405187, -0.788675134594813],
       [ 0.813389564930239, -0.06467463913945 ,  0.578112970547089]],
    "mat313": [[ 0.084831170689394, -0.535447307420881, -0.840297479143664],
       [ 0.229415733870562,  0.831175279838831, -0.506474160483506],
       [ 0.969624717884243, -0.149812666889583,  0.19334960901779 ]],
    "mat213": [[ 0.270617537982782, -0.52444594132329 , -0.807293380850893],
       [ 0.267261241912424,  0.846547751617515, -0.460356745147455],
       [ 0.924844623008409, -0.091177622588318,  0.369254741921585]],
    "mat212": [[ 0.082171222288292, -0.724958831974832, -0.683873220830686],
       [ 0.333333333333333,  0.666666666666667, -0.666666666666667],
       [ 0.939221368537012, -0.173176925418034,  0.296433758850472]],
    "mat001c1": [
        [0.80585162458034332, -0.099044669238326555, -0.58377514453272938],
        [-0.050447334156050835, 0.97084166554927731, -0.23435350971946972],
        [0.58996466978353546, 0.21830403690176159, 0.77735774278669989],
    ],
    "OrientSurf111": [
        [-0.68482849999999995, 0.25444870000000003, -0.68283649999999996],
        [-0.7064087, -0.0017916, 0.70780189999999998],
        [0.1788759, 0.96708459999999996, 0.18097170000000001],
    ],
    "mat001": [[ 0.766044443118978,  0.               , -0.642787609686539],
       [ 0.               ,  1.               ,  0.               ],
       [ 0.642787609686539,  0.               ,  0.766044443118978]
       ],
    "mat101": [[ 0.087155742747658,  0.               , -0.996194698091746],
       [ 0.               ,  1.               ,  0.               ],
       [ 0.996194698091746,  0.               ,  0.087155742747658]],
    "mat103":[[ 0.523466278916904,  0.               , -0.85204639242056 ],
       [ 0.               ,  1.               ,  0.               ],
       [ 0.85204639242056 ,  0.               ,  0.523466278916904]],
    "mat102": [[ 0.397707621369261,  0.               , -0.917512205861483],
       [ 0.               ,  1.               ,  0.               ],
       [ 0.917512205861483,  0.               ,  0.397707621369261]],
    "mat111alongx": [
        [1 / sq3, 1 / sq2, -1 / sq6],
        [1 / sq3, -1 / sq2, -1 / sq6],
        [1 / sq3, 0, 2.0 / sq6]]
        }

# dictionary of some rotations from a sequence of three elementary rotations
dict_Eul = {
    "Identity": [0.0, 0.0, 0.0],
    "misorient_0": [21.0, 1.0, 53.0],
    "misorient_1": [21.2, 1.0, 53.0],
    "misorient_2": [21.4, 1.0, 53.0],
    "misorient_3": [21.6, 1.0, 53.0],
    "misorient_4": [21.6, 1.2, 53.0],
    "misorient_5": [21.0, 1.4, 53.0],
    "misorient_6": [21.0, 1.6, 53.0],
    "EULER_1": [10.0, 52.0, 45.0],
    "EULER_2": [14.0, 2.0, 56.0],
    "EULER_3": [38.0, 85.0, 1.0],
    "EULER_4": [1.0, 1.0, 53.0],
}

# --- ---------- Example to add a new material
# Umat = dict_Rot['mat311c1']
# Dc = dict_Vect['shear4']
# Bmat = dict_Vect['543_909075']
# Id = np.eye(3)
#
# dict_Materials['mycell_s'] = ['mycell_s', [Id, Umat, Dc, Bmat], 'fcc']
# dict_Materials['mycell'] = ['mycell', [Id, Umat, Id, Bmat], 'fcc']
SAMPLETILT = 40.0

DEG = np.pi / 180.0
PI = np.pi
RotY40 = np.array([[np.cos(SAMPLETILT * DEG), 0, -np.sin(SAMPLETILT * DEG)],
                    [0, 1, 0],
                    [np.sin(SAMPLETILT * DEG), 0, np.cos(SAMPLETILT * DEG)]])
RotYm40 = np.array([[np.cos(SAMPLETILT * DEG), 0, np.sin(SAMPLETILT * DEG)],
                        [0, 1, 0],
                        [-np.sin(SAMPLETILT * DEG), 0, np.cos(SAMPLETILT * DEG)]])

# planck constant h * 2pi in 1e-16 eV.sec  unit
hbarrex1em16 = 6.58211899
# light speed : c in 1e7 m/s units
ccx1e7 = 29.9792458

E_eV_fois_lambda_nm = np.pi * 2.0 * hbarrex1em16 * ccx1e7
# print "E_eV_fois_lambda_nm = ", E_eV_fois_lambda_nm
CST_ENERGYKEV = 12.398  # keV * angstrom  in conversion formula:E (keV) = 12.398 / lambda (angstrom)

# --- ----------- cubic permutation operators
opsymlist = np.zeros((48, 9), float)

opsymlist[0, :] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])  # identity
opsymlist[1, :] = np.array([-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0])

opsymlist[2, :] = np.array([1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0])
opsymlist[3, :] = np.array([-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

opsymlist[4, :] = np.array([-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
opsymlist[5, :] = np.array([1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0])

opsymlist[6, :] = np.array([-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0])
opsymlist[7, :] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0])

opsymlist[8, :] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
opsymlist[9, :] = np.array([0.0, -1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0])

opsymlist[10, :] = np.array([0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0])
opsymlist[11, :] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0])

opsymlist[12, :] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0])
opsymlist[13, :] = np.array([0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
opsymlist[14, :] = np.array([0.0, -1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0])
opsymlist[15, :] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0])

opsymlist[16, :] = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
opsymlist[17, :] = np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0])
opsymlist[18, :] = np.array([0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0])
opsymlist[19, :] = np.array([0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
opsymlist[20, :] = np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
opsymlist[21, :] = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0])
opsymlist[22, :] = np.array([0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0])
opsymlist[23, :] = np.array([0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

opsymlist[24, :] = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
opsymlist[25, :] = np.array([0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0])
opsymlist[26, :] = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0])
opsymlist[27, :] = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
opsymlist[28, :] = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0])
opsymlist[29, :] = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
opsymlist[30, :] = np.array([0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
opsymlist[31, :] = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0])

opsymlist[32, :] = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
opsymlist[33, :] = np.array([0.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0])
opsymlist[34, :] = np.array([0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
opsymlist[35, :] = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0])
opsymlist[36, :] = np.array([0.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0])
opsymlist[37, :] = np.array([0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
opsymlist[38, :] = np.array([0.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0])
opsymlist[39, :] = np.array([0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])

opsymlist[40, :] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
opsymlist[41, :] = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0])
opsymlist[42, :] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0])
opsymlist[43, :] = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
opsymlist[44, :] = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
opsymlist[45, :] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0])
opsymlist[46, :] = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0])
opsymlist[47, :] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0])

OpSymArray = np.reshape(opsymlist, (48, 3, 3))

# FCC slips systems  plane normal (p) and slip direction (burger b)
# rotation axis is given by cross product of p ^ b
SLIPSYSTEMS_FCC = np.array([[1, 1, 1], [0, -1, 1],
    [1, 1, 1], [-1, 0, 1],
    [1, 1, 1], [-1, 1, 0],
    [-1, 1, 1], [0, -1, 1],
    [-1, 1, 1], [1, 0, 1],
    [-1, 1, 1], [1, 1, 0],
    [1, -1, 1], [0, 1, 1],
    [1, -1, 1], [-1, 0, 1],
    [1, -1, 1], [1, 1, 0],
    [1, 1, -1], [0, 1, 1],
    [1, 1, -1], [1, 0, 1],
    [1, 1, -1], [-1, 1, 0]]).reshape((12, 2, 3))
