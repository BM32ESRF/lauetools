#!/usr/bin/env python
# -*- coding:  utf-8 -*-
"""

"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

# predefined material             elements              weight percentage      density
dict_mat = {'316L'  : (('Fe', 'Cr','Ni','Mo'), [0.685, 0.17, 0.12, 0.025], 7.950),
            'Ni5Ti' : (('Ni', 'Ti')          , [0.95, 0.05],               8.489), # approx density
            'YSZ-8' : (('Zr', 'Y', 'O')      , [0.639, 0.108, 0.253],      6.10) # note sure ?
            }
    
# typical densities of elements g/cm3
dict_density = {'H' : 0.090,
                'He': 0.178,
                'Li': 0.53,
                'Be': 1.85,
                'B' : 2.34,
                'C' : 2.26,
                'N' : 1.251E-2,
                'O' : 1.429E-2,
                'F' : 1.696E-2,
                'Ne': 0.900,
                'Na': 0.970,
                'Mg': 1.74,
                'Al': 2.70,
                'Si': 2.33,
                'P' : 1.82,
                'S' : 2.07,
                'Cl': 3.214,
                'Ar': 1.784,
                'K' : 0.86,
                'Ca': 1.55,
                'Sc': 2.99,
                'Ti': 4.54,
                'V' : 6.11,
                'Cr': 7.19,
                'Mn': 7.44,
                'Fe': 7.874,
                'Co': 8.89,
                'Ni': 8.91,
                'Cu': 8.96,
                'Zn': 7.13,
                'Ga': 5.91,
                'Ge': 5.323,
                'As': 5.78,
                'Se': 4.79,
                'Br': 3.12,
                'Kr': 3.75,
                'Rb': 1.53,
                'Sr': 2.54,
                'Y' : 4.47,
                'Zr': 6.51,
                'Nb': 8.57,
                'Mo': 10.22,
                'Tc': 11.49,
                'Ru': 12.37,
                'Rh': 12.41,
                'Pd': 12.02,
                'Ag': 10.49,
                'Cd': 8.65,
                'In': 7.31,
                'Sn': 7.31,
                'Sb': 6.69,
                'Te': 6.24,
                'I' : 4.93,
                'Xe': 5.9,
                'Cs': 1.87,
                'Ba': 3.59,
                'La': 6.15,
                'Ce': 6.77,
                'Pr': 6.77,
                'Nd': 7.01,
                'Pm': 7.22,
                'Sm': 7.52,
                'Eu': 5.24,
                'Gd': 7.9,
                'Tb': 8.23,
                'Dy': 8.55,
                'Ho': 8.8,
                'Er': 9.07,
                'Tm': 9.32,
                'Yb': 6.97,
                'Lu': 9.84,
                'Hf': 13.31,
                'Ta': 16.65,
                'W' : 19.25,
                'Re': 21.03,
                'Os': 22.61,
                'Ir': 22.65,
                'Pt': 21.45,
                'Au': 19.32,
                'Hg': 13.55,
                'Tl': 11.85,
                'Pb': 11.35,
                'Bi': 9.75,
                'Po': 9.20,
                'At': None,
                'Rn': 9.73,
                'Fr': None,
                'Ra': 5.50,
                'Ac': 10.07,
                'Tb': 11.72,
                'Pa': 15.4,
                'U' : 18.95,
                'Np': 20.2,
                'Pu': 19.84,
                'Am': 13.7,
                'Cm': 13.5,
                'Bk': None,
                'Cf': None,
                'Es': None,
                'Fm': None,
                'Md': None,
                'No': None,
                'Lr': None,
                'air':1.225E-3}   

# fluorescence yields
dict_fluo_yield = { 'B' : 1.7E-3,
                    'C' : 2.8E-3,
                    'N' : 5.2E-3,
                    'O' : 8.3E-3,
                    'F' : 0.013,
                    'Ne': 0.016,
                    'Na': 0.023,
                    'Mg': 0.030,
                    'Al': 0.039,
                    'Si': 0.050,
                    'P' : 0.063,
                    'S' : 0.078,
                    'Cl': 0.097,
                    'Ar': 0.118,
                    'K' : 0.140,
                    'Ca': 0.163,
                    'Sc': 0.188,
                    'Ti': 0.214,
                    'V' : 0.243,
                    'Cr': 0.275,
                    'Mn': 0.308,
                    'Fe': 0.340,
                    'Co': 0.373,
                    'Ni': 0.406,
                    'Cu': 0.440,
                    'Zn': 0.474,
                    'Ga': 0.507,
                    'Ge': 0.535,
                    'As': 0.562,
                    'Se': 0.589,
                    'Br': 0.618,
                    'Kr': 0.643,
                    'Rb': 0.667,
                    'Sr': 0.690,
                    'Y' : 0.710,
                    'Zr': 0.730,
                    'Nb': 0.747,
                    'Mo': 0.765,
                    'Tc': 0.780,
                    'Ru': 0.794,
                    'Rh': 0.808,
                    'Pd': 0.820,
                    'Ag': 0.831,
                    'Cd': 0.843,
                    'In': 0.853,
                    'Sn': 0.862,
                    'Sr': 0.870,
                    'Te': 0.877,
                    'I' : 0.884,
                    'Xe': 0.891,
                    'Cs': 0.897,
                    'Ba': 0.902,
                    'La': 0.907,
                    'Ce': 0.912,
                    'Pr': 0.917,
                    'Nd': 0.917,
                    'Pm': 0.925,
                    'Sm': 0.929,
                    'Eu': 0.932,
                    'Gd': 0.935,
                    'Tb': 0.938,
                    'Dy': 0.941,
                    'Ho': 0.944,
                    'Er': 0.947,
                    'Tm': 0.949,
                    'Yb': 0.951,
                    'Lu': 0.953,
                    'Hf': 0.955,
                    'Ta': 0.957,
                    'W' : 0.958,
                    'Re': 0.959,
                    'Os': 0.961,
                    'Ir': 0.962,
                    'Pt': 0.963,
                    'Au': 0.964,
                    'Hg': 0.965,
                    'Tl': 0.966,
                    'Pb': 0.967,
                    'Bi': 0.968,
                    'Po': 0.968,
                    'At': 0.969,
                    'Rn': 0.969,
                    'Fr': 0.970,
                    'Ra': 0.970,
                    'Ac': 0.971,
                    'Tb': 0.971,
                    'Pa': 0.972,
                    'U' : 0.972,
                    'Np': 0.973,
                    'Pu': 0.973,
                    'Am': 0.974,
                    'Cm': 0.974,
                    'Bk': 0.975,
                    'Cf': 0.975,
                    'Es': 0.975,
                    'Fm': 0.976,
                    'Md': 0.976,
                    'No': 0.976,
                    'Lr': 0.977}

# K absorption edges (within 5 - 25 keV), corresponding fluorescence energies (Ka1)
dict_fluo_K ={  'B' : (False, 0, 0),
                'C' : (False, 0, 0),
                'N' : (False, 0, 0),
                'O' : (False, 0, 0),
                'F' : (False, 0, 0),
                'Ne': (False, 0, 0),
                'Na': (False, 1.0721, 1.04098),
                'Mg': (False, 1.3050, 1.25360),
                'Al': (False, 1.5596, 1.48670),
                'Si': (False, 1.8389, 1.73998),
                'P' : (False, 2.1455, 2.01370),
                'S' : (False, 2.4720, 2.30784),
                'Cl': (False, 2.8224, 2.62239),
                'Ar': (False, 3.2029, 2.95770),
                'K' : (False, 3.6074, 3.31380),
                'Ca': (False, 4.0381, 3.69168),
                'Sc': (True, 4.4928, 4.09060),
                'Ti': (True, 4.9664, 4.51084),
                'V' : (True, 5.4651, 4.95220),
                'Cr': (True, 5.9892, 5.41472),
                'Mn': (True, 6.5390, 5.89875),
                'Fe': (True, 7.1120, 6.40384),
                'Co': (True, 7.7089, 6.93032),
                'Ni': (True, 8.3328, 7.47815),
                'Cu': (True, 8.9789, 8.04778),
                'Zn': (True, 9.6586, 8.63886),
                'Ga': (True, 10.3671, 9.25174),
                'Ge': (True, 11.1031, 9.88642),
                'As': (True, 11.8667, 10.54372),
                'Se': (True, 12.6578, 11.22240),
                'Br': (True, 13.4737, 11.92420),
                'Kr': (True, 14.3256, 12.649),
                'Rb': (True, 15.1997, 13.3953),
                'Sr': (True, 16.1046, 14.165),
                'Y' : (True, 17.0384, 14.9584),
                'Zr': (True, 17.9976, 15.7751),
                'Nb': (True, 18.9856, 16.6151),
                'Mo': (True, 19.9995, 17.47934),
                'Tc': (True, 21.0440, 18.3671),
                'Ru': (True, 22.1172, 19.2792),
                'Rh': (True, 23.2199, 20.2161),
                'Pd': (True, 24.3503, 21.1771),
                'Ag': (False, 25.5140, 22.16292),
                'Cd': (False, 26.7112, 23.1736),
                'In': (False, 27.9399, 24.2097),
                'Sn': (False, 29.2001, 25.2713),
                'Sr': (False, 30.4912, 26.3591),
                'Te': (False, 31.8138, 27.4723),
                'I' : (False, 33.1694, 28.6120),
                'Xe': (False, 34.5614, 29.779),
                'Cs': (False, 35.9846, 30.9728),
                'Ba': (False, 37.4406, 32.1936),
                'La': (False, 38.9246, 33.4418),
                'Ce': (False, 40.4430, 34.7197),
                'Pr': (False, 0, 0),
                'Nd': (False, 0, 0),
                'Pm': (False, 0, 0),
                'Sm': (False, 0, 0),
                'Eu': (False, 0, 0),
                'Gd': (False, 0, 0),
                'Tb': (False, 0, 0),
                'Dy': (False, 0, 0),
                'Ho': (False, 0, 0),
                'Er': (False, 0, 0),
                'Tm': (False, 0, 0),
                'Yb': (False, 0, 0),
                'Lu': (False, 0, 0),
                'Hf': (False, 0, 0),
                'Ta': (False, 0, 0),
                'W' : (False, 0, 0),
                'Re': (False, 0, 0),
                'Os': (False, 0, 0),
                'Ir': (False, 0, 0),
                'Pt': (False, 0, 0),
                'Au': (False, 0, 0),
                'Hg': (False, 0, 0),
                'Tl': (False, 0, 0),
                'Pb': (False, 0, 0),
                'Bi': (False, 0, 0),
                'Po': (False, 0, 0),
                'At': (False, 0, 0),
                'Rn': (False, 0, 0),
                'Fr': (False, 0, 0),
                'Ra': (False, 0, 0),
                'Ac': (False, 0, 0),
                'Tb': (False, 0, 0),
                'Pa': (False, 0, 0),
                'U' : (False, 0, 0),
                'Np': (False, 0, 0),
                'Pu': (False, 0, 0),
                'Am': (False, 0, 0),
                'Cm': (False, 0, 0),
                'Bk': (False, 0, 0),
                'Cf': (False, 0, 0),
                'Es': (False, 0, 0),
                'Fm': (False, 0, 0),
                'Md': (False, 0, 0),
                'No': (False, 0, 0),
                'Lr': (False, 0, 0),
                }