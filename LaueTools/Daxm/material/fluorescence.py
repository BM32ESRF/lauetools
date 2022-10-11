#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import numpy as np

import LaueTools.Daxm.material.dict_datamat as dm
import LaueTools.Daxm.material.absorption as abso


def calc_fluorescence(element, energy, intensity, vmass=-1, abs_coeff=None):
    
    if not hasattr(vmass, "__len__") and vmass == -1:
            
        vmass = dm.dict_density[element]
        
    else:
        vmass = np.array(vmass)[:, np.newaxis]
              
    if abs_coeff is None:
        
        abs_coeff, _, _ = abso.calc_absorption(element, energy)

    fluo_yield, energy_edge, energy_fluo = get_fluorescence_data(element)
    
    mask = np.interp(energy, [energy_edge, energy.max()], [1., 1.], left=0)

    abs_coeff = np.atleast_2d(abs_coeff)
    
    mask = np.atleast_2d(mask)
    
    return fluo_yield * np.trapz(vmass * mask * abs_coeff * intensity, energy, axis=1)


def get_fluorescence_data(element):
    
    return dm.dict_fluo_yield[element], dm.dict_fluo_K[element][1], dm.dict_fluo_K[element][2]  


def is_fluorescent(element):
    
    return dm.dict_fluo_K[element][0]
