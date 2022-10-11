#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import os

import numpy as np

import LaueTools.Daxm.material.dict_datamat as dm

__path__ = os.path.dirname(__file__)


class MaterialDataError(Exception):
    pass


def calc_absorption(material, energy=None, absolute=False):

    # energy scale
    if energy is None:
        energy = np.arange(1, 30, 0.01)

    else:
        energy = np.array(energy)

    # load or calculate data
    if is_available_absorption(material):

        abscoeff, energy = load_absorption_coeff(material, energy, absolute)

        abscoeff_p = [abscoeff]

    elif material in dm.dict_mat:
        
        abscoeff, energy, abscoeff_p = calc_absorption_mix(dm.dict_mat[material][0],
                                                           np.array(dm.dict_mat[material][1])*dm.dict_mat[material][2],
                                                           energy)

    else:
        raise(MaterialDataError("Absorption of " + material + " is not implemeted!"))

    return abscoeff, energy, abscoeff_p


def calc_absorption_mix(element, vmass, energy=None, abscoeff_p=None):

    if energy is None:
        energy = np.arange(1, 30, 0.01)
        
    else:
        energy = np.array(energy)

    if abscoeff_p is None:
        
        abscoeff_p = []
        
        for elt in element:
            
            coeff, _ = load_absorption_coeff(elt, energy, absolute=False)
            
            abscoeff_p.append(coeff) 

    abscoeff = np.array([c*coeff for c, coeff in zip(vmass, abscoeff_p)])
    
    return np.sum(abscoeff, axis=0), energy, abscoeff_p


def load_absorption_coeff(element, energy, absolute=False):
    filename = os.path.join(__path__, "data", element + ".txt")

    abs_energy = np.loadtxt(filename, usecols=(0,))  # keV

    abs_coeff = np.loadtxt(filename, usecols=(1,))

    if absolute:
        abs_coeff = abs_coeff * dm.dict_density[element]

    abs_coeff = np.interp(energy, abs_energy, abs_coeff) / 10.  # from 1/cm to 1/mm

    abs_energy = energy

    return np.atleast_1d(abs_coeff), np.atleast_1d(abs_energy)


def list_available_element():
    elt = []

    for fn in os.listdir(os.path.join(__path__, "data")):

        if len(fn) < 7 and fn.endswith(".txt"):
            elt.append(fn.split(".")[0])

    return elt


def is_available_absorption(element):
    
    return element in list_available_element()
