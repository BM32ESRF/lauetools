#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import numpy as np
import scipy.integrate as spi

import matplotlib as mpl
import matplotlib.pylab as mplp

from LaueTools.Daxm.utils.list import closest_value_idx, array_indices
from LaueTools.Daxm.utils.num import crop_curve

import LaueTools.Daxm.material.absorption as abso
import LaueTools.Daxm.material.fluorescence as fluo
import LaueTools.Daxm.material.dict_datamat as dm


# fast source creator
def new_source(material, thickness=None, ystep=0.001, angle=40.):

    return SecondarySource(get_compo_from_mat(material, thickness), ystep, angle)

# Composition
def list_available_material():

    return list(dm.dict_mat.keys())


def list_available_element():

    return abso.list_available_element()


def list_available_all():

    mat = list_available_material()

    for item in list_available_element():

        mat.append(item)

    return mat


def is_available_material(material):

    return material in dm.dict_mat


def is_available_element(element):

    return abso.is_available_absorption(element)


def get_compo_from_mat(material, thickness=None, detail=True):

    if thickness is None:
        thickness = 0.3
    else:
        thickness = float(thickness)

    if is_available_element(material):

        compo = [[material, [0, thickness], dm.dict_density[material]]]

    elif is_available_material(material):

        mat = dm.dict_mat[material]

        if detail:

            compo = [[elt, [0, thickness], mat[1][i]*mat[2]] for i, elt in enumerate(mat[0])]

        else:

            compo = [[material, [0, thickness], mat[2]]]

    else:
        raise ValueError("Unknown element or material '{}'.".format(material))
        # compo = []

    print('results compo in get_compo_from_mat()', compo)
    return compo  


def get_density_from_mat(material):

    if is_available_element(material):

        density = dm.dict_density[material]

    elif is_available_material(material):

        density = dm.dict_mat[material][2]

    else:
        raise ValueError("Unknown element or material '{}'.".format(material))
        # density = 0

    return density


# Main source class
class SecondarySource:
    # Constructors
    def __init__(self, arg=None, ystep=0.001, angle=40.):

        self.ystep = ystep
        self.angle = np.radians(angle)

        # components
        self.cmpnt_qty = 0
        self.cmpnt_elt = []
        self.cmpnt_yrange = []
        self.cmpnt_vmass = []

        # component properties
        self.energy = None
        self.cmpnt_abscoeff = []
        self.cmpnt_fluo = []
        self.cmpnt_Kedge = []
        self.cmpnt_yield = []
        self.cmpnt_Kfluo = []

        # phases
        self.phase_cmpnt = []
        self.phase_cmpnt_vmass = []
        self.phase_yrange = []
        self.phase_abscoeff = []
        self.phase_cmpnt_there = []

        # sample mesh
        self.mesh_y = []
        self.mesh_dy = []

        self.mesh_phase = []

        self.mesh_cmpnt_there = []
        self.mesh_cmpnt_vmass = []

        self.mesh_sam_there = []
        self.mesh_sam_vmass = []
        self.mesh_sam_abscoeff = []

        # secondary source
        self.source_ysrc = []
        self.source_energy = []
        self.source_I = []

        self.source_fluo_elt = []
        self.source_fluo_I = []
        self.source_fluo_E = []
        self.source_fluo_mu = []
        self.source_fluo_dy = []

        # initialize
        if arg is not None:

            self.initialize(arg)

    def initialize(self, arg):
        print("arg in  initialize SecondarySource",arg)
        print("arg",type(arg))
        if isinstance(arg, str):

            arg = get_compo_from_mat(arg, 0.3)

        for item in arg:

            self.add_component(*item)

        self.make_incline()

        self.make_mesh()

        self.make_phases()

    # Methods to add components
    def add_component(self, material, yi, vi=None):

        if vi is not None and vi <= 0:
            raise ValueError("Negative density value ({}) for '{}'.".format(vi, material))

        if is_available_element(material):

            if vi is None:
                vi = dm.dict_density[material]

            self.add_element(material, yi, vi)

        elif is_available_material(material):

            if vi is None:
                vi = dm.dict_mat[material][2]

            elt = dm.dict_mat[material][0]
            vi = vi * np.array(dm.dict_mat[material][1])

            for e, v in zip(elt, vi):
                self.add_element(e, yi, v)

        else:
            raise ValueError("Unknown element or material '{}'.".format(material))

    def add_element(self, element, yi, vi):

        if is_available_element(element):

            if element in self.cmpnt_elt:
                self.add_element_prev(element, yi, vi)
            else:
                self.add_element_new(element, yi, vi)

    def add_element_new(self, element, yi, vi):

        # append new component
        self.cmpnt_qty = self.cmpnt_qty + 1

        self.cmpnt_elt.append(element)

        self.cmpnt_yrange.append([np.array(yi)])  

        self.cmpnt_vmass.append([vi])

        # get component properties for fluorescence
        self.cmpnt_fluo.append(fluo.is_fluorescent(element))

        fyield, Eedge, Efluo = fluo.get_fluorescence_data(element)
        
        self.cmpnt_yield.append(fyield)
        
        self.cmpnt_Kedge.append(Eedge)
        
        self.cmpnt_Kfluo.append(Efluo)
        
        # get component properties for absorption
        abscoeff, _, _ = abso.calc_absorption(element, self.energy)
        
        self.cmpnt_abscoeff.append(abscoeff)
    
    def add_element_prev(self, element, yi, vi):
    
        idx = self.cmpnt_elt.index(element)
    
        self.cmpnt_yrange[idx].append(np.array(yi))   
        
        self.cmpnt_vmass[idx].append(vi)
    
    # Other internal constructors
    def make_incline(self):

        self.cmpnt_yrange = [[yi/np.sin(self.angle) for yi in yrange] for yrange in self.cmpnt_yrange]

    def make_energy(self):
        
        energy = np.arange(1, 30, 0.001)
        
        energy = np.concatenate((energy, 
                                 np.array(self.cmpnt_Kedge), 
                                 np.array(self.cmpnt_Kfluo)))    
        
        self.energy = np.unique(energy)
        
        self.cmpnt_abscoeff = []
        
        for element in self.cmpnt_elt:
            
            abscoeff, _, _ = abso.calc_absorption(element, self.energy)
  
            self.cmpnt_abscoeff.append(abscoeff)
    
    def make_mesh(self):
    
        self.make_energy()
    
        self.make_mesh_y()
        
        self.make_mesh_composition()
        
        self.make_mesh_sample()
    
    def make_mesh_y(self):
    
        yi = [item for sublist in self.cmpnt_yrange for item in sublist]
        
        yi = [[item.min(), item.max()] for item in yi]
        
        yi = np.array(yi).flatten()

        ymesh = np.arange(np.amin(yi), np.amax(yi), self.ystep)
        
        self.mesh_y = np.unique(np.append(yi, ymesh))
        
        self.mesh_dy = np.ediff1d(self.mesh_y, to_begin=0)
            
    def make_mesh_composition(self):    
        
        self.mesh_cmpnt_there = []
        
        self.mesh_cmpnt_vmass = []
        
        for yrange, vmass in zip(self.cmpnt_yrange, self.cmpnt_vmass):
            
            self.mesh_cmpnt_there.append(np.zeros(len(self.mesh_y), dtype=bool))
            
            self.mesh_cmpnt_vmass.append(np.zeros(len(self.mesh_y)))
            
            for y, v in zip(yrange, vmass):
            
                there = np.logical_and(self.mesh_y > y.min(),
                                       self.mesh_y <= y.max())
                
                vmass = np.interp(self.mesh_y, y, [v, v], left=0, right=0)
                
                vmass[np.logical_not(there)] = 0
                
                self.mesh_cmpnt_there[-1] = np.logical_or(self.mesh_cmpnt_there[-1], there)
                
                self.mesh_cmpnt_vmass[-1] = self.mesh_cmpnt_vmass[-1] + vmass

        self.mesh_cmpnt_there = np.array(self.mesh_cmpnt_there).transpose()
        
        self.mesh_cmpnt_vmass = np.array(self.mesh_cmpnt_vmass).transpose()
    
    def make_mesh_sample(self):
        
        self.mesh_sam_there = np.any(self.mesh_cmpnt_there, axis=1)
        
        self.mesh_sam_vmass = np.sum(self.mesh_cmpnt_vmass, axis=1)
        
        self.mesh_sam_abscoeff = []
        
        for i, compo in enumerate(self.mesh_cmpnt_vmass):
            
            if self.mesh_sam_there[i]:
                
                coeff, _, _ = abso.calc_absorption_mix(self.cmpnt_elt,
                                                       compo,
                                                       self.energy,
                                                       self.cmpnt_abscoeff)
            
            else:
                
                coeff = np.zeros(len(self.energy))
                
            self.mesh_sam_abscoeff.append(coeff)    

    def make_phases(self):
        
        nb_phase = 0
         
        for i, there in enumerate(self.mesh_sam_there):
                
            if there:
                
                if not (nb_phase and
                        np.array_equal(self.mesh_cmpnt_there[i-1],
                                       self.mesh_cmpnt_there[i]) and
                        np.array_equal(self.mesh_cmpnt_vmass[i-1],
                                       self.mesh_cmpnt_vmass[i])):
                    
                    nb_phase = nb_phase + 1
                
                self.mesh_phase.append(nb_phase-1)
                
            else:
                self.mesh_phase.append(-1)

        phase, index0 = np.unique(np.array(self.mesh_phase), 
                                  return_index=True)
        _, index1 = np.unique(np.array(self.mesh_phase[::-1]),
                              return_index=True)
        
        index1 = len(self.mesh_phase) - 1 - index1
          
        if phase[0] == -1:  # -1 for none phase
            
            phase, index0, index1 = phase[1:], index0[1:], index1[1:]
            
        self.phase_cmpnt_there = array_indices(self.mesh_cmpnt_there, index0)
        self.phase_cmpnt_vmass = array_indices(self.mesh_cmpnt_vmass, index0)
        self.phase_abscoeff = array_indices(self.mesh_sam_abscoeff, index0)
        self.phase_yrange = [array_indices(self.mesh_y, [i, j]) for i, j in zip(index0, index1)]
        
    # Getters
    def get_absorption(self, Elims=None):

        if Elims is None:
            Elims = [4, 26]

        yrange = [(y[0]-self.ystep, y[1]) for y in self.phase_yrange]
        
        abscoeff = [crop_curve(self.energy, c, *Elims)[1] for c in self.phase_abscoeff]
        
        energy, _ = crop_curve(self.energy, self.energy, *Elims)
        
        return abscoeff, energy, yrange
    
    def get_source_trans(self, Ei):
        
        if not len(self.source_I):
         
            self.calc_source()
         
        res = []
         
        for E in Ei:
            idx = closest_value_idx(self.energy, E)
            res.append(self.source_I[:, idx])
             
        return np.array(res), self.source_ysrc      
    
    def get_source_fluo(self):
            
        if not len(self.source_fluo_I):
         
            self.calc_source()
            
        return self.source_fluo_I, self.source_ysrc, self.source_fluo_E, self.source_fluo_elt
    
    def get_fluo_edges(self):
        
        Kedge = [e for i, e in enumerate(self.cmpnt_Kedge) if self.cmpnt_fluo[i]]
        
        Kelt = [e for i, e in enumerate(self.cmpnt_elt) if self.cmpnt_fluo[i]]
        
        return Kedge, Kelt
    
    # Core methods to model source profiles
    def calc_absorption(self, E):

        abscoeff, energy, _ = self.get_absorption()

        return np.interp(E, energy, abscoeff[0])

    def calc_source(self, Esrc=None, Isrc=None):

        if Esrc is None:
            Esrc = [4.999, 25.001]

        if Isrc is None:
            Isrc = [1., 1.]

        self.source_ysrc = self.mesh_y
        
        self.source_energy = self.energy
        
        self.calc_source_trans(Esrc, Isrc)
        
        self.calc_source_fluo(Esrc, Isrc)
    
    def calc_source_trans(self, Esrc, Isrc):
        
        I0 = np.interp(self.source_energy, Esrc, Isrc, left=0, right=0)
         
        arg = spi.cumtrapz(np.array(self.mesh_sam_abscoeff), self.source_ysrc, axis=0, initial=0)
        
        # beer-lambert law: I = I0*exp(-rho.mu.length)
        self.source_I = np.array([I0])*np.exp(-arg)
    
    def calc_source_fluo(self, Esrc, Irsc):
        # /!\ must have computed first the transmitted source
        
        # calculate fluoresced intensity for each single component
        Ifluo, Efluo, cmpnt = [], [], []
         
        for i, elt in enumerate(self.cmpnt_elt):
             
            if self.cmpnt_fluo[i]:
                 
                Ifluo.append(fluo.calc_fluorescence(elt,
                                                    self.source_energy,
                                                    self.source_I,
                                                    self.mesh_cmpnt_vmass[:, i],
                                                    self.cmpnt_abscoeff[i]))
                Efluo.append(self.cmpnt_Kfluo[i])
                cmpnt.append(self.cmpnt_elt[i])
    
        self.source_fluo_elt = cmpnt
        self.source_fluo_I = Ifluo
        self.source_fluo_E = Efluo
        
        # TODO: dy as a function of position y in sample...
        # pre-compute absorption coeffs and lengths through sample
        self.source_fluo_mu = []
        
        for abscoeff in self.phase_abscoeff:
            
            mu = np.interp(self.source_fluo_E, self.energy, abscoeff)
            
            self.source_fluo_mu.append(mu)
            
        self.source_fluo_mu = np.transpose(self.source_fluo_mu)
            
        self.source_fluo_dy = [np.abs(y[1]-y[0]) for y in self.phase_yrange]

    # Plotting methods
    def plot_absorption(self, energy_lims=None, fontsize=14, show=True):

        if energy_lims is None:
            energy_lims = [5., 25.]

        abscoeff, energy, yrange = self.get_absorption([energy_lims[0]-0.1,
                                                        energy_lims[1]+0.1])
        K_energy, K_elt = self.get_fluo_edges()
        
        fig = mplp.figure()
        ax = fig.add_subplot(111)
        
        # plot absorption
       
        ax.plot(energy, np.transpose(abscoeff), linewidth=2)          
    
        ax.set_xlim(energy_lims)
        
        # plot K edges
        for E in K_energy:
            ax.plot([E, E], ax.get_ylim(), '--', linewidth=1)
    
        # labels
        xlabel = 'Energy (keV)'
        ylabel = 'Absorption coefficient (1/mm)'
        title = 'Absorption profile(s) of the sample'
    
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        fig.suptitle(title, fontsize=fontsize+2)
        
        # legend
        lgd = []
        lgd.extend(["Absorption in y = [{}, {}]".format(*y) for y in yrange])
        lgd.extend(["K edge of {} at E = {} keV".format(e, E) for e, E in zip(K_elt, K_energy)])
        
        mplp.legend(lgd, loc='upper right', fontsize=fontsize)
    
        fig.show(show)
    
    def plot_source(self, Ei=None, fontsize=14, show=True):

        if Ei is None:
            Ei = [5, 10, 15, 20, 25]

        Isrc, ysrc = self.get_source_trans(Ei)
        
        fig = mplp.figure()
        
        # 2D plot
        ax1 = fig.add_subplot(111)
         
        ax1.plot(ysrc, Isrc.transpose())
        
        ax1.set_xlim(left=np.min(ysrc), right=np.max(ysrc))

        xlabel = 'depth (mm)'
        ylabel = 'Intensity'
        title = 'Secondary source profile'
        
        ax1.set_xlabel(xlabel, fontsize=fontsize)
        ax1.set_ylabel(ylabel, fontsize=fontsize)

        ax1.legend(["E = {} keV".format(E) for E in Ei], 
                   loc='upper right', fontsize=fontsize)

        fig.suptitle(title, fontsize=fontsize+2)
    
        fig.show(show)

    def plot_source_3d(self, Ei=None, fontsize=14, show=True):

        if Ei is None:
            Ei = [5, 10, 15, 20, 25]

        # 3D plot
        fig = mplp.figure()
        
        ax2 = fig.add_subplot(111, projection='3d')
        
        E, ysrc = np.meshgrid(self.source_energy, self.source_ysrc)
        
        ax2.plot_surface(ysrc, E, self.source_I, linewidth=0, 
                         antialiased=False, cm=mpl.cm.coolwarm)
        
        xlabel = 'depth (mm)'
        ylabel = 'energy (keV)'
        zlabel = 'Intensity'
        title = 'Secondary source profile'
        
        ax2.set_xlabel(xlabel, fontsize=fontsize)
        ax2.set_ylabel(ylabel, fontsize=fontsize)
        ax2.set_zlabel(zlabel, fontsize=fontsize)
        
        fig.suptitle(title,   fontsize=fontsize+2)
    
        fig.show(show)

    def plot_source_fluo(self, fontsize=14, show=True):
        
        Ifluo, ysrc, Kfluo, Kelt = self.get_source_fluo()
            
        I0 = np.sum(Ifluo, axis=0).max()
        
        xlabel = 'Depth (mm)'
        ylabel = 'Intensity'
        title = 'Fluorescence profile(s) of the sample'
     
        fig = mplp.figure()
        ax = fig.add_subplot(111)
         
        ysrc_plot = np.append(np.array(ysrc), ysrc[-1]+1E-8)
        Ifluo_plot = [np.append(I, 0) for I in np.array(Ifluo)/I0] 
         
        ax.plot(ysrc_plot, np.transpose(Ifluo_plot), linewidth=2)
         
        ax.set_xlim(left=np.min(ysrc)-0.005, right=np.max(ysrc)+0.005)
         
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        fig.suptitle(title,   fontsize=fontsize+2)
         
        mplp.legend(["{} - E = {} keV".format(elt, E) for elt, E in zip(Kelt, Kfluo)],
                    loc='upper right', fontsize=fontsize)
        
        fig.show(show)


# Test
if __name__ == '__main__':
      
    ini = [("Ge", [-0.2, -0.1], None),
           ("Ni", [0, 0.1], None),
           ("316L", [0.1, 0.3], None)]
    
    # ini = [("316L", [0, 0.3], None)]
    test = SecondarySource(arg=ini)
    
    test.calc_source()
    
    test.plot_source()

    mplp.show()
