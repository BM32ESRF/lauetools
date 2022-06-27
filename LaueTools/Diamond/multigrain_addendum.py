# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:50:44 2017

@author: robach
"""

print("multigrain_addendum module in DiamondAnalysis  folder")
from time import time, asctime
import math
import sys
import os
import time

from copy import copy
import numpy as np
from numpy import linalg as LA

import matplotlib.pylab as p
import matplotlib as mpl
import matplotlib.colors as colors
if sys.version_info.major == 3:
    from .. import dict_LaueTools as DictLT
    from .. import LaueGeometry as F2TC
    from .. import findorient as FindO
    from .. import generaltools as GT
    from .. import readmccd as rmccd
    #WARNING looped imports !!
    from . import diamond as DIA
    from . import multigrain as MG
else:
    import dict_LaueTools as DictLT
    import find2thetachi as F2TC
    import findorient as FindO
    import generaltools as GT
    import readmccd as rmccd
    from . import diamond as DIA
    from . import multigrain as MG

MG.PAR.cr_string = os.linesep

class column_file :
    """
    Parse the columns in a MG summary file object 
    text file with data columns
    list of column names on second line in file
    
    """                     
    def __init__(self, filename, verbose=False, only_one_line_in_header = False):

        dict_few_columns_float_variables = {

             'matstarlab' : ["matstarlab_0",9],
             'euler3' : ["euler3_0",3],
             'strain6_crystal' : ['strain6_crystal_0', 6] ,
             'strain6_sample' : ["strain6_sample_0",6],
             'rgb_x_sample' : ["rgb_x_sample_0",3],
             'rgb_z_sample' : ["rgb_z_sample_0",3],
             'stress6_crystal' : ["stress6_crystal_0",6],
             'stress6_sample' : ["stress6_sample_0",6],
             'res_shear_stress' : ["res_shear_stress_0",12],
             'xyfit' : ['xfit',2],
            'xymax' : ['xmax',2],
             'xywidth' : ["xwidth",2],
            "w_mrad" : ["w_mrad_0",3],
            'dxymicrons' : ['dxymicrons_0',2],
            'xy1' : ['xy1_0',2],            
            'xy2' : ['xy2_0',2],
             }     
        
        list_one_column_int_variables = ['img', 'gnumloc', 'npeaks']
        list_one_column_float_variables = ['pixdev', 'intensity', "Imax",
                                           "Ibox", 'max_rss','von_mises',
                                           "eq_strain", 'misorientation_angle',
                                           'maxpixdev','stdpixdev',"dalf","Monitor"]
         
        if only_one_line_in_header == False :
            skiprows1 = 2
        else : skiprows1 = 1
        try :            
            if verbose :
                print("reading summary file")
                print("first two lines :")
            f = open(filename, 'r')
            self.filename = filename
            print(filename)
            i = 0
            nameline0 = ""
            try:
                for line in f:
                    if only_one_line_in_header == False :
                        if i == 0 : nameline0 = line.rstrip("  "+ MG.PAR.cr_string)
                        if i == 1 : nameline1 = line.rstrip(MG.PAR.cr_string)
                        i = i+1
                        if i>2 : break
                    else :
                        if i == 0 : nameline1 = line.rstrip(MG.PAR.cr_string)
                        i = i+1
                        if i>1 : break
            finally:
                f.close() 
            
            if verbose :    
                print(nameline0)   
                print(nameline1)
            
            self.all_columns = np.loadtxt(filename, skiprows = skiprows1)
            
            print(np.shape(self.all_columns))
            
            self.list_group_column_names = nameline0
            
            self.list_single_column_names = nameline1.split()
            
            print(self.list_single_column_names)
            
            print("reading columns :")
            
            for str1 in list_one_column_int_variables : 
                if str1 in self.list_single_column_names :
                    ind1 = self.list_single_column_names.index(str1) 
                    print(str1, ind1)
                    setattr(self,str1,np.array(self.all_columns[:,ind1],int))
                    if verbose : print(str1 , " = ", getattr(self,str1,False))
                    
            for str1 in list_one_column_float_variables : 
                if str1 in self.list_single_column_names :
                    ind1 = self.list_single_column_names.index(str1) 
                    print(str1, ind1)
                    setattr(self,str1,np.array(self.all_columns[:,ind1],float))
                    if verbose : print(str1 , " = ", getattr(self,str1,False))
                    
            for var1 in list(dict_few_columns_float_variables.keys()) : 
                str1 = dict_few_columns_float_variables[var1][0]
                if str1 in self.list_single_column_names :
                    ind1 = self.list_single_column_names.index(str1) 
                    print(var1, str1, ind1)
                    ncols = dict_few_columns_float_variables[var1][1]
                    setattr(self,var1,np.array(self.all_columns[:,ind1:ind1+ncols],float))
                    if verbose : print(var1 , " = ", getattr(self,var1,False))
                    
        except IOError:
            print("file {} not found! or problem of reading it!".format(filename))
            pass              

#class calib_file :
#    
#    def __init__(self, filename, verbose=False):
#        
#        try:
#            _file = open(filename, 'r')
#            text = _file.readlines()
#            _file.close()
#
#            # first line contains parameters
#            parameters = [float(elem) for elem in str(text[0]).split(',')]
#            self.currentvalues = np.array(parameters)
#            print self.currentvalues
#            # others are comments
#            comments = text[1:]
#            
#            self.controltext = ['Distance', 'xcen', 'ycen', 'betangle', 'gammaangle',
#                              'pixelsize', 'dim1', 'dim2']
#
##            for str1 in self.controltext :
##                ind1 = self.controltext.index(str1) 
##                print str1, ind1
##                setattr(self,str1,float(self.currentvalues[:,ind1]))
##                if verbose : print str1 , " = ", getattr(self,str1,False)   
#                
#            allcomments = ''
#            for line in comments:
#                allcomments += line
#
#            self.comments.SetValue(str(allcomments))
#            
#        except IOError:
#            print "file {} not found! or problem of reading it!".format(filename)
#            pass  
        
def plot_map2(filesum, 
             filexyz = None, 
             maptype = "pixdev", 
             subtract_mean = 0,
             gnumloc = 0,
             filter_on_pixdev_and_npeaks = 0,
             maxpixdev_forfilter = 0.3, # only for filter_on_pixdev_and_npeaks = 1
             minnpeaks_forfilter = 20, # only for filter_on_pixdev_and_npeaks = 1
             min_forplot = -0.2,   
             max_forplot = 0.2, # use None for autoscale, scalar or list           
             zoom = "no",
             xylim = None,
             filter_mean_strain_on_misorientation = 0,
             max_misorientation = 0.15, # only for filter_mean_strain_on_misorientation = 1
             change_sign_xy_xz = 0,
             subtract_constant = None,
             remove_ticklabels_titles = 0,
             col_for_simple_map = None, # utiliser avec maptype = "dalf" (+/-) ou "intensity" (+)
             use_mrad_for_misorientation = "no", # only for maptype = "misorientation_angle"
             color_for_duplicate_images = None, # [0.,1.,0.]
             filter_on_intensity = 0,
             min_intensity_forfilter = 20000., # only for filter_on_intensity = 1
             plot_grid = 0,
             savefig = 0,
             fileprefix = "",
             add_symbols_for_sign = 0,
             add_symbols_at_bad_points = 0,
             plot_histograms = 0,
             min_forhist = None,
             max_forhist = None,
             xlim_forhist = None,
             filesum_contains_indlinkimg_column = 0,  #fastest
             filexyz_contains_ijcolumns = 0,   # less fast
             filepath = None, # only for filexyz_contains_ijcolumns = 0
             grid_order = "1234", #only for filexyz_contains_ijcolumns = 0
             plot_curves_along_lines = 0,
             xylim_of_lines = None,
             show_raw_data_and_filter_instead_of_filtered_data = 0,
             show_mask = 0,
             use_min_max_for_limits = 1,
             use_mean_std_for_limits = 0,
             filter_on_matrix_components = 0,
             matrix_component_num_for_filter = 0,
             matrix_component_range_for_filter = None,
             xyfilter_as_first_filter = 0,
             subtract_xypic = None, # for maptype = "xyfit" or "xymax"
             imgref_for_ref_matrix = None, # for maptype = "Rxyz_recalculated"
             max_dalf_for_mean_matrix = None, # for maptype = "Rxyz_recalculated",
             normalize_to_N_monitor_counts = None, # for maptype = "intensity", "Imax" or "Ibox"
             monitor_offset = 15000.,
             only_one_line_in_header_in_summary_file = False
                ):  
       
        #15Feb17 speed up
        
        """
        # gnumloc  = "indexing rank" of grain selected for mapping (for multigrain Laue patterns)
        first grain = grain with most intense spot  
        first grain has gnumloc = 0  (LT summary files)
        first grain has gnumloc = 1 (XMAS summary files (rebuilt))
        
        maptype =  "fit" 
                or "euler3" or "rgb_x_sample"     
                or "strain6_crystal" or "strain6_sample" 
                or "stress6_crystal" or "stress6_sample"
                or "res_shear_stress"
                or 'max_rss'
                or 'von_mises'
                
        min/max = -/+ strainscale for strain plots
        (and quantities derived from strain)
        
        """        
        
        list_column_names =  ['img', 'gnumloc', 'npeaks', 'pixdev', 'intensity', 
                                        'dxymicrons_0', 'dxymicrons_1', 
 'matstarlab_0', 'matstarlab_1', 'matstarlab_2', 'matstarlab_3', 'matstarlab_4','matstarlab_5', 'matstarlab_6', 'matstarlab_7', 'matstarlab_8', 
 'strain6_crystal_0', 'strain6_crystal_1', 'strain6_crystal_2', 'strain6_crystal_3', 'strain6_crystal_4', 'strain6_crystal_5', 
 'euler3_0', 'euler3_1', 'euler3_2', 
 'strain6_sample_0', 'strain6_sample_1', 'strain6_sample_2', 'strain6_sample_3', 'strain6_sample_4', 'strain6_sample_5', 
 'rgb_x_sample_0', 'rgb_x_sample_1', 'rgb_x_sample_2', 
 'rgb_z_sample_0', 'rgb_z_sample_1', 'rgb_z_sample_2', 
 'stress6_crystal_0', 'stress6_crystal_1', 'stress6_crystal_2', 'stress6_crystal_3', 'stress6_crystal_4', 'stress6_crystal_5', 
 'stress6_sample_0', 'stress6_sample_1', 'stress6_sample_2', 'stress6_sample_3', 'stress6_sample_4', 'stress6_sample_5', 
 'res_shear_stress_0', 'res_shear_stress_1', 'res_shear_stress_2', 'res_shear_stress_3', 'res_shear_stress_4', 'res_shear_stress_5', 'res_shear_stress_6', 'res_shear_stress_7', 'res_shear_stress_8', 'res_shear_stress_9', 'res_shear_stress_10', 'res_shear_stress_11',
 'max_rss', 'von_mises', 'misorientation_angle', 'dalf'] 
 
 
#  NB : misorientation_angle column seulement pour analyse mono-grain
# NB : dalf column seulement pour mat2spots ou fit calib

#        list_column_names =  ['img', 'gnumloc', 'npeaks', 'pixdev']

#img Ibox xfit yfit xwidth ywidth Imax xmax ymax twosintheta_x1000 

        color_grid = "k"
        
        if col_for_simple_map  is  not None :
            filter_on_pixdev_and_npeaks = 0
            filter_mean_strain_on_misorientation = 0
            
        col_obj = column_file(filesum, only_one_line_in_header = only_one_line_in_header_in_summary_file)  
        
        data_list = np.array(col_obj.all_columns, float)
        
#        data_list, listname, nameline0 = MG.read_summary_file(filesum)      
#        data_list = np.array(data_list, dtype=float)        
        nimg_filesum = np.shape(data_list)[0]
        print(nimg_filesum)
        ndata_cols = np.shape(data_list)[1]
        print(ndata_cols)
        
#        indimg = listname.index('img')
#        pixdevlist = np.zeros(nimg_filesum, float)   
#        gnumlist = np.zeros(nimg_filesum, int)       
#        npeakslist = np.ones(nimg_filesum, int)*25
        xylist = np.zeros((nimg_filesum,2),float)
#        matlist = np.zeros((nimg_filesum,9),float)
        
#        if filter_on_intensity :
#            indintensity = listname.index('intensity')
#            intensitylist = np.array(data_list[:,indintensity],dtype = float)
            
        gnumloc_col_exists = 0   # pour pouvoir traiter les analyses mono-spot  
#        if "gnumloc" in listname : 
#            gnumloc_col_exists = 1
#            indgnumloc = listname.index('gnumloc')
#            gnumlist = np.array(data_list[:,indgnumloc],dtype = int)   
        if "gnumloc" in col_obj.list_single_column_names : gnumloc_col_exists = 1
        
        npeaks_col_exists = 0    
        if "npeaks" in col_obj.list_single_column_names : npeaks_col_exists = 1
#        if 'npeaks' in listname : 
#            npeaks_col_exists = 1
#            indnpeaks = listname.index('npeaks')
#            npeakslist = np.array(data_list[:,indnpeaks],dtype = int)        
#        if 'pixdev' in listname :
#            indpixdev = listname.index('pixdev')
#            pixdevlist = data_list[:,indpixdev]        
#        dxymicrons_col_exists = 0
#        if 'dxymicrons_0' in col_obj.list_single_column_names : dxymicrons_col_exists = 1
#            indxech = listname.index('dxymicrons_0')
#            xylist = np.array(data_list[:,indxech:indxech+2],dtype = float)  
                   
#        if 'misorientation_angle' in listname :
#            indmisor = listname.index('misorientation_angle')

        if filter_mean_strain_on_misorientation :
            print("for single grain")
            print("for subtract_mean = 1 and maptype = strain or stress") 
            print("exclude points with large misorientation from mean calculation")
#            misor_list = np.array(data_list[:,indmisor],dtype = float)
            if use_mrad_for_misorientation == "yes":
                print("converting misorientation angle into mrad")
                misor_list =  col_obj.misorientation_angle * math.pi/180.*1000.
            indm = np.where(misor_list < max_misorientation)
            print("excluding points with large misorientation > ", max_misorientation)
            print("nimg after filtering : ", np.shape(indm)[1])
                
#        if "ijpos_0" in listname :
#            indijpos = listname.index("ijpos_0")
#            ijposlist = np.array(data_list[:,indijpos],dtype = int)
            
#        if 'matstarlab_0' in listname :
#            indmat = listname.index('matstarlab_0')
#            matlist = np.array(data_list[:,indmat:indmat+9],dtype = float) 
            
        matref = None

        # for 2-spots calculation, filesum = filemat
        if ('dalf' in col_obj.list_single_column_names) and (maptype == "Rxyz_recalculated") and (max_dalf_for_mean_matrix  is  not None) :
            print("recalculate rotations from orientation matrices")
            print("use mean orientation matrix as reference matrix")
#            inddalf = listname.index('dalf')
#            dalf_list = np.array(data_list[:,inddalf],dtype = float)
            absdalf_list = abs(col_obj.dalf)
            indm = np.where(absdalf_list < max_dalf_for_mean_matrix)
            print("for mean matrix : filter out img with large dalf > ", max_dalf_for_mean_matrix) 
            print("nimg with low dalf : ", np.shape(indm)[1])  
            matref = col_obj.matstarlab[indm[0],:].mean(axis = 0)
            print("filtered mean matrix")
            print(matref.round(decimals=6))
                  
        # key = maptype 
        # fields = ncolplot, nplot, ngraph, ngraphline, ngraphcol, ngraphlabels
        # ncolplot = nb of columns for these data
        # nplot = 3 per rgb color map
        # ngraph = number of graphs
        # ngraphline, ngraphcol = subplots
        # ngraphlabels = subplot number -1 for putting xlabel and ylabel on axes
        # list of labels for each graph
        # color_under
        # color_over
        # color_bad
        
        dict_nplot = {
            "euler3_rgb" : [3,3,1,1,1,0,["rgb_euler",],],
            "euler3" : [3,3,3,1,3,0,["euler3_0","euler3_1","euler3_2"],],
            "rgb_x_sample" : [9,9,3,1,3,0,["x_sample","y_sample", "z_sample"],],
            "rgb_x_lab" : [9,9,3,1,3,0,["x_lab","y_lab", "z_lab"],],
            "strain6_crystal" : [6,6,6,2,3,3,["aa","bb","cc","ca","bc","ab"],], 
            "strain6_sample" : [6,6,6,2,3,3, ["XX","YY","ZZ","YZ","XZ","XY"],
                                np.array([0.,0.,1.]), np.array([1.,0.,0.]),"k", mpl.cm.RdBu_r, "(1e-3)",], 
            "stress6_crystal" : [6,6,6,2,3,3, ["aa","bb","cc","ca","bc","ab"],], 
            "stress6_sample" : [6,6,6,2,3,3,["XX","YY","ZZ","YZ","XZ","XY"],],
#            "w_mrad" : [3,3,3,1,3,0,["RX","RY","RZ"]],  # 3 graphes sur une ligne
#           "w_mrad" : [3,3,3,1,3,0,["RX","RY","RZ"],
#                        np.array([0.,0.,1.]), np.array([1.,0.,0.]),"k", mpl.cm.RdBu_r, "(mrad)"], # 3 graphes sur une ligne
            "Rxyz_recalculated" : [3,3,3,1,3,0,["RX","RY","RZ"],
                        np.array([0.,0.,1.]), np.array([1.,0.,0.]),"k", mpl.cm.RdBu_r, "(mrad)"], # 3 graphes sur une ligne
            "w_mrad" : [3,3,3,3,1,0,["RX","RY","RZ"],
                        np.array([0.,0.,1.]), np.array([1.,0.,0.]),"k", mpl.cm.RdBu_r, "(mrad)"], # 3 graphes sur une colonne
            "w_mrad_gradient" : [6,6,6,2,3,3,["dRX/x","dRY/x","dRZ/x","dRX/y","dRY/y","dRZ/y" ],
                        np.array([0.,0.,1.]), np.array([1.,0.,0.]),"k", mpl.cm.RdBu_r, "(mrad/pixel)"], #
            "res_shear_stress": [12,12,12,3,4,8,["rss0", "rss1","rss2", "rss3","rss4", "rss5","rss6", "rss7","rss8", "rss9","rss10", "rss11"]],
            'max_rss': [1,1,1,1,1, 0,["max_rss",],],
            'von_mises': [1,1,1,1,1,0,["von Mises stress"],
                          None, np.array([0.8,0.,0.]),np.array([1.0,0.8,0.8]),mpl.cm.Greys,"(100 MPa)",],
            'eq_strain': [1,1,1,1,1,0,["equivalent strain",], 
                          None, np.array([0.8,0.,0.]),np.array([1.0,0.8,0.8]),mpl.cm.Greys,"(1e-3)"],
            'misorientation_angle': [1,3,1,1,1,0,["misorientation angle",],
                     np.array([0.,1.,0.]),np.array([0.8,0.,0.]),np.array([1.0,0.8,0.8]),mpl.cm.Greys],
            'intensity': [1,1,1,1,1,0,["intensity",],np.array([0.8,0.,0.]),
                          np.array([0.,1.,0.]),np.array([1.0,0.8,0.8]),mpl.cm.Greys, "(counts)"],
            'Imax': [1,1,1,1,1,0,["Imax",],np.array([0.8,0.,0.]),
                          np.array([0.,1.,0.]),np.array([1.0,0.8,0.8]),mpl.cm.Greys, "(counts)"],
            'Ibox': [1,1,1,1,1,0,["Ibox",],np.array([0.8,0.,0.]),
                          np.array([0.,1.,0.]),np.array([1.0,0.8,0.8]),mpl.cm.Greys, "(counts)"],
            'Ipix0': [1,1,1,1,1,0,["Ipix0",],np.array([0.8,0.,0.]),
                          np.array([0.,1.,0.]),np.array([1.0,0.8,0.8]),mpl.cm.Greys, "(counts)"],
            'maxpixdev': [1,1,1,1,1,0,["maxpixdev",],np.array([0.,1.,0.]),
                          np.array([0.8,0.,0.]),np.array([1.0,0.8,0.8]),mpl.cm.Greys_r,"(pixel)"],
            'stdpixdev': [1,1,1,1,1,0,["stdpixdev",],np.array([0.,1.,0.]),
                          np.array([0.8,0.,0.]),np.array([1.0,0.8,0.8]),mpl.cm.Greys_r,"(pixel)"],
            "npeaks" : [1,1,1,1,1,0,["npeaks",],np.array([0.8,0.,0.]),
                        np.array([0.,1.,0.]),np.array([1.0,0.8,0.8]),mpl.cm.Greys, None],
            "pixdev" : [1,1,1,1,1,0,["pixdev"],np.array([0.,1.,0.]),
                        np.array([0.8,0.,0.]),np.array([1.0,0.8,0.8]),mpl.cm.Greys_r,"(pixel)"],
            "dalf" : [1,1,1,1,1,0,["delta_alf exp-theor"],
                      np.array([0.,0.,1.]), np.array([1.,0.,0.]),"k", mpl.cm.RdBu_r, "(mrad)"],
            "xmax" : [2,2,2,1,2,0,["xmax", "ymax"],
                      np.array([0.,0.,1.]), np.array([1.,0.,0.]),"k", mpl.cm.RdBu_r, "(pixel)"],
            "xfit" : [2,2,2,1,2,0,["xfit", "yfit"],
                      np.array([0.,0.,1.]), np.array([1.,0.,0.]),"k", mpl.cm.RdBu_r, "(pixel)"],
            "xwidth" : [2,2,2,1,2,0,["xwidth", "ywidth"],
                      np.array([0.,1.,0.]), np.array([1.,0.,0.]),"k", mpl.cm.Greys, "(pixel)"],

            "matstarlab": [9,9,9,3,3,6,["mat0", "mat1","mat2", "mat3","mat4", "mat5","mat6", "mat7","mat8", "mat9"],
                           np.array([0.,0.,1.]), np.array([1.,0.,0.]),"k", mpl.cm.RdBu_r, None],
            
            }

        list_maptype_rgb = ["euler3_rgb","rgb_x_sample","rgb_x_lab"]
            
        ncolplot = dict_nplot[maptype][0]
        nplot = dict_nplot[maptype][1]
        ngraph = dict_nplot[maptype][2]
        ngraphline = dict_nplot[maptype][3]
        ngraphcol = dict_nplot[maptype][4]
        ngraphlabels = dict_nplot[maptype][5]
        list_of_labels = dict_nplot[maptype][6]
        
        print("enter plot_map2",asctime())
        
        print("ncolplot, nplot, ngraph, ngraphline, ngraphcol, ngraphlabels")
        print(ncolplot,  nplot, ngraph, ngraphline, ngraphcol, ngraphlabels)  
        print("list of labels",  list_of_labels)      
                
        palette = copy.copy(dict_nplot[maptype][10])        

        color_under = dict_nplot[maptype][7]
        color_over = dict_nplot[maptype][8]
        color_bad = dict_nplot[maptype][9]
        
#        if min_forplot  is None :
#            color_under = None
#        if max_forplot  is None :
#            color_over = None
        
        if add_symbols_for_sign :  
                color_bad =  np.array([1.0,0.8,0.8]) 
                
        if color_over  is  not None :
            palette.set_over(color_over, 1.0)
        if color_under  is  not None :
            palette.set_under(color_under, 1.0)
        if color_bad  is  not None :    
            palette.set_bad(color_bad, 1.0) 

        # calculate indcolplot  ******************

        if ncolplot == 1 :
            map_first_col_name = maptype
            if col_for_simple_map  is  not None :
                map_first_col_name = col_for_simple_map
        else :
            map_first_col_name = maptype + "_0"  
            if col_for_simple_map  is  not None :
                map_first_col_name = col_for_simple_map   
            if maptype == "w_mrad_gradient":
                map_first_col_name = "w_mrad_0"
            if maptype == "xmax":
                map_first_col_name = "xmax"
            if maptype == "xfit":
                map_first_col_name = "xfit"
            if maptype == "xwidth":
                map_first_col_name = "xwidth" 
            if maptype == "Rxyz_recalculated" :
                map_first_col_name = "matstarlab_0"

        ind_first_col = col_obj.list_single_column_names.index(map_first_col_name)    
        print(ind_first_col)        
        indcolplot = np.arange(ind_first_col, ind_first_col + ncolplot)
        
        if maptype == "w_mrad_gradient":
            indcolplot = np.arange(ind_first_col, ind_first_col + 3) 
            
        if maptype == "Rxyz_recalculated":
            indcolplot = np.arange(ind_first_col, ind_first_col + 9)                    

        if plot_curves_along_lines :
            print("xylim_of_lines = ", xylim_of_lines)
            cond_xmin = (xylist[:,0]> xylim_of_lines[0])
            cond_xmax = (xylist[:,0]< xylim_of_lines[1])
            cond_ymin = (xylist[:,1]> xylim_of_lines[2])
            cond_ymax = (xylist[:,1]< xylim_of_lines[3])
            cond_total = cond_xmin  * cond_xmax * cond_ymin * cond_ymax
            ind1 = np.where(cond_total > 0)
#            print "ind1 = ", ind1
            p.figure(num = 5)
            xx = xylist[ind1[0],0]
            color_list = ["r","g","b","k","m","y"]
            for j in range(ncolplot) :
                if j < 6 :
                    p.plot(xx,data_list[ind1[0],indcolplot[j]],"o-", color = color_list[j])
                    print(list_of_labels[j], "circles", color_list[j])
#                    continue
                else :
#                    continue
                    p.plot(xx,data_list[ind1[0],indcolplot[j]],"s-", color = color_list[j-5])
                    print(list_of_labels[j], "squares", color_list[j-5])                  
            p.xlabel("xech_microns")
#            p.ylabel("mrad")
            jklsdf    
            
        same_scale_for_all_components = 1     
        if np.isscalar(max_forplot):
            max_forplot = np.ones(ncolplot,float)*max_forplot
        else : same_scale_for_all_components = 0
        
        if np.isscalar(min_forplot):
            min_forplot = np.ones(ncolplot,float)*min_forplot           
        else : same_scale_for_all_components = 0
        
        if zoom == "yes" :
            listxj = []
            listyi = []

        xylim_new = xylim
        
        # FILEXYZ

        imgxyz, grid_order,dxystep,nlines,ncol=  read_filexyz(filexyz)

        img_in_filexyz = np.array(imgxyz[:,0]+1e-5, int) # verifier si ok avec int
        xy_in_filexyz = np.array(imgxyz[:,1:3], float)
        ij_in_filexyz = np.array(imgxyz[:,3:5]+1e-5, int) # verifier si ok avec int

        # get or calculate ind_link_img ************************* 
 
        if filesum_contains_indlinkimg_column :
            indlink = col_obj.list_single_column_names.index('ind_link_img')
            ind_link_img = np.array(data_list[:,indlink]+0.00001,dtype = int) 
            filexyz_contains_ijcolumns = 1
        else :    
            print("start calculation of ind_link_img list")
            nimg_xyz = len(img_in_filexyz)  
            nimg_min = min(nimg_filesum, nimg_xyz)
            ind_link_img = np.ones(nimg_filesum,int)*(-1)
            dimg = col_obj.img[:nimg_min] - img_in_filexyz[:nimg_min]
            img_offset = 0
            for i in range(nimg_min) :
                if dimg[i] < 1e-3 : 
                    ind_link_img[i] = i
                else :
                    ind_link_img[i] = i + dimg[i] + img_offset
                    dimg2 = col_obj.img[i] - img_in_filexyz[ind_link_img[i]]
#                    print i,  ind_link_img[i], dimg[i]                        
#                    print  i, col_obj.img[i], img_in_filexyz[i], dimg[i], ind_link_img[i],\
#                         img_in_filexyz[ind_link_img[i]], dimg2
                    if abs(dimg2) > 1e-5 :
    #                    print "jump in img num in filexyz => add offset"
                        ind_link_img2 = np.where(img_in_filexyz == col_obj.img[i])
                        img_offset += ind_link_img2[0] - ind_link_img[i] 
    #                    print "img_offset = ", img_offset
                        ind_link_img[i] = ind_link_img2[0]
                        dimg2 = col_obj.img[i] - img_in_filexyz[ind_link_img[i]]
    #                    print  i, col_obj.img[i], img_in_filexyz[i], dimg[i], ind_link_img[i],\
    #                         img_in_filexyz[ind_link_img[i]], dimg2    
                        if abs(dimg2) > 1e-5 :
                            kjlqd
                        
        if not filexyz_contains_ijcolumns :
            print("add i j columns to filexyz")
            print("please update parameter filexyz to use :")
            print(filepath + "filexyz_ij2.dat")
            add_i_j_columns_to_filexyz(filexyz_3col,
                                      filepath = filepath,
                                      grid_order = grid_order)    
                                      
      # FILTERS
                
#        use_filter1 = 0                
        use_filter = 0
        
        cond_total = None
 
        if gnumloc_col_exists : 
            print("grain : ", gnumloc)
            cond_gnumloc = (col_obj.gnumloc == gnumloc)
            cond_total = cond_gnumloc
            
        print("****************************************************")
        use_first_filter = 0
        
        if npeaks_col_exists :
            cond_npeaks0 = (col_obj.npeaks > 0)
            cond_total = cond_total * cond_npeaks0
            use_first_filter = 1
        
        if xylim  is  not None :     
            if 'dxymicrons_0' in col_obj.list_single_column_names :
                xylist = col_obj.dxymicrons
            else :
                xylist = xy_in_filexyz[ind_link_img]
                
            cond_xmin = (xylist[:,0]> xylim[0])
            cond_xmax = (xylist[:,0]< xylim[1])
            cond_ymin = (xylist[:,1]> xylim[2])
            cond_ymax = (xylist[:,1]< xylim[3])
            cond_xy = cond_xmin  * cond_xmax * cond_ymin * cond_ymax 

            if xyfilter_as_first_filter :
                print("filtering on xy :")
                print("xylim = ", xylim)
                cond_total = cond_total * cond_xy

        ind_in_first_filter = None
        
        if cond_total  is  not None :
            ind_in_first_filter = np.where(cond_total > 0)
            print("len(ind_in_first_filter[0]) = ", len(ind_in_first_filter[0]))

        if xylim  is  not None and not xyfilter_as_first_filter :       
            print("filtering on xy :")
            print("xylim = ", xylim)
            if cond_total  is  not None :
                cond_total = cond_total * cond_xy
            else :
                cond_total = cond_xy
            use_filter = 1
        
        if filter_on_pixdev_and_npeaks :
            print("filtering on pixdev and npeaks :")
            print("maxpixdev :", maxpixdev_forfilter)
            print("minnpeaks :", minnpeaks_forfilter)
            cond_pixdev = (col_obj.pixdev < maxpixdev_forfilter)
            cond_npeaks = (col_obj.npeaks > minnpeaks_forfilter)
            cond_total = cond_total * cond_pixdev * cond_npeaks
            use_filter = 1

        if filter_on_intensity :
            print("filtering on intensity :")
            print("min_intensity_forfilter :", min_intensity_forfilter)
            if "intensity" in col_obj.list_single_column_names :
                cond_intensity = (col_obj.intensity > min_intensity_forfilter)
            elif "Imax" in col_obj.list_single_column_names :
                cond_intensity = (col_obj.Imax > min_intensity_forfilter)
            if cond_total  is  not None :
                cond_total = cond_total * cond_intensity
            else : cond_total = cond_intensity
            use_filter = 1

        if filter_on_matrix_components :
            print("filter on matrix component :")
            print("matrix_component_num_for_filter = ", matrix_component_num_for_filter)
            print("matrix_component_range_for_filter = ", matrix_component_range_for_filter)
            matcomp_list = col_obj.matstarlab[:,matrix_component_num_for_filter]
            cond_matcomp_min = (matcomp_list > matrix_component_range_for_filter[0])
            cond_matcomp_max = (matcomp_list < matrix_component_range_for_filter[1])
            cond_total = cond_total * cond_matcomp_min * cond_matcomp_max
            use_filter = 1

        if cond_total  is  not None :
            ind_in_second_filter = np.where(cond_total > 0)           
            print("len(ind_in_second_filter[0]) = ", len(ind_in_second_filter[0]))
    
            data_list_in = data_list[ind_in_second_filter[0],:]   # all columns, bad lines removed
            imglist_in = col_obj.img[ind_in_second_filter[0]]
            
        else : 
            data_list_in = data_list[:,:]*1.
            imglist_in = col_obj.img
            
        if use_filter :
            if use_first_filter :
                data_list_in_nofilter = data_list[ind_in_first_filter[0],:]
            else :
                data_list_in_nofilter = data_list

        data_in = data_list_in[:,indcolplot]   # columns to plot, bad lines removed
        if use_filter :        
            data_in_nofilter = data_list_in_nofilter[:,indcolplot]

#        imglist = np.array(data_list[:,indimg],dtype = int)
        
        if (imgref_for_ref_matrix  is  not None) & (maptype == "Rxyz_recalculated") :
            ind_img_ref = np.where(col_obj.img == imgref_for_ref_matrix)
            if len(ind_img_ref[0]) < 1 :
                print("imgref_for_ref_matrix not found in map")
                jkdlqs
            matref = col_obj.matstarlab[ind_img_ref[0][0],:]
            print("recalculate rotations from orientation matrices")
            print("use as reference matrix the orientation matrix at image :", imgref_for_ref_matrix)
            print("reference matrix")
            print(matref.round(decimals=6))
            
#        imglist_in = np.array(data_list_in[:,indimg],dtype = int)
        
 
                   
        print("nimg_filesum total", nimg_filesum)
        if cond_total  is  not None :
            print("nimg_filesum remaining after filtering", len(ind_in_second_filter[0]))
        
        if add_symbols_at_bad_points and filter_on_pixdev_and_npeaks :
            
            ind_out = np.where((col_obj.gnumloc == gnumloc)&((col_obj.pixdev >= maxpixdev_forfilter)|(npeakslist <=minnpeaks_forfilter)))
            
            print("nimg_filesum filtered out", len(ind_out[0]))

            if len(ind_out[0])>0 :
                list_xy_out = xy_in_filexyz[ind_link_img[ind_out[0]],:] + dxystep/2.0 
        
        # FILL PLOTDAT TABLE
        
        if maptype == "euler3_rgb" : 
            euler3 = data_list_in[:,indcolplot]
            ang0 = 360.0
            ang1 = math.arctan(math.sqrt(2.0))*180.0/MG.PI
            ang2 = 180.0
            ang012 = np.array([ang0, ang1, ang2])
            print(euler3[0,:])
            euler3norm= euler3/ ang012
            print(euler3norm[0,:])
            #print min(euler3[:,0]), max(euler3[:,0])
            #print min(euler3[:,1]), max(euler3[:,1])
            #print min(euler3[:,2]), max(euler3[:,2])
            
            
        else :  
            
            if maptype == "Rxyz_recalculated" :  
                if matref  is None :
                    print("undefined matref")
                    jkldqs
                nimg = np.shape(data_in)[0]             
                RxRyRz_mrad = np.zeros((nimg,3),float)
                dRxRyRz_mrad = np.zeros((nimg,3),float)
                dLxLyLz_mrad = np.zeros((nimg,3),float)
                
                for i in range(nimg):       # patch
    #                print "current matrix"
    #                print data_in[i,:].round(decimals=6)
                    RxRyRz_mrad[i,:], dRxRyRz_mrad[i,:], ang1_mrad, dang1_mrad, dLxLyLz_mrad[i,:] = \
                            MG.twomat_to_RxRyRz_sample_large_strain(matref, 
                                                                    data_in[i,:],
                                                                    verbose = 0,
                                                                    omega = MG.PAR.omega_sample_frame)
                data_in =  RxRyRz_mrad[:,:]                          
                  
            print("maptype = ", maptype)    
                           
            if plot_histograms :
                
                print("plotting histograms")
                
                if maptype == "Rxyz_recalculated" :  
                    
                    data1 = data_in * 1.
                
                else :
                    
                    data2 = data_list[:,indcolplot]  
                    
                    print("data2[0:10,:] = ")
                    print(data2[0:10,:])
                    
                    if ind_in_first_filter  is  not None :
                        data1 = data2[ind_in_first_filter[0],:]
                    else :
                        data1 = data2 * 1.
                                   
                if subtract_xypic  is  not None : 
                    data1 = data1 - subtract_xypic
                    
                if use_filter :            
                    data1_f = data2[ind_in_second_filter[0],:]

                if xylim  is  not None :   
                    title1 = "xylim = "+ str(xylim) + "\n"
                else : title1 = ""
                                    
                numfig = 10
                if max_forhist  is  not None :
                    print("min_forhist, max_forhist", min_forhist, max_forhist)
                    title1 += "min_forhist = " + str(min_forhist) + ", " +  "max_forhist = " + str(max_forhist)
                
                fig1 = p.figure(num = numfig, figsize=(15,10))    
#                p.title(title1)
#                print title1

                nbins = 50
                print("default number of bins :", nbins)
                if max_forhist  is None :                        
                    print("min_forhist, max_forhist not set : use min/max of data")
                    if maptype in ["xmax", "xfit"] : 
                        print("maptype = ", maptype)
                        print("use one pixel per bin") 
                    
                if use_filter :               
                    print("unfiltered (blue) / filtered (red) :")
                else :
                    print("unfiltered  :")                    
                for j in range(ngraph):                
        #            print p.setp(fig1)
        #            print p.getp(fig1)
                    ax = p.subplot(ngraphline, ngraphcol, j+1)
                    print("component : ", j)
                    if max_forhist  is  not None :                  
                        if np.isscalar(max_forhist) :
                            histo = np.histogram(data1[:,j], bins = nbins, range = (min_forhist, max_forhist))
                            if use_filter :    
                                histo_f = np.histogram(data1_f[:,j], bins = nbins, range = (min_forhist, max_forhist))
                               
                        elif len(max_forhist) == ngraph :
                            histo = np.histogram(data1[:,j], bins = nbins, range = (min_forhist[j], max_forhist[j]))
                            if use_filter :    
                                histo_f = np.histogram(data1_f[:,j], bins = nbins, range = (min_forhist[j], max_forhist[j]))
                            
                    else :
                        min_data = min(data1[:,j])
                        max_data = max(data1[:,j])
                        if maptype in ["xmax", "xfit"] : 
                            print("min / max : ", min_data, max_data)
                            min_forhist_local = int(min(data1[:,j]))
                            max_forhist_local = int(max(data1[:,j]+1.))
                            nbins = max_forhist_local-min_forhist_local
                            print("nbins = ", nbins)
                        else :
                            min_forhist_local = min_data
                            max_forhist_local = max_data
                            
                        print("min_forhist, max_forhist", min_forhist_local, max_forhist_local, end=' ') 
                        histo = np.histogram(data1[:,j], bins = nbins, range = (min_forhist_local, max_forhist_local))
                        if use_filter :  
                            histo_f = np.histogram(data1_f[:,j], bins = nbins, range = (min_forhist_local, max_forhist_local))
                #    print "histogram data : ", histo[0]
                #    print "bin edges :",  histo[1]  
                #    print shape(histo[0])
                #    print shape(histo[1])
    #                p.figure(num = j+3)
                    if use_filter :
                        print("max freq = ", max(histo[0]), max(histo_f[0]), end=' ')
                        print("at x=", round(histo[1][np.argmax(histo[0])],2), round(histo_f[1][np.argmax(histo_f[0])],2))
                    else :
                        print("max freq = ", max(histo[0]), end=' ')
                        print("at x=", round(histo[1][np.argmax(histo[0])],2))
                        
                    barwidth = histo[1][1]-histo[1][0]
                #    print "bar width = ", barwidth
                    p.bar(histo[1][:-1], histo[0], width = barwidth, linewidth = 0)
#                    half_barwidth = barwidth/2.
                    if use_filter :
                        p.bar(histo_f[1][:-1], histo_f[0],  color = "r", width = barwidth, linewidth = 0)
    #                xlabel1 = dict_nplot[maptype][6][j]
                    
                    units_label = ""
                    filter_label = ""
                    if dict_nplot[maptype][11]  is  not None :                   
                        units_label =  dict_nplot[maptype][11]
                    xlabel1 =   list_of_labels[j] + " " + units_label              
                    
                    fontsize = 30
                    fontsize = 10
                    p.xlabel(xlabel1, fontsize = fontsize)
                    p.ylabel("frequency",fontsize = fontsize )  
    #                p.ylim(0.,31000.)
    #                p.xlim(-40.,40.)  # w_mrad site 2 EBSD
                    if xlim_forhist  is  not None :
                        print("yoho")
                        p.xlim(xlim_forhist[0],xlim_forhist[1])
#                    p.xlim(-2.,2.)  # strain site 2 Laue       

                    ax.tick_params(axis='both', which='major', labelsize=fontsize)
                print("exit via dummy command")


            if (maptype == 'misorientation_angle')&(use_mrad_for_misorientation == "yes"):
                print("converting misorientation angle into mrad")
                data_in = data_in * math.pi/180. *1000.               
            
            print("np.shape(data_in) = ", np.shape(data_in))
            
            if 0 : # check sign
                xech_list = data_list_in[:,indxech]
                print(data_in[:,0])
                p.figure()
                p.plot(xech_list,  data_in[:,0], 'ro-')
                p.ylim(-3,3)
            
            if change_sign_xy_xz & (maptype == "strain6_sample"):
                data_in[:,4] = -data_in[:,4]
                data_in[:,5] = -data_in[:,5]
                
            if filter_mean_strain_on_misorientation :
                data_in_mean = data_in[indm[0]].mean(axis = 0)
            else :    
                data_in_mean = data_in.mean(axis = 0)

            if subtract_mean :
                print("subtract mean")
                data_in = data_in - data_in_mean
                print("mean : ", data_in_mean.round(decimals=2))
                data_in_mean = np.zeros(ncolplot,float)
                
            if subtract_xypic  is  not None :
                print("subtract xypic")
                data_in = data_in - subtract_xypic

            if subtract_constant  is  not None :
                if np.isscalar(subtract_constant) :
                    imgref_for_subtract = subtract_constant
                    ind4 = np.where(imglist_in == imgref_for_subtract)               
                    print("subtracting value at ref image, image = ", imgref_for_subtract)
                    print("subtracted value = ", (data_in[ind4[0]]).round(decimals=2))
                    print("ind4 = ", ind4)
                    data_in = data_in - data_in[ind4[0]]
                elif len(subtract_constant) == ngraph :
                    print("subtracting constant = ", subtract_constant)
                    data_in = data_in - subtract_constant

	        # print("normalize_to_N_monitor_counts",normalize_to_N_monitor_counts)

            if (normalize_to_N_monitor_counts is not None) and (maptype in ["intensity","Imax","Ibox"]):
                print("normalize intensity to N monitor counts, N = ", normalize_to_N_monitor_counts)
                print("monitor_offset = ", monitor_offset)
                print("mean value of Monitor column = ", col_obj.Monitor.mean())
                toto = col_obj.Monitor - monitor_offset
                len1 = len(toto)
                indcheck = np.where(toto < 0.)
                if len(indcheck[0]) > 0 :
                    print("Monitor column < monitor_offset for npts points, npts = ", len(indcheck))
                    jkdfls
                print(np.shape(toto))
                print(np.shape(data_in))
                data_in = data_in[:,0] / toto 
                data_in = data_in * normalize_to_N_monitor_counts    
                print(np.shape(data_in))
                data_in = np.reshape(data_in, (len1,1))
                print(np.shape(data_in))
                    
            data_in_min = data_in.min(axis = 0)
            data_in_max = data_in.max(axis = 0)
            data_in_std = data_in.std(axis=0)
            
            print(list_of_labels)
            print("statistics on data table (after filtering) :")
            print("min : ", data_in_min.round(decimals=2))
            print("max : ", data_in_max.round(decimals=2))
            print("mean : ", data_in_mean.round(decimals=2))
            print("std : ", data_in_std.round(decimals=2))
            
            if min_forplot  is None : 
                if use_min_max_for_limits :
                    print("use min/max for limits")
                    min_forplot = data_in_min * 1.
                elif use_mean_std_for_limits :
                    print("use mean+/-std for limits")
                    min_forplot = data_in_mean - data_in_std
            if max_forplot  is None : 
                if use_min_max_for_limits :
                    max_forplot = data_in_max *1.
                elif use_mean_std_for_limits :
                    max_forplot = data_in_mean + data_in_std
            
            print("scale limits for color map :")
            print("min : ", min_forplot.round(decimals=2))
            print("max : ", max_forplot.round(decimals=2))
            
        if cond_total  is  not None :
            print("filtering")
            ind_link_img_in = ind_link_img[ind_in_second_filter[0]]
        else :
            print("no filtering")
            ind_link_img_in = ind_link_img

        if use_filter and show_raw_data_and_filter_instead_of_filtered_data :                        
            ind_link_img_in_nofilter = ind_link_img[ind_in_first_filter[0]]
            
        if add_symbols_for_sign or add_symbols_at_bad_points :
            
            # partie non verifiee Feb 2017
 
            # attention data_in a éventuellement plusieurs colonnes
            # donc ind_strain a aussi plusieurs colonnes
            ind_strain_below_min_negative = np.where(data_in < min_forplot)  
            if len(ind_strain_below_min_negative[0])>0 :
                list_xy_strain_below_min_negative = xy_in_filexyz[ind_link_img_in[ind_strain_below_min_negative[0]]] +  dxystep/2.0 
           
            ind_strain_above_max_positive = np.where(data_in > max_forplot)
            if len(ind_strain_above_max_positive[0])>0 :
                list_xy_strain_above_max_positive = xy_in_filexyz[ind_link_img_in[ind_strain_above_max_positive[0]]] +  dxystep/2.0 
            
            ind_strain_positive = np.where(data_in > 0.) 
            if len(ind_strain_positive[0])>0 :
                list_xy_strain_positive = xy_in_filexyz[ind_link_img_in[ind_strain_positive[0]]] +  dxystep/2.0 
                            
            ind_strain_negative = np.where(data_in < 0.)         
            if len(ind_strain_negative[0])>0 :
                list_xy_strain_negative = xy_in_filexyz[ind_link_img_in[ind_strain_negative[0]]] +  dxystep/2.0 
                                   
        nimg_filesum_in = np.shape(data_list_in)[0]

        if cond_total  is  not None :
            xylist_in = xylist[ind_in_second_filter[0],:]
        else :
            xylist_in = xylist
            
        # add 08Oct14 
        if add_symbols_at_bad_points & (maptype == "fit") :
            
            pixdevlist_in = col_obj.pixdev[ind_in_second_filter[0]]          
#            level1 = 0.25
            level2 = 0.5
            ind_pixdev_above_level2 = np.where(pixdevlist_in > level2)
#            print "ind_pixdev_above_level2 = ", ind_pixdev_above_level2[0]

            if len(ind_pixdev_above_level2[0])>0 :
                list_xy_pixdev_above_level2 = xy_in_filexyz[ind_link_img_in[ind_pixdev_above_level2[0]]] +  dxystep/2.0 
            
            npeakslist_in = npeakslist[ind_in_second_filter[0]]
            level3 = 20.
            ind_npeaks_below_level3 = np.where(npeakslist_in < level3)
            
            if len(ind_npeaks_below_level3[0])>0 :
                list_xy_npeaks_below_level3 = xy_in_filexyz[ind_link_img_in[ind_npeaks_below_level3[0]]] +  dxystep/2.0 
                          
        print("dxystep =", dxystep) 
                         
        dxystep_abs = abs(dxystep)
        


#        ***********************************************
    # ************************************************
    # filling of map   #MAINLOOP
    
        plotdat = -10001.*np.ones((nlines,ncol,nplot),float) 
           
        if use_filter and show_raw_data_and_filter_instead_of_filtered_data :             
            plotdat_nofilter = -10001.*np.ones((nlines,ncol,nplot),float)
            
        if maptype in list_maptype_rgb :
            for j in range(ncolplot):
                plotdat[:,:,3*j:3*j+3] = np.zeros(3,float)
                if use_filter and show_raw_data_and_filter_instead_of_filtered_data :             
                    plotdat_nofilter[:,:,3*j:3*j+3] = np.zeros(3,float)
                                
               
        iref_list_in = ij_in_filexyz[ind_link_img_in,0]
        jref_list_in = ij_in_filexyz[ind_link_img_in,1]
        if use_filter and show_raw_data_and_filter_instead_of_filtered_data :             
            iref_list_in_nofilter = ij_in_filexyz[ind_link_img_in_nofilter,0]
            jref_list_in_nofilter = ij_in_filexyz[ind_link_img_in_nofilter,1]        
                        
        if maptype in list_maptype_rgb :
            palette = None
            norm1 = None
            print("color scale = direct rgb")
        else :
            print("color scale :")
            print("missing / bad =", color_bad)
            print("under = ", color_under)
            print("over = ", color_over)             
            
        if maptype == "euler3_rgb": 
            plotdat[iref_list_in, jref_list_in, :] = euler3norm[:,:]
            
        if maptype == "Rxyz_recalculated": 
            plotdat[iref_list_in, jref_list_in, :] = RxRyRz_mrad[:,:]       

        elif maptype[:5] == "rgb_x": 
            plotdat[iref_list_in, jref_list_in, :] = data_in[:,:]
            
        elif maptype == "w_mrad_gradient" :
            plotdat2 = -10001.*np.ones((nlines,ncol,3),float)
            for j in range(3):
                plotdat2[iref_list_in, jref_list_in,j] = data_in[:,j]
            for j in range(ncolplot):
                if j < 3 : # d/dx
                    plotdat[:,1:,j] = np.diff(plotdat2[:,:,j], axis = 1) # nlines * (ncol-1)
                else : # d/dy
                    if 1 :
                        plotdat[1:,:,j] = np.diff(plotdat2[:,:,j-3], axis = 0)  # (nlines -1) * ncol              
#                    if 1 :
#                        plotdat[1:,1:,j] = np.diff(plotdat2[:,1:,j-3], axis = 0) + plotdat[1:,1:,j-3]
        else :                          
            for j in range(ncolplot):
                if add_symbols_at_bad_points :
                    plotdat[iref_list_in, jref_list_in,j]= abs(data_in[:,j])
                else :
                    plotdat[iref_list_in, jref_list_in,j] = data_in[:,j]
                    if filter_on_pixdev_and_npeaks and show_raw_data_and_filter_instead_of_filtered_data :             
                        plotdat_nofilter[iref_list_in_nofilter, jref_list_in_nofilter,j] = data_in_nofilter[:,j]   
        
        if (zoom == "yes")& npeaks_col_exists :                        
            for i in range(nimg_filesum_in) :       
                    if (col_obj.npeaks[i]> 0) :
                        listxj.append(xylist_in[i,0])
                        listyi.append(xylist_in[i,1])                        
                        
        if color_for_duplicate_images  is  not None and maptype in list_maptype_rgb:
            dimg = abs(np.diff(imglist_in))
            ind1 = np.where(dimg <1e-3)
            if len(ind1[0]) > 0 :
                print("warning : two grains on img ", ind1[0])
                for j in range(ncolplot):
                    plotdat[iref_list_in[ind1[0]],jref_list_in[ind1[0]], 3*j:3*j+3] = color_for_duplicate_images
                    
        # extent corrected 06Feb13
        xrange1 = np.array([0.0,ncol*dxystep[0]])
        yrange1 = np.array([0.0, nlines*dxystep[1]])
        xmin, xmax = min(xrange1), max(xrange1)
        ymin, ymax = min(yrange1), max(yrange1)
        extent = xmin, xmax, ymin, ymax
        print("full map extent (xmin, xmax, ymin, ymax) = ", extent)
        
        if xylim  is  not None :
            print("reduced map extent= ", xylim)
            
        if use_filter and show_mask :
            mask1 = (plotdat[:,:,0] < -10000.)
#            print "shape(mask1)", np.shape(mask1)
#            print "shape(plotdat)", np.shape(plotdat)
            p.figure()
            p.imshow(mask1, 
                     interpolation='nearest', 
                     extent=extent,
                     cmap = mpl.cm.Greys,
                     norm = colors.Normalize(vmin=0., vmax=1.),
                    aspect='equal',
                     )        
                              
            title_mask = "mask : \nwhite = (pixdev < " + str(maxpixdev_forfilter) + ") and (npeaks > " + str(minnpeaks_forfilter) + ")"
            p.title(title_mask)

        print("ngraph, ngraphline, ngraphcol, ngraphlabels")
        print(ngraph, ngraphline, ngraphcol, ngraphlabels)
        print("shape(plotdat)")
        print(np.shape(plotdat))
        
        if zoom == "yes" :         
            listxj = np.array(listxj, dtype = float)
            listyi = np.array(listyi, dtype = float)
            minxj = listxj.min()-2*dxystep_abs[0]
            maxxj = listxj.max()+2*dxystep_abs[0]
            minyi = listyi.min()-2*dxystep_abs[1]
            maxyi = listyi.max()+2*dxystep_abs[1]
            print("zoom : minxj, maxxj, minyi, maxyi : ", minxj, maxxj, minyi, maxyi)
         
        p.rcParams['figure.subplot.right'] = 0.9
        p.rcParams['figure.subplot.left'] = 0.1
        p.rcParams['figure.subplot.bottom'] = 0.1
        p.rcParams['figure.subplot.top'] = 0.9
#        p.rcParams['savefig.bbox'] = "tight"    

        if not show_raw_data_and_filter_instead_of_filtered_data :
            plotdat_m = np.ma.masked_where(plotdat < -10000., plotdat)
            
        # plot figure 

        fig1, ax1 = p.subplots(nrows = ngraphline, ncols = ngraphcol, figsize=(15,10))
#        print "ax1 = ", ax1
#        print "ax1[0] = ", ax1[0]
#        print "shape(ax1) = ", np.shape(ax1)
        if ngraphline>1 and ngraphcol>1 :
            ax2 = ax1.reshape(ngraphline*ngraphcol)
#        toto = np.array([[0,1,2],[3,4,5]])
#        print toto
#        print toto.reshape(6)
#        
        if not same_scale_for_all_components :
            fig1.suptitle(maptype, fontsize = 16)
        
        units_label = ""
        filter_label = ""
        if dict_nplot[maptype][11]  is  not None :                   
            units_label =  dict_nplot[maptype][11]
        if use_filter and not show_raw_data_and_filter_instead_of_filtered_data :
            filter_label = "(filtered)"
              
        if color_over  is None :
            if color_under  is None : extend = "neither"
            else : extend = "min"
        else :
            if color_under  is None : extend = "max"
            else : extend = "both"
        
        for j in range(ngraph):
#            fig1 = p.figure(num = 1, figsize=(15,10))
#            print p.setp(fig1)
#            print p.getp(fig1)
#            ax = p.subplot(ngraphline, ngraphcol, j+1)
#            print "ax = ", ax
#            print "plotdat = ", plotdat[:,:,3*j:3*(j+1)]

            if maptype in list_maptype_rgb :
                imrgb = ax1.imshow(plotdat_m[:,:,3*j:3*(j+1)], 
                                         interpolation='nearest', 
                                         extent=extent)
            else :    
                if ngraph > 1 :
                    norm1 = colors.Normalize(vmin=min_forplot[j], vmax=max_forplot[j])                    
                    if ngraphline>1 and ngraphcol>1 :
                        ax = ax2[j]  
                    else :
                        ax = ax1[j]
                else :
                    norm1 = colors.Normalize(vmin=min_forplot, vmax=max_forplot)
                    ax = ax1

#                print "ax = ", ax                    

                if show_raw_data_and_filter_instead_of_filtered_data :
                   imrgb = ax.imshow(plotdat_nofilter[:,:,j], 
                             interpolation='nearest', 
#                            interpolation = None,
                             extent=extent,
                             cmap=palette,
                             norm=norm1,
                            aspect='equal',
    #                        origin='lower'                       
                             )  
                else :
                    imrgb = ax.imshow(plotdat_m[:,:,j], 
                             interpolation='nearest', 
#                            interpolation = None,
                             extent=extent,
                             cmap=palette,
                             norm=norm1,
                            aspect='equal',
    #                        origin='lower'                       
                             ) 
#                    print "imrgb =", imrgb
                                                       
#            print p.setp(imrgb)
            if col_for_simple_map  is None : 
                strname = dict_nplot[maptype][6][j]
            else : strname = col_for_simple_map
            if not remove_ticklabels_titles : 
                ax.title.set_text(strname)
            if remove_ticklabels_titles :
#                print p.getp(ax)
                p.subplots_adjust(wspace = 0.05,hspace = 0.05)
                p.setp(ax,xticklabels = [])
                p.setp(ax,yticklabels = [])
            if plot_grid :
                ax.grid(color=color_grid, linestyle='-', linewidth=2)
                                
            if MG.PAR.cr_string == "\n":                    
                ax.locator_params('x', tight=True, nbins=5)
                ax.locator_params('y', tight=True, nbins=5)
                
            if not remove_ticklabels_titles :
                labels = [item.get_text() for item in ax.get_xticklabels()]
                print("tick labels = ", labels)
                if (j == ngraphlabels) :
                    ax.set_xlabel("dxech (microns)")
                    ax.set_ylabel("dyech (microns)")
            if zoom == "yes" :
                ax.set_xlim(minxj, maxxj)
                ax.set_ylim(minyi, maxyi)
            if xylim_new  is  not None :
                ax.set_xlim(xylim_new[0], xylim_new[1])
                ax.set_ylim(xylim_new[2], xylim_new[3])

            if add_symbols_at_bad_points and filter_on_pixdev_and_npeaks :

                ms1 = 8
                ms1 = 3
                xysymb = list_xy_out
                if np.isscalar(xysymb[0]) : p.plot(xysymb[0], xysymb[1], "x", ms = ms1, mew = 1, mec = "w", mfc = "None") 
                else : p.plot(xysymb[:,0], xysymb[:,1], "x", ms = ms1, mew = 1, mec = "w", mfc = "None")                          
            
            if add_symbols_for_sign :
                if (maptype == "fit") :
                    if j == 0 :  # npeaks map
                        xysymb = list_xy_npeaks_below_level3
                        if np.isscalar(xysymb[0]) : p.plot(xysymb[0], xysymb[1], "wx", mew = 2) 
                        else : p.plot(xysymb[:,0], xysymb[:,1], "wx", mew = 2)      # "_"          
                    if j == 1 :  # pixdev map
#                        xysymb = list_xy_pixdev_above_level1
#                        p.plot(xysymb[:,0], xysymb[:,1], 'o', mec = "k", mfc = "None")
                        xysymb = list_xy_pixdev_above_level2
                        if np.isscalar(xysymb[0]) : p.plot(xysymb[0], xysymb[1], "wx", mew = 2) 
                        else : p.plot(xysymb[:,0], xysymb[:,1], 'w+', mew = 2)
                
                elif maptype in list_of_positive_or_negative_quantities :
                    
                    if 0 :
                        ind2 = np.where(ind_strain_above_max_positive[1] == j)
    #                    print ind2
    #                    print len(ind2[0])
                        if len(ind2[0]) > 0 :
                            xysymb = list_xy_strain_above_max_positive[ind2[0],:]
                            if np.isscalar(xysymb[0]) : p.plot(xysymb[0], xysymb[1], "w+", mew = 1 )#, mec = "w", mfc = "None") 
                            else : p.plot(xysymb[:,0], xysymb[:,1], "w+", mew = 1 ) # , mec = "w", mfc = "None")                          

                    ind2 = np.where(ind_strain_positive[1] == j)
#                    print ind2
#                    print len(ind2[0])
                    if len(ind2[0]) > 0 :
                        xysymb = list_xy_strain_positive[ind2[0],:]
                        if np.isscalar(xysymb[0]) : p.plot(xysymb[0], xysymb[1], "o", mew = 1, mec = "w", mfc = "None") 
                        else : p.plot(xysymb[:,0], xysymb[:,1], "o", mew = 1, mec = "w", mfc = "None")                          

                    if 0 :
                        ind2 = np.where(ind_strain_negative[1] == j)
    #                    print ind2
    #                    print len(ind2[0])
                        if len(ind2[0]) > 0 :
                            xysymb = list_xy_strain_negative[ind2[0],:]
                            if np.isscalar(xysymb[0]) : p.plot(xysymb[0], xysymb[1], "s", mew = 2, mec = "w", mfc = "None") 
                            else : p.plot(xysymb[:,0], xysymb[:,1], "s", mew = 2, mec = "w", mfc = "None")                          
    
                    if 0 :
                        ind2 = np.where(ind_strain_below_min_negative[1] == j)
    #                    print ind2
    #                    print len(ind2[0])
                        if len(ind2[0]) > 0 :
                            xysymb = list_xy_strain_below_min_negative[ind2[0],:]
                            if np.isscalar(xysymb[0]) : p.plot(xysymb[0], xysymb[1], "w_", mew = 1 )#, mec = "w", mfc = "None") 
                            else : p.plot(xysymb[:,0], xysymb[:,1], "w_", mew = 1 ) # , mec = "w", mfc = "None")                          

            if not same_scale_for_all_components  :
                  
                cbar = fig1.colorbar(imrgb, extend=extend, 
#                                     shrink = 0.5, 
                                     ax = ax, 
                                    use_gridspec = True)

                label1 =   list_of_labels[j] + " " + filter_label + " " + units_label                  
                cbar.set_label(label1) 
                
        if same_scale_for_all_components :     
            fig1.subplots_adjust(right=0.8)
            cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])
#            cbar_ax = fig1.add_axes([0.85, 0.15, 0.3, 0.3])
            cbar = fig1.colorbar(imrgb, extend = extend, cax=cbar_ax)      
            cbar.ax.tick_params(labelsize=20)
            label1 = maptype + " " + filter_label + " " + units_label
            cbar.set_label(label1, fontsize = 20)
            
        if 0:
            if ngraph > 1 :
                fig2 = p.figure()
            else :
                fig2 = fig1                    
            cbar = fig2.colorbar(imrgb, extend='both', shrink = 0.5)
            cbar.set_label(maptype)  
              
        if savefig :
            figfilename = fileprefix + "_" + maptype + ".png"
            print(figfilename)
            p.savefig(figfilename, bbox_inches='tight')                
                    
        return(0)    
    
########                          
#    histo = np.histogram(dxylist_in[:,0], bins = 600)
##    print "histogram data : ", histo[0]
##    print "bin edges :",  histo[1]  
##    print shape(histo[0])
##    print shape(histo[1])
#    p.figure()
#    barwidth = histo[1][1]-histo[1][0]
##    print "bar width = ", barwidth
#    p.bar(histo[1][:-1], histo[0], width = barwidth)
#    p.xlabel("dx")
#    p.ylabel("number of pairs")
#    
#    histo = np.histogram(dxylist_in[:,1], bins = 600)
##    print "histogram data : ", histo[0]
##    print "bin edges :",  histo[1]  
##    print shape(histo[0])
##    print shape(histo[1])
#    p.figure()
#    barwidth = histo[1][1]-histo[1][0]
##    print "bar width = ", barwidth
#    p.bar(histo[1][:-1], histo[0], width = barwidth)
#    p.xlabel("dy")
#    p.ylabel("number of pairs")  
#    klmfsd

#def add_ijref_columns_to_filexyz(filexyz,
#                    mosaic_order = "3412") :  #10Feb17
#                    
#        # remplace calc_map_imgnum de MG
#
#        # hypothese grille rectangulaire
#        # mosaic order = "1234"  signifie
#        # img1 x0,y0 img2 x0+dx,y0
#        # img3 x0, y0-dy img4 x0+dx, y0-dy
#        # avec dx dy positif     
#        # début carto = coin en haut à gauche
#        
#        # setup location of images in map based on xech yech + map pixel size
#        # permet pixels rectangulaires
#        # permet cartos incompletes
#
#
#
#        return(map_imgnum, dxystep, pixsize, impos_start)

def plot_spot_traj(img_list,
                         Intensity_list,
                         threshold_factor_for_traj,
                         xyfit_list,
                         xboxsize = None,
                         yboxsize = None,
                         xpic = None,
                         ypic = None,
                         titre = None,
                         overlay = 0
                         ):
                             
        p.rcParams["figure.subplot.right"] = 0.85            
        p.rcParams["figure.subplot.left"] = 0.15
        p.rcParams['font.size']=20 
        
        print("plot spot trajectory")
        print("filtering :")
        
        Ithreshold = threshold_factor_for_traj*max(Intensity_list)
        print("intensity filter :")
        print("exclude images with low spot intensity < Ithreshold =  ", Ithreshold)
        print("Ithreshold = threshold_factor_for_traj*max(Intensity_list)")
        print("threshold_factor_for_traj =", threshold_factor_for_traj)
        
        cond_intensity = (Intensity_list > Ithreshold)
        
        filter_string = 'filtering : exclude I < ' + str(Ithreshold)

        filter_on_xypos = 0        
        
        if (xpic  is  not None) and (ypic  is  not None) and (xboxsize  is  not None) and (yboxsize  is  not None) :
            filter_on_xypos = 1
            print("xy position filter :")
            print("exclude images with strange values of spot xy position")
            print("(spot outside search box)") 
            xypic = np.array([xpic, ypic], float)
            dxy_list = abs(xyfit_list - xypic)
            cond_xinbox = (dxy_list[:,0] < xboxsize)
            cond_yinbox = (dxy_list[:,1] < yboxsize)
            cond_total = cond_intensity * cond_xinbox * cond_yinbox
            
            filter_string = filter_string + " + check if spot inside box"
            
        else :
            cond_total = cond_intensity
        
        ind_filter = np.where(cond_total > 0)#        print "ind_filter =", ind_filter 
        xyfit_list_in = xyfit_list[ind_filter]
#        xymax_list_high_int2 = xymax_list[ind_filter]
        Intensity_list_in = Intensity_list[ind_filter]
#        print "xyfit_list_in = \n",xyfit_list_in.round(decimals=2)
#        print "Intensity_list_in = ", Intensity_list_in
#        xymoy2 = np.zeros(2,float)
#
#        for i in range(2):
#            xymoy2[i]=np.average(xyfit_list_in[:,i],weights=Intensity_list_in)
        print("**********************************")
        print("statistics on spot position (after filtering)")
        print("number of points : before / after filtering", len(Intensity_list), len(ind_filter[0])) 
        print("mean std range")        
        xymean = xyfit_list_in.mean(axis = 0)
        xystd = xyfit_list_in.std(axis = 0)
        xyrange = (xyfit_list_in.max(axis = 0)-xyfit_list_in.min(axis = 0))
        print(xymean.round(decimals=2), end=' ')
        print(xystd.round(decimals=2), end=' ')
        print(xyrange.round(decimals=2))
        print("mean spot position with intensity weighting :")
        xymean_weighted = np.average(xyfit_list_in, axis=0, weights=Intensity_list_in)
        print(xymean_weighted.round(decimals=2))
#        print "xymoy2 =", xymoy2.round(decimals=2)
    
#        n_high_int2 = len(ind_filter[0])
        if titre  is  not None :
            titre2 = titre + "\n"+ filter_string
            
            #titre2 = titre2 + "\n"+ 'img ' + str(img_list[ind_filter[0][0]])\
            #        + ' to '+ str(img_list[ind_filter[0][n_high_int2-1]])

        if not overlay :
            p.figure()
            color1 = 'bo-'
        else :
            color1 = "rs-"
        #p.plot(xymax_list_high_int[:,0],xymax_list_high_int[:,1],'ro-',label = 'xy max')
        p.plot(xyfit_list_in[:,0],-xyfit_list_in[:,1],color1)
        #p.text(min(xyfit_list_in[:,0]),min(xyfit_list_in[:,1]),titre2)
        p.xlabel('xpix')
        p.ylabel('ypix')
        for i in ind_filter[0]:
            x =xyfit_list[i,0]
            y =-xyfit_list[i,1]
            p.text(x,y,str(img_list[i]), fontsize = 16)
            
        if not overlay : 
            if titre  is  not None : p.title(titre2)       
        #p.axvline(x=xymoy2[0])
        #p.axhline(y=xymoy2[1])
        if filter_on_xypos :
           p.axvline(x = xpic-xboxsize,color ="r")
           p.axvline(x = xpic+xboxsize,color ="r")
           p.axhline(y = -ypic-yboxsize,color ="r")
           p.axhline(y = -ypic+yboxsize,color ="r")
           
        ax = p.subplot(111)
        ax.axis("equal")
           
        return(0)
        
def add_i_j_columns_to_filexyz(filexyz,
                               filepath = os.curdir + os.sep,
                    grid_order = "3412") :  #10Feb17
                    
        # remplace calc_map_imgnum de MG

        # hypothese grille rectangulaire
        # mosaic order = "1234"  signifie
        # img1 x0,y0 img2 x0+dx,y0
        # img3 x0, y0-dy img4 x0+dx, y0-dy
        # avec dx dy positif     
        # début carto = coin en haut à gauche
        
        # setup location of images in map based on xech yech + map pixel size
        # permet pixels rectangulaires
        # permet cartos incompletes
          
    data_xyz = np.loadtxt(filexyz, skiprows = 1)
    nimg_xyz = np.shape(data_xyz)[0]
    
    print("nimg_xyz = ", nimg_xyz)
 
    imglist = np.array(data_xyz[:,0]+0.0001,int)
        
    xylist =  data_xyz[:,1:3]- data_xyz[0,1:3]
    
    dxylist =  np.diff(xylist, axis = 0)
    dxylist_abs =  abs(dxylist)
    print("dxylist : \n", dxylist)
    print("dxylist_abs : \n", dxylist_abs)
    print("dxylist_abs[:,0] : \n", dxylist_abs[:,0])
    print(np.shape(xylist), np.shape(dxylist))
    print(dxylist_abs.max(axis=0))
    print(dxylist_abs.argmax(axis=0))
    print(dxylist_abs.min(axis=0))
    print(dxylist_abs.argmin(axis=0))
    
    # tri par ordre croissant
    index1 = np.argsort(dxylist_abs[:,0])
    print("index1 : \n", index1)
    print("dxylist_abs[index1,0] :\n", dxylist_abs[index1,0])
    index2 = np.argsort(dxylist_abs[:,1])
    print("index2 : \n", index2)
    print("dxylist_abs[index2,1] :\n", dxylist_abs[index2,1])
    
#    for i in range(npics):
#        index2[i]=index1[npics-i-1]
#    #print "index2 =", index2
#    data_str2 = data_str[index2] 

    signdxy_fromlist = np.zeros(2,int)
        
    if dxylist_abs[index1[0],0] < 1e-5 : # y fast
        xyfast_fromlist = 1
        xyslow_fromlist = 0
        print("yfast")
        print("dxylist[index1[-1],0] =", dxylist[index1[-1],0])
        print("dxylist[index2[0],1] = ", dxylist[index2[0],1])
        signdxy_fromlist[0] = np.sign(dxylist[index1[-1],0])
        signdxy_fromlist[1] = np.sign(dxylist[index2[0],1])

        
    elif dxylist_abs[index2[0],1] < 1e-5 : # x fast
        xyfast_fromlist = 0
        xyslow_fromlist = 1
        print("xfast")
        print("dxylist[index1[0],0] = ", dxylist[index1[0],0])
        print("dxylist[index2[-1],1] = ", dxylist[index2[-1],1])
        signdxy_fromlist[0] = np.sign(dxylist[index1[0],0])
        signdxy_fromlist[1] = np.sign(dxylist[index2[-1],1])
        
    print("signdxy_fromlist = ", signdxy_fromlist)
        
    print("xyfast_fromlist, xyslow_fromlist = ", xyfast_fromlist, xyslow_fromlist)
        
    xylist_abs = abs(xylist)
    print(xylist_abs.max(axis=0))
    print(xylist_abs.argmax(axis=0))
    
    xylist_abs_max = xylist_abs.max(axis=0)
    
    dxylist_abs_max = dxylist_abs.max(axis=0)
    dxylist_abs_min = dxylist_abs.min(axis=0)  
        
    # xfast yfast dxsign dysign i,j of impos_start
    dict_grid_order = {"1234" : [[1,0,1,-1],[0,0]],
                        "1324" : [[0,1,1,-1],[0,0]],
                        "3412" : [[1,0,1,1],[1,0]],
                        "2413" : [[0,1,1,1],[1,0]],
                        "2143" : [[1,0,-1,-1],[0,1]],
                        "3142" : [[0,1,-1,-1],[0,1]],
                        "4321" : [[1,0,-1,1],[1,1]],
                        "4231" : [[0,1,-1,1],[1,1]]                   
                    } 
 
    print("grid order = ", grid_order)
    print("xfast yfast dxsign dysign = ", dict_grid_order[grid_order][0])
    print("i,j of impos_start = ",  dict_grid_order[grid_order][1] , "*[nptsy, nptsx]")
    
    dxystep = np.zeros(2,float)
    xyfastslow = dict_grid_order[grid_order][0][:2]
    xyfast = np.argmax(xyfastslow)
    xyslow = np.argmin(xyfastslow)
    
    print("xyfast, xyslow = ", xyfast, xyslow)
    
    if xyfast != xyfast_fromlist :
        print("xyfast from grid_order contradicts xyfast from xylist")
        jklsd
    
    signdxy = dict_grid_order[grid_order][0][2:]
    
    if signdxy_fromlist[1] != signdxy[1] :
        print("sign of dy from grid_order contradicts sign of dy from list")
        jlqsd
        
    if signdxy_fromlist[0] != signdxy[0] :
        print("sign of dx from grid_order contradicts sign of dy from list")
        klmqds
    
    dxystep[xyslow] = dxylist_abs_max[xyslow]*signdxy[xyslow]
    dxystep[xyfast] = dxylist_abs_min[xyfast]*signdxy[xyfast]
    
    print("dxystep =", dxystep)
               
    nptsxy = np.zeros(2,int)
    
    toto = xylist_abs_max[xyfast]/dxylist_abs_min[xyfast]    
    nptsxy[xyfast] = int(toto+1e-5)+1
    toto = xylist_abs_max[xyslow]/dxylist_abs_max[xyslow]   
    nptsxy[xyslow] = int(toto+1e-5)+1    
 
    print("nptsxy = ", nptsxy)
    
    abs_step = abs(dxystep)
    print("dxy step abs ", abs_step)
    largestep = max(abs_step)
    smallstep = min(abs_step)
    pix_r = largestep / smallstep
    if pix_r != 1.0 :
            print("|dx| and |dy| steps not equal : will use rectangular pixels in map")
            print("aspect ratio : ", pix_r)
    else :
            print("equal |dx| and |dy| steps")
            pix1 = pix2 = 1
            
    if float(int(round(pix_r,1))) < (pix_r -0.01) :
            print("non integer aspect ratio")
            for nmult in (2,3,4,5):
                    toto = float(nmult)*pix_r
                    #print toto
                    #print int(round(toto,1))
                    if abs(float(int(round(toto,1))) - toto)< 0.01 :
                            #print nmult
                            break
            pix1 = nmult
            pix2 = int(round(float(nmult)*pix_r,1))
            #print "map pixel size will be ", pix1, pix2
    else :
            pix1 = 1
            pix2 = int(round(pix_r,1))

    print("pixel size for map (pix1= small, pix2= large):" , pix1, pix2)

    large_axis = np.argmax(abs_step)
    small_axis = np.argmin(abs_step)

    #print large_axis, small_axis
    if large_axis == 1 :
            pixsize = np.array([pix1, pix2], dtype=int)
    else :
            pixsize = np.array([pix2, pix1], dtype = int)
    print("pixel size for map dx dy" , pixsize)  
    
    map_imgnum = np.zeros((nptsxy[1],nptsxy[0]),int)

    print("map raw size ", np.shape(map_imgnum))
    
    toto = np.array(dict_grid_order[grid_order][1])
    
    toto1 = np.array([nptsxy[1],nptsxy[0]])
    impos_start = np.multiply(toto,toto1-1)   
    if (toto[0] == 0) :
        startx = "left"
    else :
        startx = "right"          
    if (toto[1]==0) :
        starty = "upper"
    else :
        starty = "lower"
    startcorner = starty + " " + startx + " corner" 
    print("map starts in : ", startcorner)
    print("impos_start = ", impos_start) 
    
    impos = np.zeros(2,float) # y x   # 22Jan14 changed from int to float

    # tableau normal : y augmente vers le bas, x vers la droite
    # xech yech : xech augmente vers la droite, yech augmente vers le hut

    # tester orientation avec niveaux gris = imgnum
#        print type(xylist[0,0])
#        print type(dxystep[0])

    ijpos = np.zeros((nimg_xyz,2),int)
    for i in range(nimg_xyz) :
    #for i in range(200) :
            imnum = int(round(imglist[i],0))
            impos[1] = xylist[i,0]/ abs(dxystep[0])
            impos[0] = -xylist[i,1]/ abs(dxystep[1]) 
#                print type(impos[0])
#            print "impos = ", impos
            impos = impos_start+impos
#            impos[1]= mod(impos1[1],nptsxy[1])
#            impos[0]= mod(impos1[0],nptsxy[0])
#            print "impos_start = ", impos_start
#            print "i, imnum, impos = ", i, imnum, impos
            impos1 = np.array((impos+0.01).round(decimals=0), dtype = int)
#            print i, imnum, impos1
            #print impos
            map_imgnum[impos1[0], impos1[1]] = imnum
            
            ijpos[i] = impos1
            
#                if i == 1000 : jklds
    print("map_imgnum")
    print(map_imgnum)
    
#    toto = np.column_stack((data_xyz,ijpos))
#    
#    print np.shape(toto)
    
    nlines = np.shape(map_imgnum)[0]
    ncol = np.shape(map_imgnum)[1]
        
    header = '#File created at %s with read_EBSD_quartz_Jul14.py \n' % (asctime())
    header += "#grid_order : " + grid_order + "\n"
    header += "#dxystep : " + str(dxystep) + "\n"
    header += "#pixsize : " + str(pixsize) + "\n"
    header += "#impos_start : " + str(impos_start) + "\n"
    header += "#nlines : " + str(nlines) + "\n"
    header += "#ncol : " + str(ncol) + "\n"
    header += "#img    X    Y  iref  jref \n" 
    outfilename = filepath + "filexyz_ij2.dat"
    print("xy i j grid file = ", outfilename)
    outputfile = open(outfilename,'w')
    outputfile.write(header)   
    for i in range(nimg_xyz):
        line1 = str(imglist[i]) + "  " + str(xylist[i,0]) + "  " + str(xylist[i,1]) + "  " + str(ijpos[i,0]) + "  " + str(ijpos[i,1]) +"\n" 
        outputfile.write(line1)    
#    np.savetxt(outputfile, toto, fmt='%.2f')
    outputfile.close()
    
    return(filexyz)

def read_filexyz(filexyz) :
    
    print("enter read_filexyz") #, asctime()
    
    f = open(filexyz, 'r')
    i = 0
    try:
        for line in f:
            if line.startswith( "#grid_order") :
                toto = line.rstrip(MG.PAR.cr_string).split(":")
                grid_order = str(toto[1])
                print("grid_order = ", grid_order)
            if line.startswith( "#dxystep") :
                toto = line.rstrip(MG.PAR.cr_string).replace('[', '').replace(']', '').split(":")
                toto1 = toto[1].split()
                dxystep = np.array(toto1, float)
                print("dxystep= ", dxystep)
            if line.startswith( "#nlines") :
                toto = line.rstrip(MG.PAR.cr_string).split(":")
                nlines = int(toto[1])
                print("nlines= ", nlines)     
            if line.startswith( "#ncol") :
                toto = line.rstrip(MG.PAR.cr_string).split(":")
                ncol = int(toto[1])
                print("ncol= ", ncol)
            i = i+1
    finally:
        f.close()
    imgxyz = np.loadtxt(filexyz)
    
#    print "exit read_filexyz", asctime()
    
    return(imgxyz,grid_order,dxystep,nlines,ncol)

def calc_indlinkimg(img_list,img_in_filexyz):
     
    print("start calculation of ind_link_img list")
    nimg1 = len(img_list)
    nimg_xyz = len(img_in_filexyz)  
    nimg_min = min(nimg1, nimg_xyz)
    ind_link_img = np.ones(nimg1,int)*(-1)
    dimg = img_list[:nimg_min] - img_in_filexyz[:nimg_min]
    img_offset = 0
    for i in range(nimg_min) :
        if dimg[i] < 1e-3 : 
            ind_link_img[i] = i
        else :
            ind_link_img[i] = i + dimg[i] + img_offset
            dimg2 = img_list[i] - img_in_filexyz[ind_link_img[i]]
#                    print i,  ind_link_img[i], dimg[i]                        
#                    print  i, col_obj.img[i], img_in_filexyz[i], dimg[i], ind_link_img[i],\
#                         img_in_filexyz[ind_link_img[i]], dimg2
            if abs(dimg2) > 1e-5 :
#                    print "jump in img num in filexyz => add offset"
                ind_link_img2 = np.where(img_in_filexyz == img_list[i])
                img_offset += ind_link_img2[0] - ind_link_img[i] 
#                    print "img_offset = ", img_offset
                ind_link_img[i] = ind_link_img2[0]
                dimg2 = img_list[i] - img_in_filexyz[ind_link_img[i]]
#                    print  i, col_obj.img[i], img_in_filexyz[i], dimg[i], ind_link_img[i],\
#                         img_in_filexyz[ind_link_img[i]], dimg2    
                if abs(dimg2) > 1e-5 :
                    kjlqd
                    
    return(ind_link_img)
    
def serial_two_spots_to_mat(filepathout, 
                                   fileprefix, 
                                   filespotmon1,
                                   filespotmon2,
                                   hkl_2spots, 
                                   calib, 
                                    test = 0,
                                    add_str = "",
                                    pixelsize = DictLT.dict_CCD[MG.PAR.CCDlabel][1],
                                    threshold_on_Imax_ratio = 0.002,
                                    imgref_for_superimposing_spot_trajectories = None,
                                    omega = None, # was PAR.omega_sample_frame,
                                    mat_from_lab_to_sample_frame = MG.PAR.mat_from_lab_to_sample_frame,
                                    elem_label = None,
                                    point_num_for_superimposing_spot_trajectories = None,
                                    add_last_N_columns_from_filespotmon = 0,
                                    verbose = 1,
                                    use_xy_max_or_xyfit = "xyfit"
                                    ):
    
    # matrices non deformee a partir des positions de 2 pics de HKL connus
    # ou matrice deformee a partir des positions de 4 pics de HKL connus
    # xy des 2 ou 4 spots a partir d'un fichier liste : img x0 y0 x1 y1 x2 y2 x3 y3 

    data_list1, listname, nameline0 = MG.read_summary_file(filespotmon1)  
    data_list2, listname, nameline0 = MG.read_summary_file(filespotmon2)  
    
    data_list1 = np.array(data_list1, dtype=float)
    data_list2 = np.array(data_list2, dtype=float)    

    nimg = np.shape(data_list1)[0]
    print(nimg)
    ndata_cols = np.shape(data_list1)[1]
    print(ndata_cols)
                       
    indimg = listname.index("img") 
    img_list1 = np.array(data_list1[:,indimg],int)
    img_list2 = np.array(data_list2[:,indimg],int)
    print("img_list1 = ", img_list1)
    
    point_list1 = np.arange(len(img_list1))
    point_list2 = np.arange(len(img_list2))
    
    if use_xy_max_or_xyfit == "xyfit" :
        indxfit = listname.index("xfit")
    elif use_xy_max_or_xyfit == "xymax" :
        indxfit = listname.index("xmax")
        
    indxyfit = np.array([indxfit, indxfit+1])
    xyfit_list1 = data_list1[:,indxyfit]
    xyfit_list2 = data_list2[:,indxyfit]
    
    indImax = listname.index("Imax") 
    Imax_list1 = data_list1[:,indImax] 
    Imax_list2 = data_list2[:,indImax]
    if "Seconds" in listname :
        indSeconds =  listname.index("Seconds")
        Seconds_list = data_list1[:,indSeconds]
        Imax_list1 = Imax_list1/Seconds_list
        Imax_list2 = Imax_list2/Seconds_list       
    
    Imax_ratio_list1 = Imax_list1 / max(Imax_list1)
    Imax_ratio_list2 = Imax_list2 / max(Imax_list2)
    
    if 1 :
        fig, ax1 = p.subplots()
        ax1.plot(point_list1,Imax_ratio_list1, "ro-" )
        ax1.plot(point_list2,Imax_ratio_list2, "bs-" )
        ax1.set_yscale('log')
        ax1.set_xlabel('npts')
        ax1.set_ylabel("(Imax/Seconds) / max(Imax/Seconds) ")
        ax1.axhline(y = threshold_on_Imax_ratio)
    
    if imgref_for_superimposing_spot_trajectories  is  not None :
        xpic, ypic = 0, 0
        xboxsize, yboxsize = 100, 100
        ind1 = np.where(img_list1 == imgref_for_superimposing_spot_trajectories)
        print(ind1[0])
        xy1 = xyfit_list1-xyfit_list1[ind1[0],:]
        ind2 = np.where(img_list2 == imgref_for_superimposing_spot_trajectories)        
        xy2 = xyfit_list2-xyfit_list2[ind2[0],:]
        
    if point_num_for_superimposing_spot_trajectories  is  not None :
        ind1 = np.where(point_list1 == point_num_for_superimposing_spot_trajectories)
        print(ind1[0])
        xy1 = xyfit_list1-xyfit_list1[ind1[0],:]
        ind2 = np.where(point_list2 == point_num_for_superimposing_spot_trajectories)        
        xy2 = xyfit_list2-xyfit_list2[ind2[0],:]
            
    xy_2spots = np.column_stack((xyfit_list1,xyfit_list2))
    Imax_ratio_2spots = np.column_stack((Imax_ratio_list1, Imax_ratio_list2))
    
    imgall = img_list1
    nimg = np.shape(xy_2spots)[0]
    
    ndec = 6
    ncol = 9
    fmt1 = "%.6f"  # doit coller avec ndec
    
    str_names_to_add = ""
    if add_last_N_columns_from_filespotmon > 0 :
        n1 = add_last_N_columns_from_filespotmon
        listnames_to_add = listname[-n1:]
        str_names_to_add = " ".join(listnames_to_add)
        
    print("unstrained matrix from two spots + delta_alf epx-theor")
    header = "img 0, xy1 1:3, xy2 3:5, matstarlab 5:14 delta_alf 14 " + str_names_to_add + "\n"
    header2 = "img"
    for i in range(2):
        label2 = " xy"+str(i+1)+"_0 xy"+str(i+1)+"_1"
        header2 = header2 + label2
    for i in range(ncol) :
        label2 = " matstarlab_" + str(i) 
        header2 = header2 + label2        
    header2 = header2 + " dalf " + str_names_to_add + "\n"  
    outfilename = filepathout + "mat_2spots_" + fileprefix + "img" +\
        str(imgall[0]) + "to" + str(imgall[-1]) + "_alf" + add_str + ".dat"    
    print(outfilename)
    print(header)
    print(header2)

    #nimg = 50
    
    hkl = hkl_2spots
    print(hkl)
    
    Imax_ratio_min = Imax_ratio_2spots.min(axis=1)
    if 0 :
        print("Imax_ratio_2spots =", Imax_ratio_2spots)
        print("Imax_ratio_min = ", Imax_ratio_min)
    ind0 = np.where( Imax_ratio_min > threshold_on_Imax_ratio )  
    print(np.shape(ind0))
    
    nimg2 = len(ind0[0])
    
    range1 = ind0[0]
    
    print("found %d images with Imax of 2 spots larger than %.4f" %(nimg2, threshold_on_Imax_ratio))

    if (imgref_for_superimposing_spot_trajectories  is  not None) or (point_num_for_superimposing_spot_trajectories  is  not None) : 
        p.figure()
        xy1_short = xy1[range1,:]
        xy2_short = xy2[range1,:]
        img_list1_short = img_list1[range1]
        img_list2_short = img_list2[range1]
        p.plot(xy1_short[:,0], -xy1_short[:,1], "ro-")
        for i in range(nimg2) : p.text(xy1_short[i,0],-xy1_short[i,1],str(img_list1_short[i]), fontsize = 16)
        p.plot(xy2_short[:,0], -xy2_short[:,1], "bs-")
        for i in range(nimg2) : p.text(xy2_short[i,0],-xy2_short[i,1],str(img_list2_short[i]), fontsize = 16)
        p.xlabel("xpix")
        p.ylabel("ypix")


    if add_last_N_columns_from_filespotmon > 0 :
        n1 = add_last_N_columns_from_filespotmon
        columns_to_add = data_list1[ind0[0],-n1:]
     
    matstarlab_all = np.zeros((nimg2,ncol), float)
    dalf_all = np.zeros(nimg2, float)
    
    print("spot nimg xymean xystd xyrange")    
    for i in range(2):           
        print(i, np.shape(ind0)[1], xy_2spots[ind0[0],2*i:2*i+2].mean(axis=0).round(decimals=2),\
            xy_2spots[ind0[0],2*i:2*i+2].std(axis=0).round(decimals=2), \
            xy_2spots[ind0[0],2*i:2*i+2].max(axis=0)-xy_2spots[ind0[0],2*i:2*i+2].min(axis=0))   

# point d'arret standard
    if test == 1 : jlfsdfs
          
    k = 0
    
    iscubic = MG.test_if_cubic(elem_label)
    for i in ind0[0]:
        #print "k = ", k
        xycam = xy_2spots[i,:].reshape(2,2)        
#        print xycam

        print("img = ", imgall[i])
        if iscubic :  # simplification pour le cas cubique
             matstarlab_all[k,:], dalf_all[k] = MG.two_spots_to_mat(hkl,
                                                    xycam,
                                                    calib,
                                                    pixelsize = pixelsize,
                                                    verbose = verbose)           
        else :
            matstarlab_all[k,:], dalf_all[k] = MG.two_spots_to_mat_gen(hkl,
                                                    xycam,
                                                    calib,
                                                    pixelsize = pixelsize,
                                                    elem_label = elem_label,
                                                    verbose = verbose)
        
        if 0 :                                            
            mat1 = MG.matstarlab_to_matstarsample3x3(matstarlab_all[k,:],
                                              omega = omega,
                                              mat_from_lab_to_sample_frame = mat_from_lab_to_sample_frame)

            mat2 = mat1.transpose()  # OK seulement pour matrice OND
            print("xyzsample on abcstar :")
            print(mat2.round(decimals = 4))
            print("columns normalized to largest component :")
            str1 = ["HKLx", "HKLy", "HKLz"]
            for i in range(3):
                print(str1[i], (mat2[:,i]/abs(mat2[:,i]).max()).round(decimals = 4))
                                                   
        k = k+1
     
#    print matstarlab_all.round(decimals=ndec)

    toto1 = []    
    
    toto1 = np.column_stack((imgall[ind0[0]].round(decimals=1), xy_2spots[ind0[0],:].round(decimals=2), matstarlab_all.round(decimals=ndec),dalf_all.round(decimals=3)))

#    print dalf_all.round(decimals=2)
    if add_last_N_columns_from_filespotmon > 0 :
        toto1 = np.column_stack((toto1,columns_to_add))
    
    print(np.shape(toto1))
    print(outfilename)
    outputfile = open(outfilename,'w')
    outputfile.write(header)
    outputfile.write(header2)
    np.savetxt(outputfile, toto1, fmt = fmt1)
    outputfile.close()

    return(outfilename)       
