# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:20:21 2013

@author: micha
"""
import os

import multigrainFS as MG
print('MG position',MG.__file__)
import param_multigrain as PAR

import wx
if wx.__version__ <'4.':
    WXPYTHON4 = False
else:
    WXPYTHON4 = True
    wx.OPEN = wx.FD_OPEN
    wx.CHANGE_DIR = wx.FD_CHANGE_DIR
    
    def sttip(argself, strtip):
        return wx.Window.SetToolTip(argself,wx.ToolTip(strtip))
    wx.Window.SetToolTipString = sttip
from wx.lib.agw.shapedbutton import SButton, SBitmapButton
    
list_txtparamPM = ['Map Summary File',
                 'File xyz',
                 'maptype',
                 'filetype',
                 'substract_mean',
                 'probed_grainindex',
                 'filter_on_pixdev_and_npeaks',
                 'maxpixdev_forfilter',
                 'minnpeaks_forfilter',
                 'min_forplot',
                 'max_forplot',
                 'pixdevmin_forplot',
                 'pixdevmax_forplot',
                 'npeaksmin_forplot',
                 'npeaksmax_forplot',
                 'zoom',
             'xylim',
             'filter_mean_strain_on_misorientation',
             'max_misorientation',  # only for filter_mean_strain_on_misorientation = 1
             'change_sign_xy_xz',
             'subtract_constant',
             'remove_ticklabels_titles',
             'col_for_simple_map',
             'low_npeaks_as_missing',
             'low_npeaks_as_red_in_npeaks_map',  # only for maptype = "fit"
             'low_pixdev_as_green_in_pixdev_map',  # only for maptype = "fit"
             'use_mrad_for_misorientation',  # only for maptype = "misorientation_angle"
             'color_for_duplicate_images',  # [0.,1.,0.]
             'color_for_missing',
             'high_pixdev_as_blue_and_red_in_pixdev_map',  # only for maptype = "fit"
             'filter_on_intensity',
             'min_intensity_forfilter',  # only for filter_on_intensity = 1
             'color_for_max_strain_positive',  # red  # [1.0,1.0,0.0]  # yellow
             'color_for_max_strain_negative',  # blue
             'plot_grid',
             'map_rotation'
                 ]


import numpy as np
list_txtparamPM_type_dict = {'Map Summary File':str,
                 'File xyz':str,
                 'maptype':str,
                 'filetype':str,
                 'substract_mean':str,
                 'probed_grainindex':int,
                 'filter_on_pixdev_and_npeaks':float,
                 'maxpixdev_forfilter':float,
                 'minnpeaks_forfilter':float,
                 'min_forplot':float,
                 'max_forplot':float,
                 'pixdevmin_forplot':float,
                 'pixdevmax_forplot':float,
                 'npeaksmin_forplot':float,
                 'npeaksmax_forplot':float,
                 'zoom':str,
             'xylim':str,
             'filter_mean_strain_on_misorientation':float,
             'max_misorientation':float,  # only for filter_mean_strain_on_misorientation = 1
             'change_sign_xy_xz':int,
             'subtract_constant':str,
             'remove_ticklabels_titles':str,
             'col_for_simple_map':str,
             'low_npeaks_as_missing':str,
             'low_npeaks_as_red_in_npeaks_map':str,  # only for maptype = "fit"
             'low_pixdev_as_green_in_pixdev_map':str,  # only for maptype = "fit"
             'use_mrad_for_misorientation':str,  # only for maptype = "misorientation_angle"
             'color_for_duplicate_images':str,  # [0.:,1.:,0.]
             'color_for_missing':str,
             'high_pixdev_as_blue_and_red_in_pixdev_map':str,  # only for maptype = "fit"
             'filter_on_intensity':int,
             'min_intensity_forfilter':float,  # only for filter_on_intensity = 1
             'color_for_max_strain_positive':str,  # red  # [1.0:,1.0:,0.0]  # yellow
             'color_for_max_strain_negative':str,  # blue
             'plot_grid':int,
             'map_rotation':int
                            }


class MainFrame_plotmaps (wx.Frame):
    def __init__(self, parent, id, title, dict_params):
        wx.Frame.__init__(self, parent, id, title, wx.DefaultPosition, wx.Size(1000, 1280))

#         print "__init__ of MainFrame_plotmaps"
        self.dict_params = dict_params

        self.list_of_windows = []

        print((self.dict_params))

        self.panel = wx.Panel(self)

        nb_of_params = len(list_txtparamPM) - 2
        
        if WXPYTHON4:
            grid0 = wx.FlexGridSizer(4, 10, 10)
        else:
            grid0 = wx.FlexGridSizer(7, 4)

        grid0.SetFlexibleDirection(wx.HORIZONTAL)

        txt_fileparameters = wx.StaticText(self.panel, -1, "File parameters ")
        font = wx.Font(18, wx.MODERN, wx.ITALIC, wx.NORMAL)
        txt_fileparameters.SetFont(font)
        txt_none = wx.StaticText(self.panel, -1, " ")
        txt_none2 = wx.StaticText(self.panel, -1, " ")
        txt_none3 = wx.StaticText(self.panel, -1, " ")

        grid0.Add(txt_fileparameters)
        grid0.Add(txt_none)
        grid0.Add(txt_none2)
        grid0.Add(txt_none3)


        #          txt_none2= wx.StaticText(self.panel, -1, " ")
        #          grid.Add(txt_none2)
        #          txt_none3= wx.StaticText(self.panel, -1, " ")
        #          grid.Add(txt_none3)


        #          Affichage summary file et filexyzfile par défaut

        txt_summary = wx.StaticText(self.panel, -1, "Map SUMMARY .dat File")

        fullpath_summaryfile = self.dict_params['Map Summary File']
        folderpath, summaryfile = os.path.split(fullpath_summaryfile)

        self.summarypath_folder = wx.TextCtrl(self.panel, -1,
                                       folderpath,
                                       size=(250, -1))

        self.summarypath_file = wx.TextCtrl(self.panel, -1,
                                       summaryfile,
                                       size=(250, -1))

        btnbrowse = wx.Button(self.panel, -1, "Browse")
        btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_filesum)

        grid0.Add(txt_summary)
        grid0.Add(self.summarypath_folder)
        grid0.Add(self.summarypath_file)
        grid0.Add(btnbrowse)

        txt_filexyz = wx.StaticText(self.panel, -1, "Map SUMMARY .dat File XYZ")
        txt_filexyz.SetToolTipString("A file containing 3 columns imageindex, x, y")


        fullpath_filexyz = self.dict_params['File xyz']
        folderpathxyz, summaryfilexyz = os.path.split(fullpath_filexyz)

        self.filexyzpath_folder = wx.TextCtrl(self.panel, -1,
                                       folderpathxyz,
                                       size=(250, -1))

        self.filexyzpath_file = wx.TextCtrl(self.panel, -1,
                                       summaryfilexyz,
                                       size=(250, -1))

        btnbrowse = wx.Button(self.panel, -1, "Browse")
        btnbrowse.Bind(wx.EVT_BUTTON, self.OnbtnBrowse_filexyz)

        grid0.Add(txt_filexyz)
        grid0.Add(self.filexyzpath_folder)
        grid0.Add(self.filexyzpath_file)
        grid0.Add(btnbrowse)

        txt_mapparameters = wx.StaticText(self.panel, -1, "Map parameters ")
        txt_mapparameters.SetFont(font)
        txt_none4 = wx.StaticText(self.panel, -1, " ")
        txt_none5 = wx.StaticText(self.panel, -1, " ")
        txt_none6 = wx.StaticText(self.panel, -1, " ")
        grid0.Add(txt_mapparameters)
        grid0.Add(txt_none4)
        grid0.Add(txt_none5)
        grid0.Add(txt_none6)

        #          txt_none7= wx.StaticText(self.panel, -1, " ")
        #          grid.Add(txt_none7)
        #          txt_none8= wx.StaticText(self.panel, -1, " ")
        #          grid.Add(txt_none8)

        txt_maptype = wx.StaticText(self.panel, -1, "Maptype: ")

        self.choice_maptype = wx.Choice(self.panel, -1, choices=['fit',
                                                               'strain6_crystal',
                                                               'rgb_x_sample',
                                                               'strain6_sample',
                                                               'euler3',
                                                               'stress6_crystal',
                                                               'stress6_sample',
                                                               'res_shear_stress',
                                                               'max_rss',
                                                               'von_mises',
                                                               'w_mrad'
                                                               ])

        self.choice_maptype.Bind(wx.EVT_CHOICE, self.Onchoice_maptype)

        self.btnclearwindows = wx.Button(self.panel, -1, "Clear Windows")
        self.btnclearwindows.Bind(wx.EVT_BUTTON, self.OnClearChildWindows)

        grid0.Add(txt_maptype)
        grid0.Add(self.choice_maptype)
        grid0.Add(self.btnclearwindows)

        #          Affichage des parametres selectionnés ou par défaut
        
        if WXPYTHON4:
            grid = wx.FlexGridSizer(4, 10, 10)
        else:
            grid = wx.FlexGridSizer(nb_of_params / 2, 4)

        grid.SetFlexibleDirection(wx.HORIZONTAL)

        self.list_txtctrl = []

        for kk, key in enumerate(list_txtparamPM):
            if kk >= 2:
                grid.Add(wx.StaticText(self.panel, -1, key))
                self.txtctrl = wx.TextCtrl(self.panel, -1, '', size=(150, 25))  # , style=wx.TE_READONLY,)
                self.txtctrl.SetValue(str(self.dict_params[key]))
                self.list_txtctrl.append(self.txtctrl)
                grid.Add(self.txtctrl)
            else:
                self.list_txtctrl.append('')

        btnPlot = SButton(self.panel, -1, "PLOT", size=(-1, 60))
        btnPlot.Bind(wx.EVT_BUTTON, self.OnPlot)
#         grid.Add(btnPlot, wx.EXPAND | wx.ALL)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(grid0, 1, wx.EXPAND, 0)
        vbox.Add(grid, 0, wx.EXPAND, 0)
        vbox.Add(btnPlot, 0, wx.EXPAND, 0)
        self.panel.SetSizer(vbox, wx.EXPAND)
        vbox.Fit(self)
        self.Layout()

#         self.panel.SetSizer(grid, wx.EXPAND)

    def OnClearChildWindows(self, evt):
        print("killing child windows!")
#         print "list_of_windows", self.list_of_windows
        for child in self.list_of_windows:
            try:
                child.Close()
            except:
                continue
        self.list_of_windows = []

    def OnReadParameters(self):
        self.list_txtctrl_new = []

        filesummary_path_folder = str(self.summarypath_folder.GetValue())
        filesummary_path_file = str(self.summarypath_file.GetValue())
        filesummary_path = os.path.join(filesummary_path_folder, filesummary_path_file)

        filexyz_path_folder = str(self.filexyzpath_folder.GetValue())
        filexyz_path_file = str(self.filexyzpath_file.GetValue())

        filexyz_path = os.path.join(filexyz_path_folder, filexyz_path_file)

        if filexyz_path_file in ('None',):
            filexyz_path = None

        self.dict_params['Map Summary File'] = filesummary_path
        self.dict_params['File xyz'] = filexyz_path

        dict_params = {}

        for kk, key in enumerate(list_txtparamPM):
            if kk >= 2:
                strvalue = str(self.list_txtctrl[kk].GetValue())
                
                if strvalue in ('maptype','probed_grainindex'):
                    print((strvalue, kk))

                print(("key, type, raw, kk", key, list_txtparamPM_type_dict[key], strvalue, kk))
                try:
                    convvalue = list_txtparamPM_type_dict[key](strvalue)
                except ValueError:
                    sentence = 'Something wrong for parameter "%s"\n. Should be of type: %s' % \
                                    (key, list_txtparamPM_type_dict[key])
                    wx.MessageBox(sentence, 'Error')
                    return None

                print(("converted value, type", convvalue, type(convvalue)))

                if convvalue in ('None',):
                    convvalue = None
                    print(("converted None", convvalue))

                self.dict_params[key] = convvalue

        return self.dict_params

    def convStrlist_to_array(self, liststring):
        res = []
        for elem in liststring[1:-1].split(','):
            res.append(float(elem))
        return np.array(res)

    def CheckValueParams(self):
        list_keys = ['color_for_max_strain_negative', 'color_for_max_strain_positive']
        for key in list_keys:
            val = self.dict_params[key]
            print(("val, type", val, type(val)))
            if isinstance(val, str):
                print(('val for key=%s converted to array' % key))
                cv_val = self.convStrlist_to_array(val)
                print(cv_val)
                self.dict_params[key] = cv_val

    def Onchoice_maptype (self, event):
        maptype = self.choice_maptype.GetStringSelection()
        self.dict_params['maptype'] = maptype
        self.list_txtctrl[2].SetValue(str(maptype))


    def OnbtnBrowse_filesum(self, event):
        folder = wx.FileDialog(self, "os.path.dirname(guest)")
        if folder.ShowModal() == wx.ID_OK:
            fold, filen = os.path.split(folder.GetPath())
            self.summarypath_folder.SetValue(fold)
            self.summarypath_file.SetValue(filen)

    def OnbtnBrowse_filexyz(self, event):
        folder = wx.FileDialog(self, "os.path.dirname(guest)")
        if folder.ShowModal() == wx.ID_OK:
            fold, filen = os.path.split(folder.GetPath())
            self.filexyzpath_folder.SetValue(fold)
            self.filexyzpath_file.SetValue(filen)

    def OnPlot(self, event):
        dictpars = self.OnReadParameters()
        print(("self.dict_params", self.dict_params))

        self.CheckValueParams()

        if dictpars is not None:
#             MG.plot_map_new(self.dict_params, App_parent=self)

            maptype = str(self.list_txtctrl[2].GetValue())
            grain_index = int(self.list_txtctrl[5].GetValue())
            MG.plot_map_new2(self.dict_params, maptype, grain_index, App_parent=self)



class Stock_parameters_PlotMaps:
    def __init__(self, list_txtparamPM, list_valueparamPM):
        self.list_txtparamPM = list_txtparamPM
        self.dict_params = list_valueparamPM

def fill_list_valueparamPM(initialparameters):
    """
    return a list of default value for buildsummary board from a dict initialparameters
    """
    list_valueparam_PM = [
                    initialparameters['Map Summary File'],
                     initialparameters['File xyz'],
                     initialparameters['maptype'],
                     initialparameters['filetype'],
                     initialparameters['substract_mean'],
                     initialparameters['numgrain'],
                     initialparameters['filter_on_pixdev_and_npeaks'],
                     initialparameters['maxpixdev_forfilter'],
                     initialparameters['minnpeaks_forfilter'],
                     initialparameters['min_forplot'],
                     initialparameters['max_forplot'],
                     initialparameters['pixdevmin_forplot'],
                     initialparameters['npeaksmin_forplot'],
                     initialparameters['npeaksmax_forplot']

                     ]
    return list_valueparam_PM

# list_txtparamPM = ['Map Summary File',
#                  'File xyz',
#                  'maptype',
#                  'filetype',
#                  'substract_mean',
#                  'numgrain',
#                  'filter_on_pixdev_and_npeaks',
#                  'maxpixdev_forfilter',
#                  'minnpeaks_forfilter',
#                  'min_forplot',
#                  'max_forplot',
#                  'pixdevmin_forplot',
#                  'pixdevmax_forplot',
#                  'npeaksmin_forplot',
#                  'npeaksmax_forplot'
#                  ]


initialparameters = {}

LaueToolsProjectFolder = os.path.dirname(os.path.abspath(os.curdir))

print(('LaueToolProjectFolder', LaueToolsProjectFolder))

MainFolder = os.path.join(LaueToolsProjectFolder, 'Examples', 'GeGaN')

print(("MainFolder", MainFolder))

initialparameters['IndexRefine PeakList Folder'] = os.path.join(MainFolder,
                                            'fitfiles')
initialparameters['Map Summary File'] = os.path.join(MainFolder, 'fitfiles',
                                           'nanox2_400__SUMMARY_0_to_5_add_columns.dat')
initialparameters['File xyz'] = os.path.join(MainFolder, 'fitfiles',
                                           'nanox2_400__xy_0_to_5.dat')
initialparameters['maptype'] = 'fit'
initialparameters['filetype'] = 'LT'

initialparameters['Map shape'] = (31, 41)  # (nb lines, nb images per line)

initialparameters['(stepX, stepY) microns'] = (1., 1.)
MainFolder = '/home/micha/LaueProjects/LeBaudy'
initialparameters['Map Summary File'] = os.path.join(MainFolder, 
                                           'mappos1__SUMMARY_0_to_1425_add_columns.dat')
initialparameters['File xyz'] = os.path.join(MainFolder, 'mappos1__xy_0_to_1425.dat')

initialparameters['stiffness file'] = os.path.join(MainFolder, 'si.stf')

initialparameters['nbdigits'] = 4
initialparameters['startingindex'] = 0
initialparameters['finalindex'] = 5
initialparameters['stepindex'] = 1
initialparameters['fast axis: x or y'] = 'x'

# MainFolder = '/home/micha/LaueProjects/LeBaudy'
# initialparameters['Map Summary File'] = os.path.join(MainFolder, 
#                                            'mappos1__SUMMARY_0_to_1425_add_columns.dat')
# initialparameters['File xyz'] = os.path.join(MainFolder, 'mappos1__xy_0_to_1425.dat')
# initialparameters['stiffness file'] = os.path.join(MainFolder, 'si.stf')

import numpy as np

list_valueparamPM = [initialparameters['Map Summary File'],
                     initialparameters['File xyz'],
                     "fit",
                     "LT",
                     "no",
                     0,
                     1,
                     20, 1, -0.2, 0.2, 0, 20, 6, 70,
                     'no', None, 0, 0.15, 0, None, 0,
                     None, None, None, None,
                     'no', None, None, None,
                     0, 20000,
                     [1., 0., 0.], [0., 0., 1.],
                    1,
                    0]


# dict_params = fill_list_valueparamPM(initialparameters)

def prepare_params_for_plot(list_val, list_key=list_txtparamPM):
    if len(list_val) != len(list_key):
        sentence = "Lengthes of list_val and list_key differ in prepare_params_for_plot() in Plot_maps\n"
        sentence += "Nb_of_parameters= %d, Nb of values %d !!!\n" % (len(list_key), len(list_val))
        sentence += "Please check the number of parameters for Plot_Maps inputs"
        raise ValueError(sentence)

    zippeddict = list(zip(list_key, list_val))

    resdict = {}
    for key, val in zippeddict:
        resdict[key] = val

    return resdict

if __name__ == '__main__':
#     import Version1 as mainGui

    dict_parameters = prepare_params_for_plot(list_valueparamPM)

    PlotMapsApp = wx.App()
    PMFrame = MainFrame_plotmaps(None, -1, 'PlotMaps2 Board', dict_parameters)
    PMFrame.Show(True)
    PlotMapsApp.MainLoop()

