# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 13:40:27 2013

@author: guest
"""

import wx
import os

import LaueTools.FileSeries.module_graphique as modgraph
import LaueTools.Daxm.gui.icons.icon_manager as mycons
import LaueTools.dict_LaueTools as DictLT

SB_INFO = 0

import LaueTools.FileSeries.param_multigrain as PAR
# import dict_Interface as DI

from LaueTools.FileSeries import Peak_Search as PS
from LaueTools.FileSeries import Index_Refine as IR
from LaueTools.FileSeries import Build_Summary as BS
from LaueTools.FileSeries import Plot_Maps2 as PM

# from Plot_Maps import *
#from .Sort_Grains import *
#from .Plot_Grain import *

LaueToolsProjectFolder = os.path.dirname(os.path.abspath(os.curdir))

print('LaueToolProjectFolder', LaueToolsProjectFolder)

class MainWindow(wx.Frame):
    def __init__(self, parent, id, title, initialparameters):

        wx.Frame.__init__(self, parent, id, title, wx.DefaultPosition, wx.Size(800, 600))

        self.initialparameters = initialparameters
        # menu
        menubar = wx.MenuBar()
        file = wx.Menu()
        edit = wx.Menu()
        file.Append(101, '&Open', 'Open a new document')
        save = wx.MenuItem(file, 102, '&Save\tCtrl+S', 'Save the Application')
        # save.SetBitmap(wx.Image(os.path.join(LaueToolsProjectFolder, 'icons', 'save.png'), wx.BITMAP_TYPE_PNG).ConvertToBitmap())
        iconsave = os.path.join(LaueToolsProjectFolder, 'icons', 'save.png')
        mycons.get_icon_dir(iconsave)
        save.SetBitmap(mycons.get_icon_bmp("harddrive.png"))
        file.Append(save)
        file.Append(103, '&Save as', 'Save the document and choose his name and location')
        quit = wx.MenuItem(file, 105, '&Quit\tCtrl+Q', 'Quit the Application')
        # quit.SetBitmap(wx.Image(os.path.join(LaueToolsProjectFolder, 'exit.png'), wx.BITMAP_TYPE_PNG).ConvertToBitmap())
        file.Append(quit)
        menubar.Append(file, '&File')
        menubar.Append(edit, '&Edit')
        self.SetMenuBar(menubar)
        self.Centre()
        self.Bind(wx.EVT_MENU, self.OnQuit, id=105)
        self.Bind(wx.EVT_MENU, self.OnSave, id=102)
        self.Bind(wx.EVT_MENU, self.OnSaveAs, id=103)
        self.dirname = ""
        self.filename = ""

        self.panel = wx.Panel(self)

        # buttons PEAK_SEARCH, BUILD_SUMMARY, PLOT_MAPS, SORT_GRAINS, PLOT_GRAIN_MAPS
        btnPeaksearch = wx.Button(self.panel, 10, 'PEAK_SEARCH')
        btnPeaksearch.Bind(wx.EVT_BUTTON, self.OnPeakSearch)
        btnIndexrefine = wx.Button(self.panel, 11, 'INDEX_REFINE')
        btnIndexrefine.Bind(wx.EVT_BUTTON, self.OnIndexRefine)
        btnBuildsummary = wx.Button(self.panel, 12, 'BUILD_SUMMARY')
        btnBuildsummary.Bind(wx.EVT_BUTTON, self.OnBuildSummary)
        btnPlotmaps = wx.Button(self.panel, 13, 'PLOT_MAPS')
        btnPlotmaps.Bind(wx.EVT_BUTTON, self.OnPlotMaps)
        # btnSortgrains = wx.Button(self.panel, 14, 'SORT_GRAINS')
        # btnSortgrains.Bind(wx.EVT_BUTTON, self.OnSortgrains)
        # btnPlotgrainmaps = wx.Button(self.panel, 15, 'PLOT_GRAIN_MAPS')
        # btnPlotgrainmaps.Bind(wx.EVT_BUTTON, self.OnPlotgrainmaps)

        box = wx.BoxSizer(wx.HORIZONTAL)
        box.Add(btnPeaksearch, 1)
        box.Add(btnIndexrefine, 1)
        box.Add(btnBuildsummary, 1)
        box.Add(btnPlotmaps, 1)
        # box.Add(btnSortgrains, 1)
        # box.Add(btnPlotgrainmaps, 1)

        self.panel.SetSizer(box)
        self.Centre()

    def OnPeakSearch(self, event):
        print("start peak search file series")
#        PSboard = Background_Entryparameter_peaksearch(self, -1, 'Peak Search Parameters Board',
#                                                       objet_PS=Stock_PS,
#                                                       manag=manager)
#        PSboard.Show(True)

        Stock_PS = PS.Stock_parameters_PeakSearch(PS.LIST_TXTPARAM_FILE_PS, PS.list_valueparamPS)

        PeakSearchSeries = PS.MainFrame_peaksearch(self, -1, 'Peak Search Parameters Board',
                                            self.initialparameters, Stock_PS)
        PeakSearchSeries.Show(True)

    def OnIndexRefine(self, event):
        # previous step (peak search) has been launched
        if 'PeakList Folder' in self.initialparameters:
            print('IR.fill_list_valueparamIR(self.initialparameters)', IR.fill_list_valueparamIR(self.initialparameters))
            Stock_INDEXREFINE = IR.Stock_parameters_IndexRefine(IR.LIST_TXTPARAM_FILE_INDEXREFINE,
                                                            IR.fill_list_valueparamIR(self.initialparameters))
        # if not
        else:
            Stock_INDEXREFINE = IR.Stock_parameters_IndexRefine(IR.LIST_TXTPARAM_FILE_INDEXREFINE,
                                                            IR.list_valueparamIR)


        print('self.initialparameters',self.initialparameters)
        IRboard = IR.MainFrame_indexrefine(self, -1,
                                        'Index Refine Parameters Board',
                                        self.initialparameters,  # IR.initialparameters
                                        Stock_INDEXREFINE)

        IRboard.Show(True)

    def OnBuildSummary(self, event):

        if 'IndexRefine PeakList Folder' in self.initialparameters:
            list_valueparamBS = BS.fill_list_valueparamBS(self.initialparameters)
        else:
            list_valueparamBS = BS.list_valueparamBS


        BSboard = BS.MainFrame_BuildSummary(self, -1, 'Build Summary Parameters Board',
                                            list_valueparamBS)

        BSboard.Show(True)

    def OnPlotMaps(self, event):
#         if manager.matrice_callfunctions [2,2] != 0:

        if 'IndexRefine PeakList Folder' in self.initialparameters:
            dict_parameters = PM.prepare_params_for_plot([self.initialparameters["Map Summary File"],
                                                    self.initialparameters["File xyz"],
                                                    "fit"])
        else:
            dict_parameters = PM.prepare_params_for_plot(PM.list_valueparamPM)

        PMboard = PM.MainFrame_plotmaps(self, -1, 'Plot Maps Parameters Board',
                                            dict_parameters)

        PMboard.Show(True)
#         else:
#            wx.MessageBox('Warning! No path to the summary file register, choose your summary file manually')
#            PSboard=MainFrame_plotmaps(self, -1,'Plot Maps Parameters Board', Stock_PM, Stock_BS, Stock_BSi, manager)
#            PSboard.Show(True)

    def OnSortgrains(self, event):

        wx.MessageBox('This function is not yet implemented from this GUI', 'INFO')
        return

        PSboard = Background_parameters_SortGrain(None, -1, 'Entrer parameters Sort Grain',
                                    Stock_SG, Stock_BS, Stock_BSi, Stock_IR, Stock_PS, manager)
        PSboard.Show(True)

    def OnPlotgrainmaps(self, event):

        wx.MessageBox('This function is not yet implemented from this GUI', 'INFO')
        return

        PSboard = Background_parameters_PlotGrain(None, -1, 'Parameters plot Grain', Stock_PG, Stock_BS, Stock_BSi, manager, Stock_SG, Stock_PM)
        PSboard.Show(True)


# menu's functions
    def OnQuit(self, event):
        self.Close()

    def OnSaveAs(self, event):

        ret = False
        dlg = wx.FileDialog(self, "Save As", self.dirname, self.filename,
                       "Text Files (*.txt)|*.txt|All Files|*.*", wx.SAVE)
        if (dlg.ShowModal() == wx.ID_OK):
            self.filename = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
        # ## - Use the OnFileSave to save the file
#         if self.OnFileSave(e):
#             self.SetTitle(APP_NAME + " - [" + self.fileName + "]")
#             ret = True
#             dlg.Destroy()
#             return ret

    def OnSave(self, event):
        try:
            f = file(os.path.join(self.dirName, self.fileName), 'w')
            f.write(self.rtb.GetValue())
            self.PushStatusText("Saved file: " + str(self.rtb.GetLastPosition()) +
                                " characters.", SB_INFO)
            f.close()
            return True
        except:
            self.PushStatusText("Error in saving file.", SB_INFO)
            return False


# class Manager_callfunctions:
#     def __init__(self, matrice_callfunctions,
#                  Stock_PS, Stock_IR, Stock_BS, Stock_BSi, Stock_PM, Stock_SG, Stock_PG):
#         self.Stock_PS = Stock_PS
#         self.Stock_IR = Stock_IR
#         self.Stock_BS = Stock_BS
#         self.Stock_BSi = Stock_BSi
#         self.matrice_callfunctions = matrice_callfunctions
#         self.Stock_PM = Stock_PM
#         self.Stock_SG = Stock_SG
#         self.Stock_PG = Stock_PG


#     def Activate_PeakSearch(self):
#         if matrice_callfunctions[0, 0] == 0:
#              matrice_callfunctions[0, 0] = 1

#         else:
#             pass


#     def Activate_Indexrefine(self):
#         if matrice_callfunctions[1, 1] == 0:
#             matrice_callfunctions[1, 1] = 1
#         else:
#            pass

#     def Activate_BuildSummary(self):
#         if matrice_callfunctions[2, 2] == 0:
#             matrice_callfunctions[2, 2] = 1
#         else:
#            pass

#     def Activate_PlotMaps(self):
#         if matrice_callfunctions[3, 3] == 0:
#                 matrice_callfunctions[3, 3] = 1
#         else:
#            pass

#     def Activate_SortGrain(self):
#         if matrice_callfunctions[4, 4] == 0:
#                 matrice_callfunctions[4, 4] = 1
#         else:
#            pass

#     def Activate_PlotGrain(self):
#         if matrice_callfunctions[5, 5] == 0:
#                 matrice_callfunctions[5, 5] = 1
#         else:
#            pass

initialparameters = {}

MainFolder = os.path.join(LaueToolsProjectFolder, 'Examples', 'GeGaN')

print(("MainFolder", MainFolder))

initialparameters['file stf'] = os.path.join(MainFolder,
                                            'si.stf')
initialparameters['file xyz'] = os.path.join(MainFolder, 'fitfiles',
                                           'orig_nanox2_400__xy_0_to_5.dat')

initialparameters['Map Summary File'] = os.path.join(MainFolder, 'fitfiles',
                                           'orig_nanox2_400__SUMMARY_0_to_5_add_columns.dat')

list_txtparamPS = ['filepath',
                 'filepathout',
                 'fileprefix',
                 'filesuffix',
                 'Nbdigits',
                 'Nbpicture1',
                 'Nblastpicture',
                 'increment'
                 ]
list_valueparamPS = [None, None, None, None, 4, 0, 4, 1]

list_txtparamIR = ['filepathdat',
                 'filepathout',
                 'filedet',
                 'filedat',
                 'fileprefix',
                 'filesuffix',
                 'Nbdigits',
                 'Indimg'
                 ]
list_valueparamIR = [None, None, None, None, None, None, PAR.number_of_digits_in_image_name, modgraph.indimg]

list_txtparamBS = ['filepathfit',
                 'filexyzused',
                 'filepathout',
                 'fileprefix',
                 'filesuffix',
                 'Nbdigits',
                 'Indimg',
                 'filexyz',
                 'nx',
                 'ny',
                 'xfast',
                 'yfast',
                 'xstep',
                 'ystep'
                 ]
list_valueparamBS = [None, None, None, None, None, PAR.number_of_digits_in_image_name, modgraph.indimg, None, 60, 60, 1, 0, 0.5, -1]

list_txtparamBSi = ['filepathfit', 'filexyzused', 'filepathout', 'fileprefixfit', 'filesuffixfit', 'Nbdigits', 'Indimg', 'filepathim', 'filexyzi', 'fileprefix', 'filesuffix']
list_valueparamBSi = [None, None, None, None, None, PAR.number_of_digits_in_image_name, modgraph.indimg, None, None, None, None]

list_txtvaluePM = ['filesum',
                 'filexyz',
                 'maptype',
                 'filetype',
                 'substract_mean',
                 'numgrain',
                 'filter_on_pixdev_and_npeaks',
                 'maxpixdev_forfilter',
                 'minnpeaks_forfilter',
                 'strainmin_forplot',
                 'strainmax_forplot',
                 'pixdevmin_forplot',
                 'pixdevmax_forplot',
                 'npeaksmin_forplot',
                 'npeaksmax_forplot'
                 ]
list_valueparamPM = [initialparameters['Map Summary File'], initialparameters['file xyz'],
                     "fit", "LT", "no", 0, 0,
                     0.3, 20, -0.2, 0.2, 0, 0, 0, 0]

list_txtparamSG = ['filesum', 'filexyz', 'filepathout', 'test_mode']
list_valueparamSG = [None, None, None, 'no']

list_txtparamPG = ['filegrain',
                  'filexyz',
                  'filepathout',
                  'zoom',
                  'mapcolor',
                  'filter_on_pixdev_and_npeaks',
                  'maxpixdev',
                  'minnpeaks',
                  'map_prefix',
                  'test1',
                  'savefig_map',
                  'number_of_graphs',
                  'number_of_graphs_per_figure',
                  'grains_to_plot',
                  'gnumloc_min',
                  'gnumloc_max',
                  'maxvalue_for_plot',
                  'minvalue_for_plot',
                  'single_gnumloc',
                  'substract_mean'
                  ]

list_valueparamPG = [None, None, None, "yes", "gnumloc_in_grain_list",
                     "no", 0.5, 20, "new_z_", "no", 1, None, 9, "all", 0, 3, None, None, None, "no"]

#   -----------  complete list of parameters   -----------------------
# for peak_search -----------------------
LaueToolsProjectFolder = DictLT.LAUETOOLSFOLDER
print("LaueToolProjectFolder in main", LaueToolsProjectFolder)
MainFolder = os.path.join(LaueToolsProjectFolder, "Examples", "GeGaN")
print("MainFolder in main", MainFolder)

initialparameters = {}
initialparameters["ImageFolder"] = MainFolder
initialparameters["Output Folder (Peaklist)"] = os.path.join(MainFolder, "datfiles")
initialparameters["ImageFilename Prefix"] = "nanox2_400_"
initialparameters["ImageFilename Suffix"] = ".mccd"
initialparameters["PeakSearch Parameters File"] = os.path.join(
    MainFolder, "PeakSearch_nanox2_400_0000_LT_4.psp")
initialparameters["CCDLabel"] = "MARCCD165"
initialparameters["BackgroundRemoval"] = "auto"
initialparameters["BlackListed Peaks File"] = None
initialparameters["Selected ROIs File"] = None
initialparameters["startingindex"] = 0
initialparameters["finalindex"] = 5
initialparameters["stepindex"] = 1
initialparameters["nbdigits"] = 4


# ---for index_refine -----------------------------------------
initialparameters["IndexRefine PeakList Folder"] = os.path.join(MainFolder, "fitfiles")
initialparameters["PeakListCor Folder"] = initialparameters["Output Folder (Peaklist)"]
initialparameters["PeakList Filename Prefix"] = initialparameters["ImageFilename Prefix"]
initialparameters["PeakList Folder"] = os.path.join(MainFolder, "fitfiles")

initialparameters["PeakList Filename Suffix"] = ".dat"
initialparameters["Selected Peaks from File"] = 'None'
initialparameters["IndexRefine Parameters File"] = os.path.join(MainFolder, "GeGaN.irp")
initialparameters["Detector Calibration File .det"] = os.path.join(MainFolder, "calibGe_Feb2020.det")
initialparameters["GuessedUBMatrix"] = "None"
initialparameters["MinimumMatchingRate"] = 4.0
initialparameters["Selected Peaks from File"] = None

#   for build summary

initialparameters["IndexRefine PeakList Prefix"] = initialparameters["PeakList Filename Prefix"]
initialparameters["IndexRefine PeakList Suffix"] = '.fit'
initialparameters["stiffness file"] = os.path.join(LaueToolsProjectFolder, "Examples", "CuSi", "si.stf")
initialparameters["Material"] = 'Si'
initialparameters["file xyz"] = 'None'
initialparameters["Map shape"] = (1000, 1)
initialparameters["fast axis: x or y"] = 'x'
initialparameters["(stepX, stepY) microns"] = (1, 1)

# for plot_map2
initialparameters["Map Summary File"] = 'None'
initialparameters["File xyz"] = 'None'

if 0:
    # peaksearch parameters only
    list_valueparamPS = [initialparameters["ImageFolder"],
                        initialparameters["Output Folder (Peaklist)"],
                        initialparameters["ImageFilename Prefix"],
                        initialparameters["ImageFilename Suffix"],
                        initialparameters["nbdigits"],
                        initialparameters["startingindex"],
                        initialparameters["finalindex"],
                        initialparameters["stepindex"],
                        initialparameters["BackgroundRemoval"],
                        initialparameters["BlackListed Peaks File"],
                        initialparameters["Selected ROIs File"],
                        initialparameters["PeakSearch Parameters File"]]
    # indexrefine parameters only
    list_valueparamIR = IR.fill_list_valueparamIR(initialparameters)


    #matrice_callfunctions = np.zeros((6, 6))
    Stock_PS = PS.Stock_parameters_PeakSearch(PS.LIST_TXTPARAM_FILE_PS, list_valueparamPS)
    Stock_IR = IR.Stock_parameters_IndexRefine(list_txtparamIR, list_valueparamIR)
    Stock_BS = BS.Stock_parameters_BuildSummary_hand(list_txtparamBS, list_valueparamBS)
    Stock_BSi = BS.Stock_parameters_BuildSummary_image(list_txtparamBSi, list_valueparamBSi)
    # Stock_PM = Stock_parameters_PlotMaps(list_txtvaluePM, list_valueparamPM)
    #Stock_SG = Stock_parameters_SortGrain(list_txtparamSG, list_valueparamSG)
    #Stock_PG = Stock_parameters_PlotGrain(list_txtparamPG, list_valueparamPG)
    # manager = Manager_callfunctions(matrice_callfunctions, Stock_PS, Stock_IR, Stock_BS, Stock_BSi, Stock_PM, Stock_SG, Stock_PG)
    #     calc = IR.calcul(Stock_IR)

def start():
    app = wx.App()


    print('initialparameters at entrance Mainwindow')
    frame = MainWindow(None, -1, 'Laue File Series Analysis', initialparameters)
    frame.Show(True)
    app.MainLoop()

if __name__ == "__main__":
    start()


    print('initialparameters at entrance Mainwindow')
    frame = MainWindow(None, -1, 'Laue File Series Analysis', initialparameters)
    frame.Show(True)
    app.MainLoop()
