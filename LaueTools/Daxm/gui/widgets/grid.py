#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'


import wx
import wx.grid

class GridFrame(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent)

        # Create a wxGrid object
        grid = wx.grid.Grid(self, -1)

        # Then we call CreateGrid to set the dimensions of the grid
        # (100 rows and 10 columns in this example)
        grid.CreateGrid(20, 20)

        grid.SetRowLabelSize(0)
        grid.SetColLabelSize(0)

        grid.EnableDragGridSize(False)
        grid.SetDefaultColSize(1, True)
        grid.SetDefaultRowSize(1, True)

        self.Show()



if __name__ == '__main__':

    app = wx.App(0)
    frame = GridFrame(None)

    app.MainLoop()
