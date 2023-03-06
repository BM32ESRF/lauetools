#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'

import os

import wx

__path__ = os.path.dirname(__file__)


def get_icon_dir(name):
    
    return os.path.join(__path__, name)


def get_icon(name):

    fdir = get_icon_dir(name)

    return wx.Icon(fdir, wx.BITMAP_TYPE_PNG)

def get_icon_img(name, size=(16, 16)):

    img = wx.Image(get_icon_dir(name))

    return img.Scale(size[0], size[1], wx.IMAGE_QUALITY_HIGH)

def get_icon_bmp(name, size=(16, 16)):

    img = get_icon_img(name, size)

    return img.ConvertToBitmap()


def get_image_bmp(name, size=None):

    img = wx.Image(get_icon_dir(name))

    if size is not None:

        W = img.GetWidth()
        H = img.GetHeight()

        NewW = size[0]
        NewH = NewW * H * 1. / W

        img.Scale(int(NewW), int(NewH), wx.IMAGE_QUALITY_HIGH)

    return img.ConvertToBitmap()
