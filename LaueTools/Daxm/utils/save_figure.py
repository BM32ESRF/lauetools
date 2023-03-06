#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
__author__ = "Loic Renversade, CRG-IF BM32 @ ESRF"
__version__ = '$Revision$'


import os, sys

import matplotlib.backends.backend_pdf as mplpdf


def save(figs, name, ext, directory="" ,verbose=1):
    
    filename = os.path.join(directory, name)
    
    if ext == "png":
        
        saveas_png(figs, filename)
                   
        print_msg("Saved figures to '{}_xxx.png' files.".format(name), verbose)
        
    elif ext == "pdf":
        
        saveas_pdf(figs, filename)
                          
        print_msg("Saved figures to file '{}.pdf'.".format(name), verbose)
        
    else:
        
        sys.exit("Required file format '{}' is not available in saveFigure!".format(format)) 

def saveas_png(figs, name):
    
    for i, fig in enumerate(figs):
        
        fig.savefig("{}_{:03d}.png".format(name, i))

def saveas_pdf(figs, name):
    
    pdf = mplpdf.PdfPages(name+".pdf")

    for fig in figs: ## will open an empty extra figure :(
        pdf.savefig( fig )
    
    pdf.close()


#handle printing and verbosity 
def print_msg(msg, verbose):
    
    if verbose:
        
        print(msg)
        
        
        
    
    
    
    
    
    
    
    
