# -*- coding: utf-8 -*-
"""
Module of Lauetools project to parse efficiently .fit files (ASCII format) generated by Laue microdiffraction analysis.

Original and working version comes from the module of S. Bongiorno, S. Tardif and Beatriz De Goes Foschiani in the state of march 2025: IOfitfile.py
"""

from LaueTools import IOLaueTools as IOLT   # read and write ASCII file  (IO) 
from LaueTools import readmccd as RMCCD     # read CCD and detector binary file, PeakSearch methods
from LaueTools import generaltools as GT

import numpy as np
import sys,os
import time
import multiprocessing
import itertools

from tqdm.notebook import tqdm

from PIL import Image
from PIL.ExifTags import TAGS

def get_nbdigits_filename(nb_files: int) -> int:
        res_divis = nb_files
        remainder = 0
        nb_digits = 1

        while res_divis > 9:
            res_divis = nb_files // (10 ** nb_digits)
            remainder = nb_files %  (10 ** nb_digits)
            nb_digits += 1

        if res_divis == 1 and remainder == 0: ## i.e., if nb_files is 10^x no need for the extra digit
            return nb_digits - 1
        else:
            return nb_digits
    
def create_fit_obj(file_index: int, nbdigits_filename: int, folderpath: str, prefix = 'img_', suffix = '_g0.fit'):
                filename = prefix + str(file_index).zfill(nbdigits_filename) + suffix
                return parsed_fitfile(os.path.join(folderpath, filename))

# ------ get the exact motor position from the header of one image ----- #
        
def get_motor(ExpFolder: str, image_index: int, nbdigits_filename: int) -> np.ndarray:
    fileimg_temp = 'img_' + str(image_index).zfill(nbdigits_filename) + '.tif'
    fileimg_temp_abs = os.path.join(ExpFolder,fileimg_temp)
    
    EXIF = Image.open(fileimg_temp_abs).getexif()
    
    # 0x010E is the HEX value for the ImageDescription tag
    ech_pos = (EXIF.get(0x010E))
    
    # split and evaluate ech_pos (whose type is a string) to get xech and yech
    xech = float(ech_pos.split(' ')[0].split('=')[1])
    yech = float(ech_pos.split(' ')[1].split('=')[1])
    
    return np.array([xech, yech])


    
def get_motor_fileseries(ExpFolder: str, nb_rows: int, nb_cols: int, nbdigits_filename: int) -> np.ndarray:
    
    nb_cpus = multiprocessing.cpu_count()
    
    print(f"Using {nb_cpus} cpus.")
    
    nb_files = nb_rows * nb_cols

    get_motor_args = zip(itertools.repeat(ExpFolder),
                                         range(0,nb_files),
                                         itertools.repeat(nbdigits_filename))
    tpos1 = time.time()
    
    with multiprocessing.Pool(nb_cpus) as pool:
        abs_pos = pool.starmap(get_motor,
                               tqdm(get_motor_args, total = nb_files, desc = 'Fetching motor positions'),
                               chunksize = 1)

    tpos2 = time.time()

    if not multiprocessing.active_children():
        print('Done!')

    abs_pos = np.array(abs_pos)

    xech_abs = abs_pos[:,0].reshape(nb_rows, nb_cols)
    yech_abs = abs_pos[:,1].reshape(nb_rows, nb_cols)

    xech_rel = ((xech_abs - xech_abs[0,0]) * 1e3)#[:nb_images_per_row]
    yech_rel = ((yech_abs - yech_abs[0,0]) * 1e3)#[::nb_images_per_row]
    #np.savetxt('motor_position_list.txt', ech_pos)

    return xech_rel, yech_rel


# Sam's class, modified by Beatriz
class parsed_fitfile:
    """
    Parse the .fit file in a parse_fitfile object
    Attributes:
    filename :
    corfile  :
    
    UB          :
    B0          :
    UBB0        :
    Element     :
    EulerAngles :
   
    GrainIndex          :
    MeanDevPixel        :
    NumberOfIndexedSpot :
    indexed_hkls        :
    
    a           :
    b           :
    c           :
    alpha       :
    beta        :
    gamma       :
    a_prime     :
    b_prime     :
    c_prime     :
    astar_prime :
    bstar_prime :
    cstar_prime :
    boa         : b/a, b over a
    coa         : c/a, c over a

    dev_sample :
    deviatoric :
    
    peak: dictionary
          keys   -> miller indices (e.g. '0 0 6', or '-1 0 11')
          values -> spot_index intensity h k l 2theta Chi Xexp Yexp Energy GrainIndex PixDev
    
    software           :
    timestamp          :
    CCDLabel           :
    FrameDimension     :
    PixelSize          :
    DetectorParameters :
    dd   :
    xbet :
    xcen :
    xgam :
    ycen :
    """
    
    def __init__(self, filename, verbose=False):
        try:
            with open(filename, "rU") as f:
                self.filename = filename

                # ----- read the header -----
                l = f.readline()
                self.corfile = l.replace("\n","").split(" ")[-1]

                l = f.readline()
                self.timestamp, self.software = l.lstrip("# File created at ").replace("\n","").split(" with ")

                # ----- read the footer -----
                l = f.readline().replace("\n", "")
                while l != "\n" and l != "" and l != "#":
                    try:
                        self.__param__()[l](f, l)
                        if verbose:
                            print("read ", l)
                        l = f.readline().replace("\n", "")

                    except KeyError:
                        try:
                            # print l.split(':')[0]
                            self.__param__()[l.split(":")[0]](f, l)
                            l = f.readline().replace("\n", "")
                            
                        except KeyError:
                            print("could not read line {}".format(l))
                            l = f.readline().replace("\n", "")
                            
                self.indexed_hkls = list(self.peak.keys())

                # some extra calculations to get the direct and reciprocal lattice basis vector
                # NOTE: the scale of the lattice basis vector is UNKNOWN !!!
                #       they are given here with a arbitrary scale factor
                if not hasattr(self, "UBB0"):
                    self.UBB0 = np.dot(self.UB, self.B0)
                    
                try:
                    self.astar_prime = self.UBB0[:, 0]
                    self.bstar_prime = self.UBB0[:, 1]
                    self.cstar_prime = self.UBB0[:, 2]

                    self.a_prime = np.cross(self.bstar_prime, self.cstar_prime) / np.dot(self.astar_prime, np.cross(self.bstar_prime, self.cstar_prime))
                        
                    self.b_prime = np.cross(self.cstar_prime, self.astar_prime) / np.dot(self.bstar_prime, np.cross(self.cstar_prime, self.astar_prime))
                        
                    self.c_prime = np.cross(self.astar_prime, self.bstar_prime) / np.dot(self.cstar_prime, np.cross(self.astar_prime, self.bstar_prime))

                    self.boa = np.linalg.linalg.norm(self.b_prime) / np.linalg.linalg.norm(self.a_prime)
                    self.coa = np.linalg.linalg.norm(self.c_prime) / np.linalg.linalg.norm(self.a_prime)

                    self.alpha = (np.arccos(np.dot(self.b_prime, self.c_prime) / np.linalg.linalg.norm(self.b_prime) / np.linalg.linalg.norm(self.c_prime)) * 180.0 / np.pi)
                    
                    self.beta = (np.arccos(np.dot(self.c_prime, self.a_prime) / np.linalg.linalg.norm(self.c_prime) / np.linalg.linalg.norm(self.a_prime)) * 180.0 / np.pi)
                    
                    self.gamma = (np.arccos(np.dot(self.a_prime, self.b_prime) / np.linalg.linalg.norm(self.a_prime) / np.linalg.linalg.norm(self.b_prime)) * 180.0 / np.pi)
                except ValueError:
                    print("could not compute the reciprocal space from the UBB0")

        except IOError:
            if verbose: print("file {} not found! or problem of reading it!".format(filename))
            pass
            

    # dictionary definitions for handling the LaueTools .fit file lines
    def __param__(self):
        return {
            "#UB matrix in q= (UB) B0 G* ": self.__UB__, #self.UB
            "#B0 matrix in q= UB (B0) G*": self.__B0__, #self.B0
            "#UBB0 matrix in q= (UB B0) G* i.e. recip. basis vectors are columns in LT frame: astar = UBB0[:,0], bstar = UBB0[:,1], cstar = UBB0[:,2]. (abcstar as columns on xyzlab1, xlab1 = ui, ui = unit vector along incident beam)": self.__UBB0__, #self.UBB0
            "#UBB0 matrix in q= (UB B0) G* , abcstar as lines on xyzlab1, xlab1 = ui, ui = unit vector along incident beam : astar = UBB0[0,:], bstar = UBB0[1,:], cstar = UBB0[2,:]": self.__UBB0__,  #self.UBB0
            "#deviatoric strain in crystal frame (10-3 unit)": self.__devCrystal__, #self.deviatoric
            "#deviatoric strain in direct crystal frame (10-3 unit)": self.__devCrystal__, #self.deviatoric
            "#deviatoric strain in sample2 frame (10-3 unit)": self.__devSample__, #self.dev_sample
            "#DetectorParameters": self.__DetectorParameters__, #self.DetectorParametrs
            "#pixelsize": self.__PixelSize__, #self.PixelSize
            "#Frame dimensions": self.__FrameDimension__, #self.FrameDimension
            "#CCDLabel": self.__CCDLabel__,#self.CDDLabel
            "#Element": self.__Element__, #self.Element
            "#grainIndex": self.__GrainIndex__, #self.GrainIndex
            "##spot_index intensity h k l 2theta Chi Xexp Yexp Energy GrainIndex PixDev": self.__Peaks__, #self.peak
            "##spot_index Intensity h k l pixDev energy(keV) Xexp Yexp 2theta_exp chi_exp Xtheo Ytheo 2theta_theo chi_theo Qx Qy Qz": self.__Peaks__,##self.peak
            "##spot_index Intensity h k l pixDev energy(keV) Xexp Yexp 2theta_exp chi_exp Xtheo Ytheo 2theta_theo chi_theo Qx Qy Qz 2theta chi X Y I peak_X peak_Y peak_Itot peak_Isub peak_fwaxmaj peak_fwaxmin peak_inclination Xdev Ydev peak_bkg Ipixmax": self.__Peaks__,##self.peak
            "##spot_index Intensity h k l pixDev energy(keV) Xexp Yexp 2theta_exp chi_exp Xtheo Ytheo 2theta_theo chi_theo Qx Qy Qz peak_Itot peak_fwaxmaj peak_fwaxmin peak_inclination Xdev Ydev peak_bkg Ipixmax grainindex": self.__Peaks__,
            "##spot_index Intensity h k l pixDev energy(keV) Xexp Yexp 2theta_exp chi_exp Xtheo Ytheo 2theta_theo chi_theo Qx Qy Qz grainindex": self.__Peaks__,
            "#Number of indexed spots": self.__NumberIndexedSpots__, #self.NumberOfIndexedSpots
            "#Mean Deviation(pixel)": self.__MeanDev__, #self.MeanDevPixel
            "#Euler angles phi theta psi (deg)": self.__EulerAngles__, #self.EulerAngles
            "#new lattice parameters": self.__NewParam__ # self.a, self.b, self.c, self.alpha, self.beta, self.gamma        
        }

    def __UB__(self, f, l):
        
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").replace("#","").split()
        ub11, ub12, ub13 = float(l[0]), float(l[1]), float(l[2])
        
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").replace("#","").split()
        ub21, ub22, ub23 = float(l[0]), float(l[1]), float(l[2])
        
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").replace("#","").split()
        ub31, ub32, ub33 = float(l[0]), float(l[1]), float(l[2])
        
        self.UB = np.array([[ub11, ub12, ub13], 
                            [ub21, ub22, ub23], 
                            [ub31, ub32, ub33]])

    def __B0__(self, f, l):
        
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").replace("#","").split()
        b011, b012, b013 = float(l[0]), float(l[1]), float(l[2])
        
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").replace("#","").split()
        b021, b022, b023 = float(l[0]), float(l[1]), float(l[2])
        
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").replace("#","").split()
        b031, b032, b033 = float(l[0]), float(l[1]), float(l[2])
        
        self.B0 = np.array([[b011, b012, b013], 
                            [b021, b022, b023], 
                            [b031, b032, b033]])

    def __UBB0__(self, f, l):
        
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").replace("#","").split()
        ubb011, ubb012, ubb013 = float(l[0]), float(l[1]), float(l[2])
        
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").replace("#","").split()
        ubb021, ubb022, ubb023 = float(l[0]), float(l[1]), float(l[2])
        
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").replace("#","").split()
        ubb031, ubb032, ubb033 = float(l[0]), float(l[1]), float(l[2])
        
        self.UBB0 = np.array([[ubb011, ubb012, ubb013], 
                              [ubb021, ubb022, ubb023], 
                              [ubb031, ubb032, ubb033]])

    def __devCrystal__(self, f, l):
        
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").replace("#","").split()
        ep11, ep12, ep13 = float(l[0]) * 1e-3, float(l[1]) * 1e-3, float(l[2]) * 1e-3
        
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").replace("#","").split()
        ep22, ep23 = float(l[1]) * 1e-3, float(l[2]) * 1e-3
        
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").replace("#","").split()
        ep33 = float(l[2]) * 1e-3
        
        self.deviatoric = np.array([[ep11, ep12, ep13], 
                                    [ep12, ep22, ep23], 
                                    [ep13, ep23, ep33]])

    def __devSample__(self, f, l):
        
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").replace("#","").split()
        ep_sample11, ep_sample12, ep_sample13 = (float(l[0]) * 1e-3, float(l[1]) * 1e-3, float(l[2]) * 1e-3)
        
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").replace("#","").split()
        ep_sample22, ep_sample23 = float(l[1]) * 1e-3, float(l[2]) * 1e-3
        
        l = f.readline().replace("[", "").replace("]", "").replace("\n", "").replace("#","").split()
        ep_sample33 = float(l[2]) * 1e-3
        
        self.dev_sample = np.array([[ep_sample11, ep_sample12, ep_sample13],
                                    [ep_sample12, ep_sample22, ep_sample23],
                                    [ep_sample13, ep_sample23, ep_sample33]])

    def __DetectorParameters__(self, f, l):
        
        l = (f.readline().replace("[", "").replace("]", "").replace("\n", "").replace(" ", "").replace("#","").split(","))
        
        self.dd = float(l[0])
        self.xcen = float(l[1])
        self.ycen = float(l[2])
        self.xbet = float(l[3])
        self.xgam = float(l[4])
        self.DetectorParameters = [self.dd, self.xcen, self.ycen, self.xbet, self.xgam]

    def __PixelSize__(self, f, l):
        
        l = (f.readline().replace("[", "").replace("]", "").replace("\n", "").replace(" ", "").replace("#","").split(","))
        
        self.PixelSize = float(l[0])

    def __FrameDimension__(self, f, l):
        
        l = f.readline().replace("\n", "").replace("#","")
        
        if l[0] == "[":
            l = l.replace("[", "").replace("]", "").replace("#","").split(", ")
        elif l[0] == "(":
            l = l.replace("(", "").replace(")", "").replace("#","").split(", ")
            
        self.FrameDimension = [float(l[0]), float(l[1])]

    def __CCDLabel__(self, f, l):
        
        l = (f.readline().replace("[", "").replace("]", "").replace("\n", "").replace(" ", "").replace("#","").split(","))
        
        self.CCDLabel = l[0]

    def __Element__(self, f, l):
        
        l = (f.readline().replace("[", "").replace("]", "").replace("\n", "").replace(" ", "").replace("#","").split(","))
        
        self.Element = l[0]

    def __GrainIndex__(self, f, l):
        
        l = (f.readline().replace("[", "").replace("]", "").replace("\n", "").replace(" ", "").replace("#","").split(","))
        
        self.GrainIndex = l[0]
        
        
    def __EulerAngles__(self, f, l):
        
        l = f.readline()
        
        l1 = (' '.join(l.split()).replace("[", "").replace("]", "").replace("\n", "").replace("#","").strip().split(" "))
        
        self.EulerAngles = np.array([float(l1[0]), float(l1[1]), float(l1[2])])
        
    def __NewParam__(self, f, l):
        
        l = f.readline()
        
        l1 = (' '.join(l.split()).replace("[", "").replace("]", "").replace("\n", "").replace("#","").strip().split(" "))
        
        self.a, self.b, self.c = float(l1[0]), float(l1[1]), float(l1[2])

    def __Peaks__(self, f, l):
        
        self.peak = {}
        
        for _ in list(range(self.NumberOfIndexedSpots)):
            l = f.readline().replace("#","").split()
            
            self.peak["{:d} {:d} {:d}".format(int(float(l[2])), int(float(l[3])), int(float(l[4])))] = l 

    def __NumberIndexedSpots__(self, _, l):
        
        self.NumberOfIndexedSpots = int(l.split(" ")[-1])

    def __MeanDev__(self, _, l):
        
        self.MeanDevPixel = float(l.split(" ")[-1])
    
       
    def print_indexed_hkls(self, nb_cols = 6):
        
        mylist = self.indexed_hkls
        
        # extract hkl from list
        h, k, l = [], [], []

        for elem in mylist:
            #split line to get h, k and l -> convert string to int -> append it
            h.append(int(elem.split(' ')[0]))
            k.append(int(elem.split(' ')[1]))
            l.append(int(elem.split(' ')[2]))

        for i in range(1, len(mylist) + 1):
            print(f"[{h[i-1]:3d}, {k[i-1]:3d}, {l[i-1]:3d}]  ", end = "")
            if i % nb_cols == 0:
                print()                

                

# Create a list of parsed_fitfile objects

class parsed_fitfileseries:

    def __init__(self, folderpath: str, nb_cols: int, nb_rows: int, prefix = 'img_', suffix = '_g0.fit', use_multiprocessing = True, nbdigits=4):
        
        
        self.folderpath = folderpath
        self.nb_cols = nb_cols
        self.nb_rows = nb_rows
        self.nb_files = self.nb_cols * self.nb_rows

        if nbdigits:
            self.nbdigits_filename = nbdigits
        
        # ----- create a list whose elements are the class parsed_fitfile, initialized for each file in the folder ------
        
        if use_multiprocessing:
            self.nb_cpus = multiprocessing.cpu_count()
            print(f"Using {self.nb_cpus} cpus.")
            
            create_fit_obj_args = zip(range(0,self.nb_files),
                                      itertools.repeat(self.nbdigits_filename),
                                      itertools.repeat(self.folderpath),
                                      itertools.repeat(prefix),
                                      itertools.repeat(suffix))
            
            with multiprocessing.Pool(self.nb_cpus) as pool:
                self.get_fitlist = pool.starmap(create_fit_obj,
                                                tqdm(create_fit_obj_args, total = self.nb_files, desc = 'Parsing progress'),
                                                chunksize = 1)
                
            if not multiprocessing.active_children():
                GT.printgreen("Done!")
            
        else:
            print("Using a single cpu.")
            obj_list = []

            for file_index in tqdm(range(0, self.nb_files), total = self.nb_files, desc = 'Parsing progress'):
                filename = prefix + str(file_index).zfill(self.nbdigits_filename) + suffix
                obj_list.append(parsed_fitfile(os.path.join(self.folderpath, filename)))
            
            self.get_fitlist = obj_list
            
        # ----- fill object with lists of the single class attributes -----
        
        self.UB = self.__collect_attribute__('UB', (self.nb_files, 3, 3))
        self.B0 = self.__collect_attribute__('B0', (self.nb_files, 3, 3 ))
        self.UBB0 = self.__collect_attribute__('UBB0', (self.nb_files, 3, 3 ))
        
        self.EulerAngles = self.__collect_attribute__('EulerAngles', (self.nb_files, 3,))
        self.MeanDevPixel = self.__collect_attribute__('MeanDevPixel', (self.nb_files, 1,))
        self.NumberOfIndexedSpots = self.__collect_attribute__('NumberOfIndexedSpots', (self.nb_files, 1,))
        
        self.a = self.__collect_attribute__('a', (self.nb_files, 1,))
        self.b = self.__collect_attribute__('b', (self.nb_files, 1,))
        self.c = self.__collect_attribute__('c', (self.nb_files, 1,))
        self.alpha = self.__collect_attribute__('alpha', (self.nb_files, 1,))
        self.beta = self.__collect_attribute__('beta', (self.nb_files, 1,))
        self.gamma = self.__collect_attribute__('gamma', (self.nb_files, 1,))
        
        self.a_prime = self.__collect_attribute__('a_prime', (self.nb_files, 3,))
        self.b_prime = self.__collect_attribute__('b_prime', (self.nb_files, 3,))
        self.c_prime = self.__collect_attribute__('c_prime', (self.nb_files, 3,))
        self.astar_prime = self.__collect_attribute__('astar_prime', (self.nb_files, 3,))
        self.bstar_prime = self.__collect_attribute__('bstar_prime', (self.nb_files, 3,))
        self.cstar_prime = self.__collect_attribute__('cstar_prime', (self.nb_files, 3,))
        self.boa = self.__collect_attribute__('boa', (self.nb_files, 1,))
        self.coa = self.__collect_attribute__('coa', (self.nb_files, 1,))
        
        self.dev_sample = self.__collect_attribute__('dev_sample', (self.nb_files, 3, 3 ))
        self.deviatoric = self.__collect_attribute__('deviatoric', (self.nb_files, 3, 3 ))
        
        self.exx = self.deviatoric[:,0,0].reshape(self.nb_rows, self.nb_cols)
        self.eyy = self.deviatoric[:,1,1].reshape(self.nb_rows, self.nb_cols)
        self.ezz = self.deviatoric[:,2,2].reshape(self.nb_rows, self.nb_cols)
        self.exy = self.deviatoric[:,0,1].reshape(self.nb_rows, self.nb_cols)
        self.exz = self.deviatoric[:,0,2].reshape(self.nb_rows, self.nb_cols)
        self.eyz = self.deviatoric[:,1,2].reshape(self.nb_rows, self.nb_cols)

        self.exx_sample = self.dev_sample[:,0,0].reshape(self.nb_rows, self.nb_cols)
        self.eyy_sample = self.dev_sample[:,1,1].reshape(self.nb_rows, self.nb_cols)
        self.ezz_sample = self.dev_sample[:,2,2].reshape(self.nb_rows, self.nb_cols)
        self.exy_sample = self.dev_sample[:,0,1].reshape(self.nb_rows, self.nb_cols)
        self.exz_sample = self.dev_sample[:,0,2].reshape(self.nb_rows, self.nb_cols)
        self.eyz_sample = self.dev_sample[:,1,2].reshape(self.nb_rows, self.nb_cols)
            
    # ----- return a list containing the value of the specified parsed_fitfile attribute for each file -----
    
    def __collect_attribute__(self, attribute: str, shape: tuple) -> list:
        """
        Shape is (nb_files, shape of data)
        For EulerAngles  shape = (nb_files, 3,  )
        For dev_sample   shape = (nb_files, 3, 3)
        For MeanDevPixel shape = (nb_files, 1,  )"""
        
        my_list = np.full(shape, np.NaN)
        
        nb_files = shape[0]
        for i in range(0, nb_files): 
            try: 
                my_list[i] = getattr(self.get_fitlist[i], attribute)
            except AttributeError:
                pass
        
        return my_list
    
    # ----- class functions to access the spot parameters for  given miller indices -----
    
    def Xpos(self, miller_index: str) -> np.ndarray:
        my_list = np.full(self.nb_files, np.NaN)
        
        for i in range(0, self.nb_files):
            try:
                my_list[i] = (self.get_fitlist[i]).peak[miller_index][7]
                
            except KeyError:
                pass
                
        return my_list
    
    def Ypos(self, miller_index: str) -> np.ndarray:
        my_list = np.full(self.nb_files, np.NaN)
        
        for i in range(0, self.nb_files):
            try:
                my_list[i] = (self.get_fitlist[i]).peak[miller_index][8]
                
            except KeyError:
                pass
                
        return my_list
    
    def twotheta(self, miller_index: str) -> np.ndarray:
        my_list = np.full(self.nb_files, np.NaN)
        
        for i in range(0, self.nb_files):
            try:
                my_list[i] = (self.get_fitlist[i]).peak[miller_index][5]
                
            except KeyError:
                pass
                
        return my_list
    
    def chi(self, miller_index: str) -> np.ndarray:
        my_list = np.full(self.nb_files, np.NaN)
        
        for i in range(0, self.nb_files):
            try:
                my_list[i] = (self.get_fitlist[i]).peak[miller_index][6]
                
            except KeyError:
                pass
                
        return my_list
    
    def PixDev(self, miller_index: str) -> np.ndarray:
        my_list = np.full(self.nb_files, np.NaN)
        
        for i in range(0, self.nb_files):
            try:
                my_list[i] = (self.get_fitlist[i]).peak[miller_index][11]
                
            except KeyError:
                pass
                
        return my_list
    
    def intensity(self, miller_index: str) -> np.ndarray:
        my_list = np.full(self.nb_files, np.NaN)
        
        for i in range(0, self.nb_files):
            try:
                my_list[i] = (self.get_fitlist[i]).peak[miller_index][1]
                
            except KeyError:
                pass
                
        return my_list
    
