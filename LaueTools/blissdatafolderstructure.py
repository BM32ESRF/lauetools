# -*- coding: utf-8 -*-
r"""
blissdatafolderstructures is a module to provide some helper functions given a folder on nice server by means of BLISS
"""
import sys, os


def setExperimentFolder_with_date(ExperimentFolder, expDate=None, addfolder='RAW_DATA'):
    """return the experiment folder path by adding the proper date
    expDate: date is a string yearmonthday format '20231003'"""
    try:
        dirs = os.listdir(ExperimentFolder)
    except FileNotFoundError:
        print(f'the folder {ExperimentFolder} does not exist!' )
        return None
    
    if expDate:
        try:
            ExperimentFolder = os.path.join(ExperimentFolder,expDate)
            dirs = os.listdir(ExperimentFolder)
            
        except FileNotFoundError:
            print(f'the folder {ExperimentFolder} does not exist!' )
            return None
        return ExperimentFolder
    
    if len(dirs)==1:
        expDate = dirs[0]
        ExperimentFolder=os.path.join(ExperimentFolder,expDate)
    elif len(dirs)==0:
        print(f'ExperimentFolder {ExperimentFolder} is empty')
        return None
    else:
        print(f'Please recall the function with the argument expDate among:')
        print(dirs)
        return None
    
    if addfolder is not None:
        ExperimentFolder=os.path.join(ExperimentFolder,addfolder)
        
    return ExperimentFolder

def createdatfolder(folder, defaultname='datfiles'):
    """ create a mirror folder if 'RAW_DATA' is in the path, else create a subfolder with defaultname
    
     :param defaultname: last folder name to be added is added at the end of the path except if defaultname is None or '.'"""
        
    lf = os.path.abspath(folder).split('/')
    #print(lf)
    if 'RAW_DATA' not in lf or folder is None:  #use this folder or add a folder to it
        # print('%s folder does not contain RAW_DATA'%lf)
        # print('No mirror folder created')
        if defaultname in (None,'.',''):
            lastfolder = ''
        else:
            lastfolder = defaultname
        genfolder = os.path.join(folder,lastfolder)
        if not os.path.isdir(genfolder):  # not a existing path
            #print('not existing %s'%genfolder)
            os.mkdir(genfolder)
        #print('genfolder', genfolder)
        return genfolder
    else:  # folder from BLISS data on nice
        return createmirrorfolder(folder, defaultname)
    
def createmirrorfolder(folder, defaultname='datfiles'):
    """create a mirror folder of a folder in RAW_DATA
    and write a corresponding PROCESSED_DATA folder
    
    :param defaultname: last folder name to be added is added at the end of the path except if defaultname is None or '.' or ''"""

    lf = os.path.abspath(folder).split('/')
    if 'RAW_DATA' not in lf:
        return createdatfolder(folder)
    
    irawdata=lf.index('RAW_DATA')
    ifolder = ''
    for gg in lf[:irawdata]:
        ifolder=os.path.join(ifolder,gg)
    #print('ifolder',ifolder)
    ffolder = ''
    createfolder = []
    for gg in lf[irawdata+1:]:
        ffolder=os.path.join(ffolder,gg)
        createfolder.append(ffolder)
    #print('ffolder',ffolder)
    if defaultname in (None,'.',''):
        lastfolder = ''
    else:
        lastfolder = defaultname
        createfolder.append(os.path.join(createfolder[-1],lastfolder))
        
    #print('createfolder',createfolder)
    genfolder = os.path.join('/',ifolder,'PROCESSED_DATA',ffolder,lastfolder)
    #print('genfolder in mirror', genfolder)

    if os.path.isdir(genfolder):
        print('Cool! %s already exists ...'%genfolder)
    else:
        for fff in createfolder:
            fpath= os.path.join('/',ifolder,'PROCESSED_DATA',fff)
            try:
                os.mkdir(fpath)
            except (FileExistsError, FileNotFoundError):
                continue
        if os.path.isdir(genfolder):
            print('%s folder has been created'%genfolder)
    
    return genfolder

def setimagefilename(prefix, imageindex, folder=None, suffix='.tif',sizeofzeropadding = 4):
    """setter of path of full path to image file"""
    paddedindex = '%s'%str(imageindex).zfill(sizeofzeropadding)
    imagefilename = f"{prefix}{paddedindex}{suffix}"
    if folder is not None:
        return os.path.join(folder,imagefilename)
    else:
        return imagefilename



