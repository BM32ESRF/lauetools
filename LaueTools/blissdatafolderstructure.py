# -*- coding: utf-8 -*-
r"""
blissdatafolderstructures is a module to provide some helper functions given a folder on nice server by means of BLISS
"""
import sys, os
import LaueTools.generaltools as GT


def getExperimentFolder_data_at_esrf(expId, nicefolder='visitor', projectsname=None, expDate=None):
    """projectsname can be mapgrainxl if under /data/projects"""
    if nicefolder != 'visitor':
        if projectsname is not None:
            nicefolder = os.path.join('projects',projectsname)
        else:
            raise ValueError(f'Please provide "projectsname" of your project. /data/projectsname')

    #--------------------------------------------
    ExperimentFolder = os.path.join('/data',nicefolder, f'{expId}/bm32/')

    listdates = os.listdir(ExperimentFolder)
    #print('possible dates',listdates)
    if len(listdates)>1 and expDate is None:
        txt = f'\nBe careful, there are several dates ... => {listdates}'
        txt += f'\nPlease provide expDate  with the string date e.g. {listdates[0]}'
        GT.printyellow(txt)
    
        # to uncomment two lines to precise the date if there are several ones
        ExperimentFolder= setExperimentFolder_with_date(ExperimentFolder, expDate='20250212')
        ExperimentFolder = os.path.join(ExperimentFolder,'RAW_DATA')
    else:
        ExperimentFolder= setExperimentFolder_with_date(ExperimentFolder)
   

    print('Data at ESRF: found ExperimentFolder at', ExperimentFolder)
    return ExperimentFolder

def getExperimentFolder(expId=None, folder=None, data_at_esrf= True, projectsname=None, expDate=None,
                                    nicefolder='visitor', checkoutput=True):
    ExperimentFolder = None
    if data_at_esrf:
        ExperimentFolder = getExperimentFolder_data_at_esrf(expId, nicefolder=nicefolder, projectsname=projectsname,
                                                            expDate=expDate)
    elif folder is not None:
        ExperimentFolder = folder

    if not checkoutput:
        return ExperimentFolder

    if os.path.exists(ExperimentFolder):
        GT.printgreen(f'\n"ExperimentFolder" exists ! : \n{ExperimentFolder}')
        
    if data_at_esrf and 'RAW_DATA' not in ExperimentFolder:
        GT.printyellow(f'\n"ExperimentFolder" does not contain "RAW_DATA"! Are you sure?')
    
    return ExperimentFolder


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
    
def setimages_subfolder(fullpath, rootfolder='RAW_DATA'):
    """extract terminal (tail) part from the full path to images starting from a given root folder
    
    example: '/data/AAA/BBB/RAW_DATA/sample1/dataset3/scan0025' => 'sample1/dataset3/scan0025'
    """
    lf = os.path.abspath(fullpath).split('/')
    tail_subfolder = fullpath
    if 'RAW_DATA' in lf:
        irawdata=lf.index('RAW_DATA')
        tail_subfolder = ''
        for gg in lf[irawdata+1:]:
            tail_subfolder=os.path.join(tail_subfolder,gg)
        tail_subfolder=os.path.join(tail_subfolder,gg)
    
    return tail_subfolder


def tree(path, max_level=2, prefix='', dirs_only=False, sort_by_date=False):
    """List directory tree structure with optional filtering and sorting."""
    if max_level < 0:
        return

    try:
        entries = list(os.scandir(path))
    except PermissionError:
        print(f"{prefix}[Permission Denied]")
        return

    if dirs_only:
        entries = [e for e in entries if e.is_dir()]

    # Sorting logic
    if sort_by_date:
        entries.sort(key=lambda e: e.stat().st_mtime, reverse=True)  # Newest first
    else:
        # Dirs first, then files, alphabetical
        entries.sort(key=lambda e: (not e.is_dir(), e.name.lower()) if not dirs_only else e.name.lower())

    for index, entry in enumerate(entries):
        connector = "└── " if index == len(entries) - 1 else "├── "
        print(f"{prefix}{connector}{entry.name}")
        if entry.is_dir():
            next_prefix = prefix + ("    " if index == len(entries) - 1 else "│   ")
            tree(entry.path, max_level - 1, next_prefix, dirs_only, sort_by_date)

def find_files_sorted(root, extension='.h5', max_depth=None):
    root_depth = root.rstrip(os.sep).count(os.sep)
    lfiles = []
    for folder, subfolders, files in os.walk(root):
        current_depth = folder.rstrip(os.sep).count(os.sep) - root_depth
        if max_depth is not None and current_depth >= max_depth:
            # Prevent walking deeper by clearing subfolders list
            subfolders[:] = []
        for file in files:
            if file.endswith(extension):
                full_path = os.path.join(folder, file)
                try:
                    mod_time = os.path.getmtime(full_path)
                    lfiles.append((full_path, mod_time))
                except OSError:
                    pass  # Skip unreadable files
    # Sort by modification time (newest first)
    lfiles.sort(key=lambda x: x[1], reverse=True)
    return [path for path, _ in lfiles]


def findmasterh5file(folder, max_depth=0):
    lfiles = find_files_sorted(folder, '.h5', max_depth=max_depth)
    if len(lfiles)>0:
        return lfiles[0], True
    else:
        return None, False
    