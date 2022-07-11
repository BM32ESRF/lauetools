# SPEC and BLISS reader
# initially 2016-11-23
# Sam Tardif (samuel.tardif@gmail.com)
# Python3 compatibility by Nils Blanc
# Modified to store scan lines by L. Renversade (2018)
# added functions to read hdf5 file by J.S. Micha (2022)

""" this file in intended to be name logfilereader.py and be located in main lauetools folder"""

import copy
import os
import time
import numpy as np

try:
    import h5py
except ModuleNotFoundError:
    print('warning: h5py is missing. It is useful for playing with hdf5 for some LaueTools modules. Install it with pip> pip install h5py')

def ReadSpec(fname, scan, outputdate=False):
    """
    Procedure very based on that of Vincent Favre-Nicolin (ESRF) procedure

    :param scan: scan index (integer)
    return :
    spec command (str), dict of data (key=counter name, val= values), and optionnaly [date (str)]
    """
    f = open(fname, "r")

    print('ReadSpec of IOLaueTools.py')

    s = "#S %d" % scan
    title = 0

    bigmca = []
    # spec command with motors, steps and exposure
    while 1:
        title = f.readline()
        if s == title[0 : len(s)]:
            break
        if len(title) == 0:
            break
    print(title)
    # date -----------------
    date = 'unknown'
    s = "#D"
    while 1:
        line = f.readline()
        if s == line[0:len(s)]:
            date = line[3:]
            break
        if len(line) == 0:
            break
    # data   (dict with key= counter column name) -------------
    s = "#L"
    coltit = 0

    while 1:
        coltit = f.readline()
        if s == coltit[0 : len(s)]:
            break
        if len(coltit) == 0:
            break
    d = {}
    coltit = coltit.split()
    for i in list(range(1, len(coltit))):
        d[coltit[i]] = []

    ii = 0
    while 1:  # reading data
        l = f.readline()
        if len(l) < 2:
            break

        if l[:2] == "#C":            
            # deal with#C Thu Feb 25 23:54:05 2021.  Erreur com with laueT.
            if 'Erreur com' in l:
                print('line error #C :', l)
                continue
            elif not l.startswith("#C tiltcomp:"):
                print("Scan aborted after %d point(s)" % ii)
                break
            else:
                print(l)
        if l[0] == "#":
            continue
        l = l.split()
        # print "l",l
        #         print "coltit", coltit
        # print "nb columns",len(coltit)-1
        if l[0] != "@A":
            for i in list(range(1, len(coltit))):
                d[coltit[i]].append(float(l[i - 1]))
        else:
            # print "reading mca data array for one point"
            bill = np.zeros((128, 16))  # 2048=128*16
            mcadata = []
            l[-1] = l[-1][:-1]
            mcadata.append(np.array(l[1:]))
            bill[0] = np.array(np.array(l[1:]), dtype=np.int16)
            # fist line has its first element = '@A'
            if ii % 10 == 0:
                print("%d" % ii)
            for k in list(range(1, 127)):  # first and last line off , each line contains 16 integers
                l = f.readline()

                l = l.split()
                l[-1] = l[-1][:-1]
                mcadata.append(np.array(l[:16]))
                # print "uihuihui ",k,"   ",array(l)
                bill[k] = np.array(np.array(l[:16]), dtype=np.int16)
            # last line doesn't finish with \
            l = f.readline()
            l = l.split()
            # print array(l)
            mcadata.append(np.array(l))
            bill[-1] = np.array(np.array(l), dtype=np.int16)

            # bill=array(mcadata,dtype=uint16)
            bigmca.append(np.ravel(bill))
            ii += 1

            d["mca"] = np.array(bigmca)

    nb = len(d[coltit[1]])  # nb of points
    # print "nb",nb
    for i in list(range(1, len(coltit))):
        a = np.zeros(nb, dtype=float)
        for j in list(range(nb)):
            a[j] = d[coltit[i]][j]
        d[coltit[i]] = copy.deepcopy(a)
    f.close()
    if outputdate:
        return title, d, date
    else:
        return title, d

class SpecFile:
    """
    Simple class to read SPEC file. The main purposes are :
    (i) read the header and initiate the relevant properties
    (ii) build a dictionary of the scan number and their binary position in the file
    The latter allows for faster subsequent scan reading
    Definition:
    -----------
    SpecFile(spec_file, verbose = False)
     > spec_file : string
     > scan_numbers : integer or as tuple/list/array of integer
    Attributes (typical):
    -----------
    comments........all comments
    date............scan start datestamp
    file............string of the SPEC file name
    scan_dict.......dictionary {scan number : binary position in file}
    Examples:
    --------
    # read the specfile
    In : sf = SpecFile('./lineup0.dat')

    author: S. Tardif
    """

    # dictionary definitions for handling the spec identifiers
    def __param__(self):
        return {'S': self.__scanline__,
                'D': self.__dating__,
                'T': self.__counting__,
                'G': self.__configurating__,
                'Q': self.__hkl__,
                'O': self.__motorslabeling__,
                'o': self.__motorslabelingnospace__,
                'P': self.__positioning__,
                'M': self.__marccdpath__,
                'N': self.__speccol__,
                'L': self.__counterslabeling__,
                'C': self.__commenting__,
                '@': self.__special__,
                'XIAFILE': self.__xiafilenaming__,
                'XIACALIB': self.__xiacalibrating__,
                'XIAROI': self.__xiaroidefining__}

    def __scanline__(self, l):
        self.number = int(l.split()[1])
        self.type = l.split()[2]
        self.args = l.split()[3:]
        self.command = l[3 + len(l.split()[1]) + 2:].rstrip()

    def __dating__(self, l):
        self.date = l[3:]

    def __counting__(self, l):
        self.ct = float(l.split()[1])
        self.ct_units = l.split()[2]

    def __configurating__(self, l):
        self.__config__ = self.__config__ + l[3:] + " "

    def __hkl__(self, l):
        self.Qo = l[3:].split()

    def __motorslabeling__(self, l):
        self.__motorslabels__ = self.__motorslabels__ + l[3:] + " "

    def __motorslabelingnospace__(self, l):
        self.__motorslabelsnospace__ = self.__motorslabelsnospace__ + l[3:] + " "

    def __positioning__(self, l):
        self.__positions__ = self.__positions__ + l[3:] + " "

    def __speccol__(self, l):
        self.N = int(l[3:])

    def __counterslabeling__(self, l):
        self.counters = l.split()[1:]

    def __commenting__(self, l):
        self.comments = self.comments + l[3:]

    def __marccdpath__(self, l):
        self.M = l.split()[1:]

    def __special__(self, l):
        self.__param__()[l[2:].split(' ')[0]](l)

    def __xiafilenaming__(self, l):
        self.xianame = l.split()[1:][0]
        if not hasattr(self, "xiaroi"):
            self.xiaroi = dict()

    def __xiacalibrating__(self, l):
        self.xiacalib = l.split()[1:][0]

    def __xiaroidefining__(self, l):
        if not hasattr(self, "xiaroi"):
            self.xiaroi = dict()
        self.xiaroi[l.split()[1]] = [int(l.split()[2]), int(l.split()[3]), int(l.split()[4]), int(l.split()[5]),
                                     int(l.split()[6])]

    def __init__(self, spec_file, verbose=False, filetype='spec', collectallscans=True,
                                                                onlywirescan=False,onlywire=False):
        # init a bunch of stuff that will also be used by the children class Scan
        self.file = spec_file

        self.filetype = filetype # 'spec'  # 'hdf5'

        self.__motorslabels__ = ""  # list of all motors in the experiment
        self.__motorslabelsnospace__ = ""  # list of all motors in the experiment
        self.__positions__ = ""  # list the values of all motors
        self.__config__ = ""  # list the values of the UB matrix config
        self.comments = ""
        # next attributes come with the reading of the file
        self.scan_dict = {}  # dictionary to store the position in the file of the scans
        self.cmd_list =  {}  # list to store scan line strings

        if filetype == 'spec':
            try:
                with open(spec_file,
                        'rU') as f:  # the U mode indicates universal line break, essential for accurate counting
                    # first read the file header (mostly comments and motors definition)
                    # up to the first scan (identified by a line starting with "#S"
                    reading_header = True
                    while reading_header:
                        position_in_file = f.tell()  # get the position AHEAD of the scan first line
                        l = f.readline()
                        if len(l) > 1:  # not an empty line
                            if l[:2] == '#S':
                                reading_header = False
                                scan_number = int(l.split()[1])
                                if verbose:
                                    print("after reading the header, found scan {} at location {}".format(scan_number,
                                                                                                        position_in_file))
                                self.scan_dict[scan_number] = position_in_file
                                self.cmd_list[scan_number] = l[3:-1]
                            else:
                                try:
                                    self.__param__()[l[1]](l)
                                except KeyError:
                                    if verbose: print("unprocessed line:" + l)
                    reading_file = True  # change this switch when the end of file is reached (i.e. read an empty string)
                    while reading_file:
                        position_in_file = f.tell()  # get the position AHEAD of the scan first line
                        l = f.readline()
                        if len(l) > 1:  # not an empty line
                            if l[:2] == '#S':
                                scan_number = int(l.split()[1])
                                if verbose:
                                    print("found scan {} at location {}".format(scan_number, position_in_file))
                                self.scan_dict[scan_number] = position_in_file
                                self.cmd_list[scan_number] = l[3:-1]
                        if l == "":
                            reading_file = False
            except IOError:
                print("could not find the file {}".format(spec_file))
        elif filetype == 'hdf5':
            selp, allp = getwirescan_from_hdf5file(spec_file, collectallscans=collectallscans,
                                                            onlywirescan=onlywirescan, onlymesh=onlywirescan,
                                                            verbose=0)
            self.scan_dict = get_scandict(selp)
            self.cmd_list = get_cmd_list(selp)


def get_cmd_list(listprops):
    """ return dict of commands in the list (generated by getwirescan_from_hdf5file)"""
    return { elem[0]: elem[4] for elem in listprops}
def get_scandict(listprops):
    """ return dict of keys ... (only to fit the previous spec-related interface)
    (generated by getwirescan_from_hdf5file)"""
    return { elem[0]: elem[0] for elem in listprops}
def get_file(listprops):
    """ return master file corresponding to listprops (generated by getwirescan_from_hdf5file)"""
    return listprops[0][-1]

def get_date(listprops):
    """ return date of file corresponding to listprops (generated by getwirescan_from_hdf5file)"""
    return listprops[0][3]

def getmeshscan_from_hdf5file(filename):

    print("getmeshscan_from_hdf5file  %s"%filename)
    _,ext = filename.rsplit('.',1)
    headname, ffname = os.path.split(filename)

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    with h5py.File(filename, 'r') as f:

        listkeys = [kk for kk in f.keys()]
        print('hdf5 keys',listkeys)
        nbkeys = len(listkeys)
        
        # if key =   #########_int.int  then it is a pointer to a file ########.h5
        # if key =   int.int    this file contains truly the data
        #  #########   =  collectionname_datasetname
        listprops = []  # selected props
        allprops = []

        idx_key=0
        while idx_key<nbkeys:
            _key = listkeys[idx_key]
            objlink = f.get(_key, getlink=True)
            props = None
            if isinstance(objlink, h5py._hl.group.ExternalLink):
                
                lowlevelpath = objlink.filename
                print('key = %s is External link to %s'%(_key,lowlevelpath))
                foundfile = findlowesthdf5file(lowlevelpath,mainfolder=headname)
                if foundfile:
                    # removing string before _interger.integer
                    _modified_key = _key.rsplit('_',1)[-1]
                    props, isselected = getscanprops_lowest_hdf5(foundfile, _modified_key, collectallscans=False, onlymesh=True)
            elif isinstance(objlink, h5py._hl.group.HardLink):
                print('key = %s is Hard link to '%(_key))
                props, isselected = getscanprops_lowest_hdf5(filename, _key, collectallscans=False, onlymesh=True)
            
            allprops.append(props)

            if isselected:
                listprops.append(props)
            idx_key+=1

    # print('listprops',listprops)
    if listprops == []:
        #wx.MessageBox('No mesh scan in the file: %s'%filename,'INFO')
        print('\n\n*******\nNo mesh scan in the file: %s\n**********\n'%filename)
        return []

    # sorting by increasing date
    ar_lp = np.array(listprops, dtype=object)
    s_ix=np.argsort(ar_lp[:,3])
    sortedlistprops = ar_lp[s_ix]
    #print('sortedlistprops',sortedlistprops)
    
    return sortedlistprops

def getwirescan_from_hdf5file(filename, verbose=0, collectallscans=True, onlywirescan=False, onlymesh=False):

    if collectallscans:
        onlymesh=True
        onlywirescan=True

    if verbose: print("getwirescan_from_hdf5file  %s"%filename)
    _,ext = filename.rsplit('.',1)
    headname, ffname = os.path.split(filename)

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    with h5py.File(filename, 'r') as f:

        listkeys = [kk for kk in f.keys()]
        if verbose: print('filename: %s \n hdf5 keys:'%filename, listkeys)
        nbkeys = len(listkeys)
        
        # if key =   #########_int.int  then it is a pointer to a file ########.h5
        # if key =   int.int    this file contains truly the data
        #  #########   =  collectionname_datasetname
        listprops = []   # selected props
        allprops = []
        idx_key=0
        while idx_key<nbkeys:
            _key = listkeys[idx_key]
            objlink = f.get(_key, getlink=True)
            props = None
            isselected = False
            if isinstance(objlink, h5py._hl.group.ExternalLink):
                
                lowlevelpath = objlink.filename
                if verbose: print('key = %s is External link to %s'%(_key,lowlevelpath))
                foundfile = findlowesthdf5file(lowlevelpath,mainfolder=headname)
                if foundfile:
                    # removing string before _interger.integer
                    _modified_key = _key.rsplit('_',1)[-1]
                    props, isselected = getscanprops_lowest_hdf5(foundfile, _modified_key,collectallscans=collectallscans,
                                                                onlymesh=onlymesh, onlywirescan=onlywirescan)
            elif isinstance(objlink, h5py._hl.group.HardLink):
                if verbose: print('key = %s is Hard link to '%(_key))
                props, isselected = getscanprops_lowest_hdf5(filename, _key,collectallscans=collectallscans,
                                                                onlymesh=onlymesh, onlywirescan=onlywirescan)
            if props is not None:
                allprops.append(props)
            if isselected:
                listprops.append(props)
            idx_key+=1

    #print('allprops',allprops)
    if listprops == []:
        #wx.MessageBox('No mesh scan in the file: %s'%filename,'INFO')
        print('\n\n*******\nNo wire scan in the file: %s\n**********\n'%filename)
        return []

    # sorting by increasing date
    ar_lp = np.array(listprops, dtype=object)
    s_ix=np.argsort(ar_lp[:,3])
    sortedlistprops = ar_lp[s_ix]
    #print('sortedlistprops',sortedlistprops)

    # sorting by increasing date
    ar_lpall = np.array(allprops, dtype=object)
    s_ix2=np.argsort(ar_lpall[:,3])
    sortedlistallprops = ar_lpall[s_ix2]
    #print('sortedlistprops',sortedlistprops)
    
    return sortedlistprops, sortedlistallprops


def findlowesthdf5file(filename, mainfolder='.', verbose=0):
    """ find the lowest hdf5 file pointing
    To be improved to consider relative path
    """
    
    foundfile = None
    _, ffname = os.path.split(filename)
    absfolder = os.path.abspath(mainfolder)
    relativepath = os.path.join(absfolder,filename)
    if verbose:
        print('filename',filename)
        print('relativepath', relativepath)
        print('ffname',ffname)
        print('absfolder',absfolder)
    if os.path.exists(relativepath): # relative path from current folder
        foundfile = relativepath
    elif ffname in os.listdir(absfolder): #local folder
        foundfile = os.path.join(absfolder,ffname)
    if verbose:
        print('foundfile',foundfile)
    return foundfile

def getscanprops_lowest_hdf5(filename, key, collectallscans=True, onlymesh=False, onlywirescan=False, verbose=0):
    """ get scan props from hdf5 file and filter optionally wrt scan type (mesh or wirescan)"""
    #print('\n\nterminal hdf5 file')
    _,ext = filename.rsplit('.',1)
    headname, ffname = os.path.split(filename)

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    with h5py.File(filename, 'r') as f:

        idx, postfix = key.split('.')
        #print('reading key %s'%key)
        if h5py.__version__<'3.0':
            #maybe [()] is enough without decoding
            scancommand = f['%s.%s'%(idx,postfix)]['title'].value
            startdate = f['%s.%s'%(idx,postfix)]['start_time'].value
        else:
            #print('%s.%s'%(idx,postfix))
            scancommand = f['%s.%s'%(idx,postfix)]['title'][()].decode('UTF-8')
            startdate = f['%s.%s'%(idx,postfix)]['start_time'][()].decode('UTF-8')
        props = None

        if verbose:
            print('scancommand',scancommand)
            print('startdate',startdate)
        isselected = False
        if onlymesh:
            if 'amesh' in scancommand:
                keyfilename = ffname[:-3]  #  removing .h5
                props=['%s_%s'%(keyfilename,idx),idx, postfix, startdate, '%s_%s %s'%(keyfilename, idx, scancommand), filename]
                isselected=True
        if onlywirescan:
            if 'zf' in scancommand or 'yf' in scancommand:
                keyfilename = ffname[:-3]  #  removing .h5
                props=['%s_%s'%(keyfilename,idx),idx, postfix, startdate, '%s_%s %s'%(keyfilename, idx, scancommand), filename]
                isselected=True
        if collectallscans:
            if 'loopscan' in scancommand or 'ascan' in scancommand or 'a2scan' in scancommand:
                keyfilename = ffname[:-3]  #  removing .h5
                props=['%s_%s'%(keyfilename,idx),idx, postfix, startdate, '%s_%s %s'%(keyfilename, idx, scancommand), filename]
                isselected=True
    return props, isselected

class Scan(SpecFile):
    """
    Simple class to read extract scans from SPEC files. All the parameters of the scan and the data are read
    and stored as attributes.

    Definition:
    -----------
    Scan(spec_file, scan_numbers, verbose = False)
     > spec_file as SpecFile object or string (in this case a SpecFile object will be instanced)
     > scan_numbers as integer or as tuple/list/array of integer

    Attributes:
    -----------
    <countername>...data in counter <countername> (see counters for description)
    number..........scan number of the first scan in the list
    scan_numbers....list of all the scans included
    type............scan type
    args............scan arguments (motor start stop npoints counting)
    date............scan start datestamp
    ct..............scan counting time per point
    Qo..............H K L position at start of scan
    M...............MarCCD image file path
    N...............number of counters
    counters........list of counters
    motors..........dictionary of motors and their initial position
    comments........all comments
    tstart, tend....starting and finishing time
    duration........duration in s
    time_per_point..duration per point
    Examples:
    --------
    # read the specfile
    In : sf = SpecFile('./lineup0.dat')
    # read the scan
    In : scan = Scan(sf, 265)

    # read a series of scans
    In : scan = Scan(sf, (265,266,270))
    In : scan = Scan(sf, arange(265,270))

    # learn about the motors
    In : scan.motors
    Out:
    {'Chi': -144.8913,
     'Phi': 176.9778,
     ...
     'xpr3z': -8.8}

    # extract the info on a particular motor
    In : scan.motors['tth']
    Out: 35.042

    # plot two counters vs each others
    In : plot(scan.th,scan.det/scan.IC1)

    author: L. Renversade
    """

    def __init__(self, spec_file, scan_numbers, verbose=False, filetype='spec'):
        if type(spec_file) == str:
            spec_file = SpecFile(spec_file)
        self.file = spec_file.file
        try:
            len_scan_number = len(scan_numbers)  # it is a list of scan
        except TypeError:
            scan_numbers = [scan_numbers, ]  # it is a simple scan, we make a len 0 list
        scan_number = scan_numbers[0]  # for all intents and purposes
        # recover the names of the motors from the header of the SPEC file (i.e. the SpecFile instance)
        self.__motorslabels__ = spec_file.__motorslabels__  # list of all motors in the experiment
        self.__motorslabelsnospace__ = spec_file.__motorslabelsnospace__  # list of all motors in the experiment
        # prepare the scan-specific attributes
        self.__positions__ = ""  # list the values of all motors
        self.__config__ = ""  # list the values of the UB matrix config
        self.scan_numbers = scan_numbers
        self.comments = ""

        with open(self.file, 'rU') as f:
            # read the first (and possibly only) scan in the list
            # now try to find the scan
            f.seek(spec_file.scan_dict[scan_number])
            l = f.readline()
            if verbose: print("reading scan " + l)

            # read the scan header
            while l[0] == '#':
                try:
                    self.__param__()[l[1]](l)
                except KeyError:
                    if verbose: print("unprocessed line:\n" + l)
                l = f.readline()

            # finally read the data (comments at the end are also read and added to the comment attribute)
            data = [list(map(float, l.split()))]
            l = f.readline()
            while l != '\n' and l != '':
                if l[0] == '#' and l[1] != 'C':
                    break
                if l[0] == '#':
                    try:
                        self.__param__()[l[1]](l)
                    except KeyError:
                        if verbose: print("unprocessed line:\n" + l)
                else:
                    data.append(list(map(float, l.split())))
                l = f.readline()

            # now get the data for each scan in the list
            if len(scan_numbers) > 0:
                for scan_number in scan_numbers[1:]:
                    # now try to find the scan
                    f.seek(spec_file.scan_dict[scan_number])
                    l = f.readline()
                    if verbose: print("reading scan " + l)

                    # check that we actually concatenate similar scans !
                    similar_scan = (l.split()[2] == self.type)
                    if len(self.args) > 1:
                        try:
                            similar_scan = similar_scan * (l.split()[3] == self.args[0])
                        except ValueError:
                            similar_scan = False
                    if len(self.args) > 5:
                        try:
                            similar_scan = similar_scan * (l.split()[6] == self.args[3])
                        except ValueError:
                            similar_scan = False

                    if similar_scan:
                        # read pass the scan header
                        while l[0] == '#':
                            l = f.readline()

                        # finally read the data (comments at the end are also read and added to the comment attribute)
                        data.append(list(map(float, l.split())))
                        l = f.readline()
                        while l != '\n' and l != '':
                            if l[0] == '#' and l[1] != 'C':
                                break
                            if l[0] == '#':
                                try:
                                    self.__param__()[l[1]](l)
                                except KeyError:
                                    if verbose: print("unprocessed line:\n" + l)
                            else:
                                data.append(list(map(float, l.split())))
                            l = f.readline()
                    else:
                        print("not all scans are the same type")

            # set the data as attributes with the counter name
            for i in range(len(self.counters)):
                # print(i,range(len(self.counters)),self, self.counters[i], np.asarray(data).shape, np.asarray(data), data)
                setattr(self, self.counters[i], np.asarray(data)[:, i])

            # make the motors/positions dictionary
            # usual case
            if len(self.__motorslabels__.split()) == len(self.__positions__.split()):
                self.motors = dict(zip(self.__motorslabels__.split(), list(map(float, self.__positions__.split()))))
            # when some motors names have spaces and there is a second line (small o) to describe them
            elif len(self.__motorslabelsnospace__.split()) == len(self.__positions__.split()):
                self.motors = dict(
                    zip(self.__motorslabelsnospace__.split(), list(map(float, self.__positions__.split()))))

            # TEST : attribute-like dictionary
            # removed due to conflicts when a motor was also a counter
            #		for motor in self.motors:
            #								setattr(self, motor, self.motors[motor])

            # small sanity check, sometimes N is diffrent from the actual number of columns
            # which is known to trouble GUIs like Newplot and PyMCA
            if self.N != len(self.counters):
                print("Watch out! There are %i counters in the scan but SPEC knows only N = %i !!" % (
                len(self.counters), self.N))

            if hasattr(self, 'Epoch'):
                self.tstart = self.Epoch[0]
                self.tend = self.Epoch[-1]
                self.duration = self.tend - self.tstart
                self.time_per_point = self.duration / len(self.Epoch)
# class Scan2D(Scan):
# def __init__(self, spec_file, scan_number_list, verbose = verbose):


class Scan_hdf5(SpecFile):
    """
    Simple class to read extract scans from bliss files. All the parameters of the scan and the data are read
    and stored as attributes.

    Definition:
    -----------
    Scan(spec_file, scan_numbers, verbose = False)
     > hdf5 file (string) created by BLISS
     > scan_keys   string  WARNING!  single key for the moment!

    # not up to date!
    
    Attributes:
    -----------
    <countername>...data in counter <countername> (see counters for description)
    number..........scan number of the first scan in the list
    scan_numbers....list of all the scans included
    type............scan type
    args............scan arguments (motor start stop npoints counting)
    date............scan start datestamp
    ct..............scan counting time per point
    Qo..............H K L position at start of scan
    M...............MarCCD image file path
    N...............number of counters
    counters........list of counters
    motors..........dictionary of motors and their initial position
    comments........all comments
    tstart, tend....starting and finishing time
    duration........duration in s
    time_per_point..duration per point
    Examples:
    --------
    # read the specfile
    In : sf = SpecFile('./lineup0.dat')
    # read the scan
    In : scan = Scan(sf, 265)

    # read a series of scans
    In : scan = Scan(sf, (265,266,270))
    In : scan = Scan(sf, arange(265,270))

    # learn about the motors
    In : scan.motors
    Out:
    {'Chi': -144.8913,
     'Phi': 176.9778,
     ...
     'xpr3z': -8.8}

    # extract the info on a particular motor
    In : scan.motors['tth']
    Out: 35.042

    # plot two counters vs each others
    In : plot(scan.th,scan.det/scan.IC1)

    author: J.S. Micha
    """
    def __init__(self, spec_file, scan_keys,collectallscans=True,
                    onlywirescan=False, onlymesh=False, verbose=False):
        if type(spec_file) == str:
            spec_file = SpecFile(spec_file, filetype='hdf5')
        self.file = spec_file.file
        try:  # WARNING single key for the moment
            len_scan_number = len(scan_keys)  # it is a list of scan
        except TypeError:
            scan_keys = [scan_keys, ]  # it is a simple scan, we make a len 0 list
        scan_number = scan_keys[0]  # for all intents and purposes
        # recover the names of the motors from the header of the SPEC file (i.e. the SpecFile instance)
        self.__motorslabels__ = spec_file.__motorslabels__  # list of all motors in the experiment
        self.__motorslabelsnospace__ = spec_file.__motorslabelsnospace__  # list of all motors in the experiment
        # prepare the scan-specific attributes
        self.__positions__ = ""  # list the values of all motors
        self.__config__ = ""  # list the values of the UB matrix config
        self.scan_keys = scan_keys
        
        self.comments = ""

        print('beginning of __init__ of scan_hdf5')
        selp, allp = getwirescan_from_hdf5file(spec_file.file, onlywirescan=onlywirescan, collectallscans=collectallscans, onlymesh=onlymesh)
        tit, data, posmotors, fullpath, scan_date= readdata_from_hdf5key(selp, scan_keys, outputdate=True)

        if verbose:
            print('tit',tit)
            print('posmotors',posmotors)
            print('scan_date',scan_date)
            print('data',data)
            print('fullpath', fullpath)
        # set the data as attributes with the counter name
        for key,val in data.items():

            setattr(self, key, val)

        self.fullpath = fullpath
        
        scanindex= int(scan_keys.rsplit('_',1)[-1])
        basepath = os.path.split(fullpath)[0]
        self.scanfolder = os.path.join(basepath, 'scan_%04d'%scanindex)
        self.motors = posmotors
        self.command = tit
        self.scan_date = scan_date

        # small sanity check, sometimes N is diffrent from the actual number of columns
        # which is known to trouble GUIs like Newplot and PyMCA
        # if self.N != len(self.counters):
        #     print("Watch out! There are %i counters in the scan but SPEC knows only N = %i !!" % (
        #     len(self.counters), self.N))

        if hasattr(self, 'epoch'):
            self.tstart = self.epoch[0]
            self.tend = self.epoch[-1]
            self.duration = self.tend - self.tstart
            self.time_per_point = self.duration / len(self.epoch)


def parsehdf5time(strtime):
    """ parse time in hdf5 from bliss file
    Return  ascii time, epoch time"""
    sd, _ = strtime.split('+')
    da, ho = sd.split('T')
    yy, mm, dd = da.split('-')
    hh, mi, sec = ho.split(':')
    starttime = time.strptime('%s %s %s %s:%s:%s'%(yy, mm, dd, hh, mi, sec[:2]), '%Y %m %d %H:%M:%S')
    return time.asctime(starttime), time.mktime(starttime)

def ReadHdf5_v2(fname, scan, outputdate=False):
    """extract data of a scan in a ESRF Bliss made hdf5 file

    NOT USED?

    :param fname: file object or string (path)
    :type fname: file object or string
    :param scan: scan index
    :type scan: integer
    :param outputdate: output starting date of the scan in ascii format, defaults to False
    :type outputdate: bool, optional
    """
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    with h5py.File(fname, 'r') as f:

        i_scan = '%d' % scan + '.1'
        print('i_scan', i_scan)
        print('fname', f)
        print('fname[i_scan]', f[i_scan])
        datasetnames = [key for key in f[i_scan]['measurement'].keys()]
        d = {}
        for _n in datasetnames:
            d[_n] = f[i_scan]['measurement'][_n][()]

        if h5py.__version__>='3':
            title = f[i_scan]['title'][()].decode('UTF-8')
            st = f[i_scan]['start_time'][()].decode('UTF-8')
            et = f[i_scan]['end_time'][()].decode('UTF-8')
        else:
            title = f[i_scan]['title'].value
            st = f[i_scan]['start_time'].value
            et = f[i_scan]['end_time'].value
        duration = parsehdf5time(et)[1] - parsehdf5time(st)[1]
        
        print('title', title)
        print('duration', duration)
        
        if not outputdate:
            return title, d
        else:
            return title, d, st

def readdata_from_hdf5key(listkeyprops, key, outputdate=False):
    """access to a low level hdf5 file and read data corresponding to the key 

    :param listkeyprops: array of strings which lists all the keys (see getmeshscan_from_hdf5file() of plotmeshspecGUI.py). Each element is:
    props=['%s_%s'%(keyfilename,idx),idx, postfix, startdate, '%s_%s %s'%(keyfilename, idx, scancommand), filename]
    :type fname: file object or string
    :param key: scan id  (key in hdf5 file)
    :type key: string
    :param outputdate: output starting date of the scan in ascii format, defaults to False
    :type outputdate: bool, optional
    """
    # print('key',key)
    # print('listkeyprops[:,0]',listkeyprops[:,0])

    _ix = np.where(listkeyprops[:,0]==key)[0]

    # print('_ix',_ix)
    # print('listkeyprops[_ix]', listkeyprops[_ix])

    if len(_ix)==0:
        title ='Not Reachable'
        d={}
        st =''
    else:
        assert len(_ix)==1
        general_id, idx, postfix, startdate, commandtitle, fullpath = listkeyprops[_ix[0]]
        
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        with h5py.File(fullpath, 'r') as f:

            i_scan = '%s.%s'%(idx, postfix)
            # print('i_scan', i_scan)
            # print('fname', f)
            # print('fname[i_scan]', f[i_scan])
            datasetnames = [key for key in f[i_scan]['measurement'].keys()]
            d = {}
            for _n in datasetnames:
                d[_n] = f[i_scan]['measurement'][_n][()]
            
            if h5py.__version__>='3':
                title = f[i_scan]['title'][()].decode('UTF-8')
                st = f[i_scan]['start_time'][()].decode('UTF-8')
                et = f[i_scan]['end_time'][()].decode('UTF-8')
            else:
                title = f[i_scan]['title'].value
                st = f[i_scan]['start_time'].value
                et = f[i_scan]['end_time'].value
            duration = parsehdf5time(et)[1] - parsehdf5time(st)[1]

            positionersnames = [key for key in f[i_scan]['instrument']['positioners'].keys()]
            posmotors={}
            for _n in positionersnames:
                posmotors[_n] = f[i_scan]['instrument']['positioners'][_n][()]
            
            print('title', title)
            print('duration', duration)
            
        if not outputdate:
            return title, d, posmotors, fullpath
        else:
            return title, d, posmotors, fullpath, st


def ReadHdf5(fname, scan, outputdate=False):
    """extract data of a scan in a ESRF Bliss made hdf5 file when key are formatted as int.1

    :param fname: file object or string (path)
    :type fname: file object or string
    :param scan: string for scan id  (key in hdf5 file)
    :type scan: integer
    :param outputdate: output starting date of the scan in ascii format, defaults to False
    :type outputdate: bool, optional
    """
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    with h5py.File(fname, 'r') as f:

        i_scan = '%d' % scan + '.1'
        print('i_scan', i_scan)
        print('fname', f)
        print('fname[i_scan]', f[i_scan])
        datasetnames = [key for key in f[i_scan]['measurement'].keys()]
        d = {}
        for _n in datasetnames:
            d[_n] = f[i_scan]['measurement'][_n][()]

        if h5py.__version__>='3':
            title = f[i_scan]['title'][()].decode('UTF-8')
            st = f[i_scan]['start_time'][()].decode('UTF-8')
            et = f[i_scan]['end_time'][()].decode('UTF-8')
        else:
            title = f[i_scan]['title'].value
            st = f[i_scan]['start_time'].value
            et = f[i_scan]['end_time'].value
        duration = parsehdf5time(et)[1] - parsehdf5time(st)[1]
        
        print('title', title)
        print('duration', duration)
        
        if not outputdate:
            return title, d
        else:
            return title, d, st