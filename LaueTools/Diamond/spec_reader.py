# -*- coding: utf-8 -*-

# SPEC reader
# 2016-11-23
# Sam Tardif (samuel.tardif@gmail.com)
# Python3 compatibility by Nils Blanc
# spec_reader.py

import numpy as np

class SpecFile:
    """
    better use  versions of  spec_reader or logfile_reader

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

    """
    # dictionary definitions for handling the spec identifiers
    def __param__(self):
        return  {'S' : self.__scanline__,
                  'D' : self.__dating__,
                  'T' : self.__counting__,
                  'G' : self.__configurating__,
                  'Q' : self.__hkl__,
                  'O' : self.__motorslabeling__,
                  'o' : self.__motorslabelingnospace__,
                  'P' : self.__positioning__,
                  'M' : self.__marccdpath__,
                  'N' : self.__speccol__,
                  'L' : self.__counterslabeling__,
                  'C' : self.__commenting__,
                  '@' : self.__special__,
                  'XIAFILE' : self.__xiafilenaming__,
                  'XIACALIB' : self.__xiacalibrating__,
                  'XIAROI' : self.__xiaroidefining__
                  }

    first_time = 1
    def __scanline__(self, l):
        self.number = int(l.split()[1])
        self.type = l.split()[2]
        self.args = l.split()[3:]
        self.command = l[3+len(l.split()[1])+2:].rstrip()

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
        print("__xiafilenaming__")
        #    print l
        self.xianame = l.split()[1:][0]
        self.xiaroi = dict()

    def __xiacalibrating__(self, l):
        #    print l
        self.xiacalib = l.split()[1:][0]

    def __xiaroidefining__(self, l):
        if not hasattr(self, "xiaroi"):
            self.xiaroi = dict()
        #    print l
        self.xiaroi[l.split()[1]] = [int(l.split()[2]), int(l.split()[3]), int(l.split()[4]), int(l.split()[5]), int(l.split()[6])]

    def __init__(self, spec_file, verbose=False):
        # init a bunch of stuff that will also be used by the children class Scan
        self.file = spec_file
        self.__motorslabels__ = "" # list of all motors in the experiment
        self.__motorslabelsnospace__ = "" # list of all motors in the experiment
        self.__positions__ = "" # list the values of all motors
        self.__config__ = "" # list the values of the UB matrix config
        self.comments = ""
        self.scan_dict = {}  # dictionary to store the position in the file of the scans

        try:
            with open(spec_file, 'rU') as f:  # the U mode indicates universal line break, essential for accurate counting
              # first read the file header (mostly comments and motors definition)
              # up to the first scan (identified by a line starting with "#S"
                reading_header = True
                while reading_header:
                    position_in_file = f.tell() # get the position AHEAD of the scan first line
                    l = f.readline()
                    if len(l) > 1: # not an empty line
                        if l[:2] == '#S':
                            reading_header = False
                            scan_number = int(l.split()[1])
                            if verbose:
                                print(("after reading the header, found scan {} at location {}".format(scan_number, position_in_file)))
                            self.scan_dict[scan_number] = position_in_file
                        else:
                            try:
                                self.__param__()[l[1]](l)
                            except KeyError:
                                if verbose:
                                    print(("unprocessed line:" + l))
                reading_file = True  # change this switch when the end of file is reached (i.e. read an empty string)
                while reading_file:
                    position_in_file = f.tell() # get the position AHEAD of the scan first line
                    l = f.readline()
                    if len(l) > 1: # not an empty line
                        if l[:2] == '#S':
                            scan_number = int(l.split()[1])
                            if verbose:
                                print(("found scan {} at location {}".format(scan_number, position_in_file)))
                            self.scan_dict[scan_number] = position_in_file
                    if l == "":
                        reading_file = False
        except IOError:
            print(("could not find the file {}".format(spec_file)))



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

    """
    def __init__(self, spec_file, scan_numbers, verbose=False):
        if type(spec_file) == str:
            spec_file = SpecFile(spec_file)
        self.file = spec_file.file
        try:
            len_scan_number = len(scan_numbers)  # it is a list of scan
        except TypeError:
            scan_numbers = [scan_numbers,]  # it is a simple scan, we make a len 0 list
            scan_number = scan_numbers[0] #for all intents and purposes
        # recover the names of the motors from the header of the SPEC file (i.e. the SpecFile instance)
        self.__motorslabels__ = spec_file.__motorslabels__ # list of all motors in the experiment
        self.__motorslabelsnospace__ = spec_file.__motorslabelsnospace__ # list of all motors in the experiment
        # prepare the scan-specific attributes
        self.__positions__ = "" # list the values of all motors
        self.__config__ = "" # list the values of the UB matrix config
        self.scan_numbers = scan_numbers
        self.comments = ""

        with open(self.file, 'rU') as f:
            # read the first (and possibly only) scan in the list
            # now try to find the scan
            f.seek(spec_file.scan_dict[scan_number])
            l = f.readline()
            if verbose:
                print(("reading scan " + l))

            # read the scan header
            #      print "header = "
            while l[0] == '#':
                try:
                    self.__param__()[l[1]](l)
                except KeyError:
                    if verbose:
                        print(("unprocessed line:\n" + l))

                l = f.readline()
            #        print l,

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
                        if verbose:
                            print(("unprocessed line:\n" + l))
                else:
                    data.append(list(map(float, l.split())))
                l = f.readline()

            # now get the data for each scan in the list
            if len(scan_numbers) > 0:
                for scan_number in scan_numbers[1:]:
                    # now try to find the scan
                    f.seek(spec_file.scan_dict[scan_number])
                    l = f.readline()
                    if verbose:
                        print(("reading scan " + l))

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
                                    if verbose:
                                        print(("unprocessed line:\n" + l))
                            else:
                                data.append(list(map(float, l.split())))
                            l = f.readline()
                    else:
                        print("not all scans are the same type")

            # set the data as attributes with the counter name
            for i in range(len(self.counters)):
                #print(i,range(len(self.counters)),self, self.counters[i], np.asarray(data).shape, np.asarray(data), data)
                setattr(self, self.counters[i], np.asarray(data)[:, i])


            # make the motors/positions dictionary
            # usual case
            if len(self.__motorslabels__.split()) == len(self.__positions__.split()):
                self.motors = dict(list(zip(self.__motorslabels__.split(), list(map(float, self.__positions__.split())))))
            # when some motors names have spaces and there is a second line (small o) to describe them
            elif len(self.__motorslabelsnospace__.split()) == len(self.__positions__.split()):
                self.motors = dict(list(zip(self.__motorslabelsnospace__.split(),
                                        list(map(float, self.__positions__.split())))))

            #TEST : attribute-like dictionary
            # removed due to conflicts when a motor was also a counter
            #    for motor in self.motors:
            #                setattr(self, motor, self.motors[motor])


            # small sanity check, sometimes N is diffrent from the actual number of columns
            # which is known to trouble GUIs like Newplot and PyMCA
            if self.N != len(self.counters):
                print(("Watch out! There are %i counters in the scan but SPEC knows only N = %i !!" % (len(self.counters), self.N)))

            if hasattr(self, 'Epoch'):
                self.tstart = self.Epoch[0]
                self.tend = self.Epoch[-1]
                self.duration = self.tend - self.tstart
                self.time_per_point = self.duration/len(self.Epoch)
#class Scan2D(Scan):
  #def __init__(self, spec_file, scan_number_list, verbose = verbose):
