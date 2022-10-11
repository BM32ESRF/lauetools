# -*- coding: utf-8 -*-
r"""
module of lauetools project to init a subprocess running fit2d peak search

BEFORE using this macro, it is absolutely needed to set the peak search parameters 
in order to save the peak file, threshold values , etc.
TODO: output and input directories are still the same
"""
import subprocess

import sys
import os
from numpy import loadtxt, shape
from time import sleep, localtime, time

if 0:
    """
    filename='TP0010.mccd'
    myprocess = subprocess.Popen('fit2d -dim2048x2048 -com -svar#file_in=%s'%(filename),
    shell=True,stdout=subprocess.PIPE,stdin=subprocess.PIPE,stderr=subprocess.STDOUT)
    """
    myprocess = subprocess.Popen(
        "fit2d -dim2048x2048 -com",
        shell=True,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    myprocess.stdin.write("I ACCEPT" + "\n")

    myprocess.stdin.write("ON-LINE CRYSTALLOGRAPHY" + "\n")

    myprocess.stdin.write("INPUT" + "\n")
    print("input")  # could be very long

    # myprocess.stdin.write('/home/gonio/data/MicroDiffFeb08/TP/TP0010.mccd'+'\n')
    myprocess.stdin.write("TP0010.mccd" + "\n")
    # myprocess.stdin.write('#file_in')

    myprocess.stdin.write("O.K." + "\n")
    myprocess.stdin.write("PEAK SEARCH" + "\n")

    myprocess.stdin.write("OUTPUT FILE" + "\n")
    myprocess.stdin.write("TP0010.pik" + "\n")
    myprocess.stdin.write("O.K." + "\n")
    myprocess.stdin.write("EXIT" + "\n")  # only now file is created ...

    print("Finished! ...")
    myprocess.stdin.flush()
    while True:
        status = myprocess.poll()
        print("status", status)
    # print myprocess.stdout.read()


def findpeak_f2d(filename, dirname=".", fit2ddirectory=None):
    """
    use fit2d as subprocess to return peak list
    """

    try:
        pre, ext = filename.split(".")
    except ValueError:
        return False
    if ext == "mccd":
        print("mccd extension image")
        prefix = pre
    else:
        prefix = pre + ext

    myprocess = subprocess.Popen(
        "fit2d -dim2048x2048 -com",
        shell=True,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    #%(os.path.join(fit2ddirectory, 'fit2d')),

    print("Opening fit2d...")
    myprocess.stdin.write("I ACCEPT" + "\n")

    myprocess.stdin.write("ON-LINE CRYSTALLOGRAPHY" + "\n")

    myprocess.stdin.write("INPUT" + "\n")
    print("input image ... (may take some times)")  # could be very long

    # myprocess.stdin.write('/home/gonio/data/MicroDiffFeb08/TP/TP0010.mccd'+'\n')
    # myprocess.stdin.write('#file_in')

    # myprocess.stdin.write(filename+'\n')
    myprocess.stdin.write(os.path.join(dirname, filename) + "\n")

    if ext != "mccd":
        myprocess.stdin.write("TIFF" + "\n")

    myprocess.stdin.write("O.K." + "\n")
    myprocess.stdin.write("PEAK SEARCH" + "\n")

    myprocess.stdin.write("OUTPUT FILE" + "\n")
    # myprocess.stdin.write('auto_'+prefix+'.peaks'+'\n')
    myprocess.stdin.write(os.path.join(dirname, "auto_" + prefix + ".peaks") + "\n")

    myprocess.stdin.write("O.K." + "\n")
    myprocess.stdin.write("EXIT" + "\n")  # only now file is created ...

    myprocess.stdin.write("EXIT" + "\n")
    myprocess.stdin.write("EXIT FIT2D" + "\n")
    myprocess.stdin.write("YES" + "\n")
    print("fit 2D peak search is finished! ...")
    print("peak list has been written in %s" % ("auto_" + prefix + ".peaks"))
    # myprocess.stdin.flush()

    retcode = None
    i = 0
    while (retcode == None) & (i < 100):
        if i == 1:
            # needed under windows to close the created cmd.exe and fit2d.exe processes
            myprocess.communicate(input="exit \n")
            # returncode =  None if process is still running
            retcode = myprocess.returncode
            print("fit2d return code = ", retcode)
            i = i + 1
        if i > 99:
            # under windows the PID of myprocess is the one of cmd.exe (not the one of fit2d.exe)
            pidcode = myprocess.pid
            print("PID of fit2d command shell", pidcode)

    # myprocess.wait()
    ##    """
    ##    while True:
    ##        status = myprocess.poll()
    ##        print "status",status
    ##    """
    # print myprocess.stdout.read()
    print("end of findpeak_f2d")


def fitpeak_f2d(filename, dirname=".", fit2ddirectory=None):
    """
    use fit2d as subprocess to refine peak positions by 2D fit from centroid peak list
    
    TODO: use fit2ddirectory ?
    """

    try:
        pre, ext = filename.split(".")
    except ValueError:
        return False
    if ext == "mccd":
        print("mccd extension image")
        prefix = pre
    else:
        prefix = pre + ext

        boxsize = 10.0
        startwidth = 1.0

        xy_list = loadtxt(
            "Ge_blanc_11Sep08_d1_5MPa_0000.peaks", usecols=(0, 1), skiprows=1
        )
        # xy_list = loadtxt('auto_'+prefix+'.peaks',usecols=(0,1) )

        nxy = shape(xy_list)[0]
        print("number of peaks in initial list", nxy)
        time_0 = time()

    print("writing macro for fit2d...")
    # l'envoi des commandes une par une avec l'option -com de fit2d marche sur un pic
    # mais pas sur un grand nombre de pics
    # pb : commande precedente pas terminee avant l'envoi de la commande suivante

    linestowrite = []

    linestowrite.append("I ACCEPT" + "\n")
    linestowrite.append("MACROS / LOG FILE" + "\n")
    linestowrite.append("OPEN LOG FILE" + "\n")
    linestowrite.append("NO" + "\n")
    linestowrite.append("toto.log" + "\n")
    linestowrite.append("EXIT" + "\n")
    linestowrite.append("2-D FITTING" + "\n")
    linestowrite.append("INPUT" + "\n")
    linestowrite.append("Ge_blanc_11Sep08_d1_5MPa_0000.mccd" + "\n")
    # linestowrite.append([os.path.join(dirname, filename)+'\n')
    # if ext!='mccd':
    #    linestowrite.append('TIFF'+'\n')
    linestowrite.append("O.K." + "\n")
    linestowrite.append("Z-SCALING" + "\n")
    linestowrite.append("FULLY AUTOMATIC" + "\n")
    linestowrite.append("EXIT" + "\n")
    for i in range(nxy):
        xcen = round(xy_list[i, 0], 1)
        ycen = round(xy_list[i, 1], 1)
        # print "xycen = ", xcen, ycen
        xmin = max(int(xy_list[i, 0] - boxsize), 0)
        xmax = min(int(xy_list[i, 0] + boxsize), 2047)
        ymin = max(int(xy_list[i, 1] - boxsize), 0)
        ymax = min(int(xy_list[i, 1] + boxsize), 2047)
        # print "xymin xymax = ", xmin, ymin, xmax, ymax
        linestowrite.append("ZOOM IN" + "\n")
        linestowrite.append("2" + "\n")
        linestowrite.append(str(xmin) + "\n")
        linestowrite.append(str(ymin) + "\n")
        linestowrite.append(str(xmax) + "\n")
        linestowrite.append(str(ymax) + "\n")
        # print "initialise"
        linestowrite.append("INITIALISE" + "\n")
        linestowrite.append("2-D POLYNOMIAL" + "\n")
        linestowrite.append("0" + "\n")
        linestowrite.append("0" + "\n")
        linestowrite.append("0" + "\n")
        linestowrite.append("0" + "\n")
        linestowrite.append("GAUSSIAN" + "\n")
        linestowrite.append("1" + "\n")
        linestowrite.append(str(xcen) + "\n")
        linestowrite.append(str(ycen) + "\n")
        xfw = xcen + startwidth
        linestowrite.append("1" + "\n")
        linestowrite.append(str(xfw) + "\n")
        linestowrite.append(str(ycen) + "\n")
        yfw = ycen + startwidth
        # print "xyfw = ", xfw, yfw
        linestowrite.append("1" + "\n")
        linestowrite.append(str(xcen) + "\n")
        linestowrite.append(str(yfw) + "\n")
        linestowrite.append("EXIT" + "\n")
        # print "optimise"
        linestowrite.append("OPTIMISE" + "\n")
        # sleep(1.0)
        # print "results"
        linestowrite.append("RESULTS" + "\n")
        # print "save", localtime()
        # linestowrite.append('SAVE'+'\n')
        # linestowrite.append('toto'+ str(i) +'.txt'+'\n')
        # linestowrite.append([os.path.join(dirname, 'toto.txt')+'\n')
        # linestowrite.append('O.K.'+'\n')
        linestowrite.append("FULL" + "\n")
        # print "peak number ", i

        linestowrite.append("EXIT" + "\n")
        linestowrite.append("MACROS / LOG FILE" + "\n")
        linestowrite.append("CLOSE LOG FILE" + "\n")
        # print "close log", localtime()
        linestowrite.append("EXIT" + "\n")
        linestowrite.append("EXIT FIT2D" + "\n")
        linestowrite.append("YES" + "\n")
        # print "exit fit2d", localtime()
        # print 'fit 2D peak fitting is finished! ...'
        # print "peak list has been written in %s"%('fit_'+prefix+'.peaks')

        outputfilename = "fit2d_fitpeaks.mac"
        outputfile = open(outputfilename, "w")

        print("outputfile name  = %s " % outputfilename)

        for line in linestowrite:
            # print line
            # lineData = '\t'.join(line)
            outputfile.write(line)
            # outputfile.write('\n')
        outputfile.close()

        print("start fit2d")

        myprocess = subprocess.Popen(
            " fit2d -dim2048x2048 -macfit2d_fitpeaks.mac ",
            shell=True,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        retcode = None
        i = 0
        while (retcode == None) & (i < 100):
            if i == 1:
                # needed under windows to close the created cmd.exe and fit2d.exe processes
                myprocess.communicate(input="exit \n")
            # returncode =  None if process is still running
            retcode = myprocess.returncode
            print("fit2d return code = ", retcode)
            i = i + 1
        print("execution time : ", time() - time_0)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        pass
    else:
        fitpeak_f2d("Ge_blanc_11Sep08_d1_5MPa_0000.mccd")
        # findpeak_f2d('Ge_blanc_0000.mccd')
        # findpeak_f2d('CdTe_I999_03Jul06_0200.mccd', dirname = './Examples/')
