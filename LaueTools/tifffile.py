#!/usr/bin/env python
# -*- coding: utf-8 -*-
# tifffile.py

# Copyright (c) 2008-2014, Christoph Gohlke
# Copyright (c) 2008-2014, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Read and write image data from and to TIFF files.

Image and meta-data can be read from TIFF, BigTIFF, OME-TIFF, STK, LSM, NIH,
ImageJ, MicroManager, FluoView, SEQ and GEL files.
Only a subset of the TIFF specification is supported, mainly uncompressed
and losslessly compressed 2**(0 to 6) bit integer, 16, 32 and 64-bit float,
grayscale and RGB(A) images, which are commonly used in bio-scientific imaging.
Specifically, reading JPEG/CCITT compressed image data or EXIF/IPTC/GPS/XMP
meta-data is not implemented. Only primary info records are read for STK,
FluoView, MicroManager, and NIH image formats.

TIFF, the Tagged Image File Format, is under the control of Adobe Systems.
BigTIFF allows for files greater than 4 GB. STK, LSM, FluoView, SEQ, GEL,
and OME-TIFF, are custom extensions defined by MetaMorph, Carl Zeiss
MicroImaging, Olympus, Media Cybernetics, Molecular Dynamics, and the Open
Microscopy Environment consortium respectively.

For command line usage run ``python tifffile.py --help``

:Author:
  `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:Version: 2014.02.05

Requirements
------------
* `CPython 2.7 or 3.3 <http://www.python.org>`_
* `Numpy 1.7 <http://www.numpy.org>`_
* `Matplotlib 1.3 <http://www.matplotlib.org>`_  (optional for plotting)
* `Tifffile.c 2013.01.18 <http://www.lfd.uci.edu/~gohlke/>`_
  (recommended for faster decoding of PackBits and LZW encoded strings)

Notes
-----
The API is not stable yet and might change between revisions.

Tested on little-endian platforms only.

Other Python packages and modules for reading bio-scientific TIFF files:
* `Imread <http://luispedro.org/software/imread>`_
* `PyLibTiff <http://code.google.com/p/pylibtiff>`_
* `SimpleITK <http://www.simpleitk.org>`_
* `PyLSM <https://launchpad.net/pylsm>`_
* `PyMca.TiffIO.py <http://pymca.sourceforge.net/>`_
* `BioImageXD.Readers <http://www.bioimagexd.net/>`_
* `Cellcognition.io <http://cellcognition.org/>`_
* `CellProfiler.bioformats <http://www.cellprofiler.org/>`_

Acknowledgements
----------------
*  Egor Zindy, University of Manchester, for cz_lsm_scan_info specifics.
*  Wim Lewis for a bug fix and some read_cz_lsm functions.
*  Hadrien Mary for help on reading MicroManager files.

References
----------
(1) TIFF 6.0 Specification and Supplements. Adobe Systems Incorporated.
    http://partners.adobe.com/public/developer/tiff/
(2) TIFF File Format FAQ. http://www.awaresystems.be/imaging/tiff/faq.html
(3) MetaMorph Stack (STK) Image File Format.
    http://support.meta.moleculardevices.com/docs/t10243.pdf
(4) File Format Description - LSM 5xx Release 2.0.
    http://ibb.gsf.de/homepage/karsten.rodenacker/IDL/Lsmfile.doc
(5) BioFormats. http://www.loci.wisc.edu/ome/formats.html
(6) The OME-TIFF format.
    http://www.openmicroscopy.org/site/support/file-formats/ome-tiff
(7) TiffDecoder.java
    http://rsbweb.nih.gov/ij/developer/source/ij/io/TiffDecoder.java.html
(8) UltraQuant(r) Version 6.0 for Windows Start-Up Guide.
    http://www.ultralum.com/images%20ultralum/pdf/UQStart%20Up%20Guide.pdf
(9) Micro-Manager File Formats.
    http://www.micro-manager.org/wiki/Micro-Manager_File_Formats

Examples
--------
>>> data = numpy.random.rand(301, 219)
>>> imsave('temp.tif', data)
>>> image = imread('temp.tif')
>>> assert numpy.all(image == data)

>>> tif = TiffFile('test.tif')
>>> images = tif.asarray()
>>> image0 = tif[0].asarray()
>>> for page in tif:
...     for tag in page.tags.values():
...         t = tag.name, tag.value
...     image = page.asarray()
...     if page.is_rgb: pass
...     if page.is_palette:
...         t = page.color_map
...     if page.is_stk:
...         t = page.mm_uic_tags.number_planes
...     if page.is_lsm:
...         t = page.cz_lsm_info
>>> tif.close()

"""

from __future__ import division, print_function

import sys
import os
import re
import glob
import math
import zlib
import time
import json
import struct
import warnings
import datetime
import collections
from fractions import Fraction
from xml.etree import cElementTree as ElementTree

import numpy

__version__ = "2014.02.05"
__docformat__ = "restructuredtext en"
__all__ = ["imsave", "imread", "imshow", "TiffFile", "TiffSequence"]


def imsave(
    filename,
    data,
    photometric=None,
    planarconfig=None,
    resolution=None,
    description=None,
    software="tifffile.py",
    byteorder=None,
    bigtiff=False,
    compress=0,
    extratags=(),
):
    """Write image data to TIFF file.

    Image data are written in one stripe per plane.
    Dimensions larger than 2 or 3 (depending on photometric mode and
    planar configuration) are flattened and saved as separate pages.
    The 'sample_format' and 'bits_per_sample' TIFF tags are derived from
    the data type.

    Parameters
    ----------
    filename : str
        Name of file to write.
    data : array_like
        Input image. The last dimensions are assumed to be image height,
        width, and samples.
    photometric : {'minisblack', 'miniswhite', 'rgb'}
        The color space of the image data.
        By default this setting is inferred from the data shape.
    planarconfig : {'contig', 'planar'}
        Specifies if samples are stored contiguous or in separate planes.
        By default this setting is inferred from the data shape.
        'contig': last dimension contains samples.
        'planar': third last dimension contains samples.
    resolution : (float, float) or ((int, int), (int, int))
        X and Y resolution in dots per inch as float or rational numbers.
    description : str
        The subject of the image. Saved with the first page only.
    software : str
        Name of the software used to create the image.
        Saved with the first page only.
    byteorder : {'<', '>'}
        The endianness of the data in the file.
        By default this is the system's native byte order.
    bigtiff : bool
        If True, the BigTIFF format is used.
        By default the standard TIFF format is used for data less than 2000 MB.
    compress : int
        Values from 0 to 9 controlling the level of zlib compression.
        If 0, data are written uncompressed (default).
    extratags: sequence of tuples
        Additional tags as [(code, dtype, count, value, writeonce)].
        code : int
            The TIFF tag Id.
        dtype : str
            Data type of items in `value` in Python struct format.
            One of B, s, H, I, 2I, b, h, i, f, d, Q, or q.
        count : int
            Number of data values. Not used for string values.
        value : sequence
            `Count` values compatible with `dtype`.
        writeonce : bool
            If True, the tag is written to the first page only.

    Examples
    --------
    >>> data = numpy.ones((2, 5, 3, 301, 219), 'float32') * 0.5
    >>> imsave('temp.tif', data, compress=6)

    >>> data = numpy.ones((5, 301, 219, 3), 'uint8') + 127
    >>> value = u'{"shape": %s}' % str(list(data.shape))
    >>> imsave('temp.tif', data, extratags=[(270, 's', 0, value, True)])

    """
    assert photometric in (None, "minisblack", "miniswhite", "rgb")
    assert planarconfig in (None, "contig", "planar")
    assert byteorder in (None, "<", ">")
    assert 0 <= compress <= 9

    if byteorder is None:
        byteorder = "<" if sys.byteorder == "little" else ">"

    data = numpy.asarray(data, dtype=byteorder + data.dtype.char, order="C")
    data_shape = shape = data.shape
    data = numpy.atleast_2d(data)

    if not bigtiff and data.size * data.dtype.itemsize < 2000 * 2 ** 20:
        bigtiff = False
        offset_size = 4
        tag_size = 12
        numtag_format = "H"
        offset_format = "I"
        val_format = "4s"
    else:
        bigtiff = True
        offset_size = 8
        tag_size = 20
        numtag_format = "Q"
        offset_format = "Q"
        val_format = "8s"

    # unify shape of data
    samplesperpixel = 1
    extrasamples = 0
    if photometric is None:
        if data.ndim > 2 and (shape[-3] in (3, 4) or shape[-1] in (3, 4)):
            photometric = "rgb"
        else:
            photometric = "minisblack"
    if photometric == "rgb":
        if len(shape) < 3:
            raise ValueError("not a RGB(A) image")
        if planarconfig is None:
            planarconfig = "planar" if shape[-3] in (3, 4) else "contig"
        if planarconfig == "contig":
            if shape[-1] not in (3, 4):
                raise ValueError("not a contiguous RGB(A) image")
            data = data.reshape((-1, 1) + shape[-3:])
            samplesperpixel = shape[-1]
        else:
            if shape[-3] not in (3, 4):
                raise ValueError("not a planar RGB(A) image")
            data = data.reshape((-1,) + shape[-3:] + (1,))
            samplesperpixel = shape[-3]
        if samplesperpixel == 4:
            extrasamples = 1
    elif planarconfig and len(shape) > 2:
        if planarconfig == "contig":
            data = data.reshape((-1, 1) + shape[-3:])
            samplesperpixel = shape[-1]
        else:
            data = data.reshape((-1,) + shape[-3:] + (1,))
            samplesperpixel = shape[-3]
        extrasamples = samplesperpixel - 1
    else:
        planarconfig = None
        # remove trailing 1s
        while len(shape) > 2 and shape[-1] == 1:
            shape = shape[:-1]
        data = data.reshape((-1, 1) + shape[-2:] + (1,))

    shape = data.shape  # (pages, planes, height, width, contig samples)

    bytestr = (
        bytes
        if sys.version[0] == "2"
        else (lambda x: bytes(x, "utf-8") if isinstance(x, str) else x)
    )
    tifftypes = {
        "B": 1,
        "s": 2,
        "H": 3,
        "I": 4,
        "2I": 5,
        "b": 6,
        "h": 8,
        "i": 9,
        "f": 11,
        "d": 12,
        "Q": 16,
        "q": 17,
    }
    tifftags = {
        "new_subfile_type": 254,
        "subfile_type": 255,
        "image_width": 256,
        "image_length": 257,
        "bits_per_sample": 258,
        "compression": 259,
        "photometric": 262,
        "fill_order": 266,
        "document_name": 269,
        "image_description": 270,
        "strip_offsets": 273,
        "orientation": 274,
        "samples_per_pixel": 277,
        "rows_per_strip": 278,
        "strip_byte_counts": 279,
        "x_resolution": 282,
        "y_resolution": 283,
        "planar_configuration": 284,
        "page_name": 285,
        "resolution_unit": 296,
        "software": 305,
        "datetime": 306,
        "predictor": 317,
        "color_map": 320,
        "extra_samples": 338,
        "sample_format": 339,
    }
    tags = []  # list of (code, ifdentry, ifdvalue, writeonce)

    def pack(fmt, *val):
        return struct.pack(byteorder + fmt, *val)

    def addtag(code, dtype, count, value, writeonce=False):
        # compute ifdentry and ifdvalue bytes from code, dtype, count, value
        # append (code, ifdentry, ifdvalue, writeonce) to tags list
        code = tifftags[code] if code in tifftags else int(code)
        if dtype not in tifftypes:
            raise ValueError("unknown dtype %s" % dtype)
        tifftype = tifftypes[dtype]
        rawcount = count
        if dtype == "s":
            value = bytestr(value) + b"\0"
            count = rawcount = len(value)
            value = (value,)
        if len(dtype) > 1:
            count *= int(dtype[:-1])
            dtype = dtype[-1]
        ifdentry = [pack("HH", code, tifftype), pack(offset_format, rawcount)]
        ifdvalue = None
        if count == 1:
            if isinstance(value, (tuple, list)):
                value = value[0]
            ifdentry.append(pack(val_format, pack(dtype, value)))
        elif struct.calcsize(dtype) * count <= offset_size:
            ifdentry.append(pack(val_format, pack(str(count) + dtype, *value)))
        else:
            ifdentry.append(pack(offset_format, 0))
            ifdvalue = pack(str(count) + dtype, *value)
        tags.append((code, b"".join(ifdentry), ifdvalue, writeonce))

    def rational(arg, max_denominator=1000000):
        # return nominator and denominator from float or two integers
        try:
            f = Fraction.from_float(arg)
        except TypeError:
            f = Fraction(arg[0], arg[1])
        f = f.limit_denominator(max_denominator)
        return f.numerator, f.denominator

    if software:
        addtag("software", "s", 0, software, writeonce=True)
    if description:
        addtag("image_description", "s", 0, description, writeonce=True)
    elif shape != data_shape:
        addtag(
            "image_description",
            "s",
            0,
            "shape=(%s)" % (",".join("%i" % i for i in data_shape)),
            writeonce=True,
        )
    addtag(
        "datetime",
        "s",
        0,
        datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
        writeonce=True,
    )
    addtag("compression", "H", 1, 32946 if compress else 1)
    addtag("orientation", "H", 1, 1)
    addtag("image_width", "I", 1, shape[-2])
    addtag("image_length", "I", 1, shape[-3])
    addtag("new_subfile_type", "I", 1, 0 if shape[0] == 1 else 2)
    addtag("sample_format", "H", 1, {"u": 1, "i": 2, "f": 3, "c": 6}[data.dtype.kind])
    addtag(
        "photometric", "H", 1, {"miniswhite": 0, "minisblack": 1, "rgb": 2}[photometric]
    )
    addtag("samples_per_pixel", "H", 1, samplesperpixel)
    if planarconfig:
        addtag("planar_configuration", "H", 1, 1 if planarconfig == "contig" else 2)
        addtag(
            "bits_per_sample",
            "H",
            samplesperpixel,
            (data.dtype.itemsize * 8,) * samplesperpixel,
        )
    else:
        addtag("bits_per_sample", "H", 1, data.dtype.itemsize * 8)
    if extrasamples:
        if photometric == "rgb":
            addtag("extra_samples", "H", 1, 1)  # alpha channel
        else:
            addtag("extra_samples", "H", extrasamples, (0,) * extrasamples)
    if resolution:
        addtag("x_resolution", "2I", 1, rational(resolution[0]))
        addtag("y_resolution", "2I", 1, rational(resolution[1]))
        addtag("resolution_unit", "H", 1, 2)
    addtag("rows_per_strip", "I", 1, shape[-3])

    # use one strip per plane
    strip_byte_counts = (data[0, 0].size * data.dtype.itemsize,) * shape[1]
    addtag("strip_byte_counts", offset_format, shape[1], strip_byte_counts)
    addtag("strip_offsets", offset_format, shape[1], (0,) * shape[1])

    # add extra tags from users
    for t in extratags:
        addtag(*t)

    # the entries in an IFD must be sorted in ascending order by tag code
    tags = sorted(tags, key=lambda x: x[0])

    with open(filename, "wb") as fh:
        seek = fh.seek
        tell = fh.tell

        def write(arg, *args):
            fh.write(pack(arg, *args) if args else arg)

        write({"<": b"II", ">": b"MM"}[byteorder])
        if bigtiff:
            write("HHH", 43, 8, 0)
        else:
            write("H", 42)
        ifd_offset = tell()
        write(offset_format, 0)  # first IFD

        for pageindex in range(shape[0]):
            # update pointer at ifd_offset
            pos = tell()
            seek(ifd_offset)
            write(offset_format, pos)
            seek(pos)

            # write ifdentries
            write(numtag_format, len(tags))
            tag_offset = tell()
            write(b"".join(t[1] for t in tags))
            ifd_offset = tell()
            write(offset_format, 0)  # offset to next IFD

            # write tag values and patch offsets in ifdentries, if necessary
            for tagindex, tag in enumerate(tags):
                if tag[2]:
                    pos = tell()
                    seek(tag_offset + tagindex * tag_size + offset_size + 4)
                    write(offset_format, pos)
                    seek(pos)
                    if tag[0] == 273:
                        strip_offsets_offset = pos
                    elif tag[0] == 279:
                        strip_byte_counts_offset = pos
                    write(tag[2])

            # write image data
            data_offset = tell()
            if compress:
                strip_byte_counts = []
                for plane in data[pageindex]:
                    plane = zlib.compress(plane, compress)
                    strip_byte_counts.append(len(plane))
                    fh.write(plane)
            else:
                # if this fails try update Python/numpy
                data[pageindex].tofile(fh)
                fh.flush()

            # update strip_offsets and strip_byte_counts if necessary
            pos = tell()
            for tagindex, tag in enumerate(tags):
                if tag[0] == 273:  # strip_offsets
                    if tag[2]:
                        seek(strip_offsets_offset)
                        strip_offset = data_offset
                        for size in strip_byte_counts:
                            write(offset_format, strip_offset)
                            strip_offset += size
                    else:
                        seek(tag_offset + tagindex * tag_size + offset_size + 4)
                        write(offset_format, data_offset)
                elif tag[0] == 279:  # strip_byte_counts
                    if compress:
                        if tag[2]:
                            seek(strip_byte_counts_offset)
                            for size in strip_byte_counts:
                                write(offset_format, size)
                        else:
                            seek(tag_offset + tagindex * tag_size + offset_size + 4)
                            write(offset_format, strip_byte_counts[0])
                    break
            seek(pos)
            fh.flush()
            # remove tags that should be written only once
            if pageindex == 0:
                tags = [t for t in tags if not t[-1]]


def imread(files, *args, **kwargs):
    """Return image data from TIFF file(s) as numpy array.

    The first image series is returned if no arguments are provided.

    Parameters
    ----------
    files : str or list
        File name, glob pattern, or list of file names.
    key : int, slice, or sequence of page indices
        Defines which pages to return as array.
    series : int
        Defines which series of pages in file to return as array.
    multifile : bool
        If True (default), OME-TIFF data may include pages from multiple files.
    pattern : str
        Regular expression pattern that matches axes names and indices in
        file names.

    Examples
    --------
    >>> im = imread('test.tif', 0)
    >>> im.shape
    (256, 256, 4)
    >>> ims = imread(['test.tif', 'test.tif'])
    >>> ims.shape
    (2, 256, 256, 4)

    """
    kwargs_file = {}
    if "multifile" in kwargs:
        kwargs_file["multifile"] = kwargs["multifile"]
        del kwargs["multifile"]
    else:
        kwargs_file["multifile"] = True
    kwargs_seq = {}
    if "pattern" in kwargs:
        kwargs_seq["pattern"] = kwargs["pattern"]
        del kwargs["pattern"]

    if isinstance(files, basestring) and any(i in files for i in "?*"):
        files = glob.glob(files)
    if not files:
        raise ValueError("no files found")
    if len(files) == 1:
        files = files[0]

    if isinstance(files, basestring):
        with TiffFile(files, **kwargs_file) as tif:
            return tif.asarray(*args, **kwargs)
    else:
        with TiffSequence(files, **kwargs_seq) as imseq:
            return imseq.asarray(*args, **kwargs)


class lazyattr(object):
    """Lazy object attribute whose value is computed on first access."""

    __slots__ = ("func",)

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        if instance is None:
            return self
        value = self.func(instance)
        if value is NotImplemented:
            return getattr(super(owner, instance), self.func.__name__)
        setattr(instance, self.func.__name__, value)
        return value


class TiffFile(object):
    """Read image and meta-data from TIFF, STK, LSM, and FluoView files.

    TiffFile instances must be closed using the close method, which is
    automatically called when using the 'with' statement.

    Attributes
    ----------
    pages : list
        All TIFF pages in file.
    series : list of Records(shape, dtype, axes, TiffPages)
        TIFF pages with compatible shapes and types.
    micromanager_metadata: dict
        Extra MicroManager non-TIFF metadata in the file, if exists.

    All attributes are read-only.

    Examples
    --------
    >>> tif = TiffFile('test.tif')
    ... try:
    ...     images = tif.asarray()
    ... except Exception as e:
    ...     print(e)
    ... finally:
    ...     tif.close()

    """

    def __init__(self, arg, name=None, multifile=False):
        """Initialize instance from file.

        Parameters
        ----------
        arg : str or open file
            Name of file or open file object.
            The file objects are closed in TiffFile.close().
        name : str
            Human readable label of open file.
        multifile : bool
            If True, series may include pages from multiple files.

        """
        if isinstance(arg, basestring):
            filename = os.path.abspath(arg)
            self._fh = open(filename, "rb")
        else:
            filename = str(name)
            self._fh = arg

        self._fh.seek(0, 2)
        self._fsize = self._fh.tell()
        self._fh.seek(0)
        self.fname = os.path.basename(filename)
        self.fpath = os.path.dirname(filename)
        self._tiffs = {self.fname: self}  # cache of TiffFiles
        self.offset_size = None
        self.pages = []
        self._multifile = bool(multifile)
        try:
            self._fromfile()
        except Exception:
            self._fh.close()
            raise

    def close(self):
        """Close open file handle(s)."""
        for tif in self._tiffs.values():
            if tif._fh:
                tif._fh.close()
                tif._fh = None
        self._tiffs = {}

    def _fromfile(self):
        """Read TIFF header and all page records from file."""
        self._fh.seek(0)
        try:
            self.byteorder = {b"II": "<", b"MM": ">"}[self._fh.read(2)]
        except KeyError:
            raise ValueError("not a valid TIFF file")
        version = struct.unpack(self.byteorder + "H", self._fh.read(2))[0]
        if version == 43:  # BigTiff
            self.offset_size, zero = struct.unpack(
                self.byteorder + "HH", self._fh.read(4)
            )
            if zero or self.offset_size != 8:
                raise ValueError("not a valid BigTIFF file")
        elif version == 42:
            self.offset_size = 4
        else:
            raise ValueError("not a TIFF file")
        self.pages = []
        while True:
            try:
                page = TiffPage(self)
                self.pages.append(page)
            except StopIteration:
                break
        if not self.pages:
            raise ValueError("empty TIFF file")

        if self.is_micromanager:
            # MicroManager files contain metadata not stored in TIFF tags.
            self.micromanager_metadata = read_micromanager_metadata(self._fh)

    @lazyattr
    def series(self):
        """Return series of TiffPage with compatible shape and properties."""
        series = []
        if self.is_ome:
            series = self._omeseries()
        elif self.is_fluoview:
            dims = {
                b"X": "X",
                b"Y": "Y",
                b"Z": "Z",
                b"T": "T",
                b"WAVELENGTH": "C",
                b"TIME": "T",
                b"XY": "R",
                b"EVENT": "V",
                b"EXPOSURE": "L",
            }
            mmhd = list(reversed(self.pages[0].mm_header.dimensions))
            series = [
                Record(
                    axes="".join(
                        dims.get(i[0].strip().upper(), "Q") for i in mmhd if i[1] > 1
                    ),
                    shape=tuple(int(i[1]) for i in mmhd if i[1] > 1),
                    pages=self.pages,
                    dtype=numpy.dtype(self.pages[0].dtype),
                )
            ]
        elif self.is_lsm:
            lsmi = self.pages[0].cz_lsm_info
            axes = CZ_SCAN_TYPES[lsmi.scan_type]
            if self.pages[0].is_rgb:
                axes = axes.replace("C", "").replace("XY", "XYC")
            axes = axes[::-1]
            shape = [getattr(lsmi, CZ_DIMENSIONS[i]) for i in axes]
            pages = [p for p in self.pages if not p.is_reduced]
            series = [
                Record(
                    axes=axes,
                    shape=shape,
                    pages=pages,
                    dtype=numpy.dtype(pages[0].dtype),
                )
            ]
            if len(pages) != len(self.pages):  # reduced RGB pages
                pages = [p for p in self.pages if p.is_reduced]
                cp = 1
                i = 0
                while cp < len(pages) and i < len(shape) - 2:
                    cp *= shape[i]
                    i += 1
                shape = shape[:i] + list(pages[0].shape)
                axes = axes[:i] + "CYX"
                series.append(
                    Record(
                        axes=axes,
                        shape=shape,
                        pages=pages,
                        dtype=numpy.dtype(pages[0].dtype),
                    )
                )
        elif self.is_imagej:
            shape = []
            axes = []
            ij = self.pages[0].imagej_tags
            if "frames" in ij:
                shape.append(ij["frames"])
                axes.append("T")
            if "slices" in ij:
                shape.append(ij["slices"])
                axes.append("Z")
            if "channels" in ij and not self.is_rgb:
                shape.append(ij["channels"])
                axes.append("C")
            remain = len(self.pages) // (numpy.prod(shape) if shape else 1)
            if remain > 1:
                shape.append(remain)
                axes.append("I")
            shape.extend(self.pages[0].shape)
            axes.extend(self.pages[0].axes)
            axes = "".join(axes)
            series = [
                Record(
                    pages=self.pages,
                    shape=shape,
                    axes=axes,
                    dtype=numpy.dtype(self.pages[0].dtype),
                )
            ]
        elif self.is_nih:
            series = [
                Record(
                    pages=self.pages,
                    shape=(len(self.pages),) + self.pages[0].shape,
                    axes="I" + self.pages[0].axes,
                    dtype=numpy.dtype(self.pages[0].dtype),
                )
            ]
        elif self.pages[0].is_shaped:
            shape = self.pages[0].tags["image_description"].value[7:-1]
            shape = tuple(int(i) for i in shape.split(b","))
            series = [
                Record(
                    pages=self.pages,
                    shape=shape,
                    axes="Q" * len(shape),
                    dtype=numpy.dtype(self.pages[0].dtype),
                )
            ]

        if not series:
            shapes = []
            pages = {}
            for page in self.pages:
                if not page.shape:
                    continue
                shape = page.shape + (page.axes, page.compression in TIFF_DECOMPESSORS)
                if not shape in pages:
                    shapes.append(shape)
                    pages[shape] = [page]
                else:
                    pages[shape].append(page)
            series = [
                Record(
                    pages=pages[s],
                    axes=(("I" + s[-2]) if len(pages[s]) > 1 else s[-2]),
                    dtype=numpy.dtype(pages[s][0].dtype),
                    shape=((len(pages[s]),) + s[:-2] if len(pages[s]) > 1 else s[:-2]),
                )
                for s in shapes
            ]
        return series

    def asarray(self, key=None, series=None, memmap=False):
        """Return image data of multiple TIFF pages as numpy array.

        By default the first image series is returned.

        Parameters
        ----------
        key : int, slice, or sequence of page indices
            Defines which pages to return as array.
        series : int
            Defines which series of pages to return as array.
        memmap : bool
            If True, use numpy.memmap to read arrays from file if possible.

        """
        if key is None and series is None:
            series = 0
        if series is not None:
            pages = self.series[series].pages
        else:
            pages = self.pages

        if key is None:
            pass
        elif isinstance(key, int):
            pages = [pages[key]]
        elif isinstance(key, slice):
            pages = pages[key]
        elif isinstance(key, collections.Iterable):
            pages = [pages[k] for k in key]
        else:
            raise TypeError("key must be an int, slice, or sequence")

        if len(pages) == 1:
            return pages[0].asarray(memmap=memmap)
        elif self.is_nih:
            result = numpy.vstack(
                p.asarray(colormapped=False, squeeze=False, memmap=memmap)
                for p in pages
            )
            if pages[0].is_palette:
                result = numpy.take(pages[0].color_map, result, axis=1)
                result = numpy.swapaxes(result, 0, 1)
        else:
            if self.is_ome and any(p is None for p in pages):
                firstpage = next(p for p in pages if p)
                nopage = numpy.zeros_like(firstpage.asarray(memmap=memmap))
            result = numpy.vstack(
                (p.asarray(memmap=memmap) if p else nopage) for p in pages
            )
        if key is None:
            try:
                result.shape = self.series[series].shape
            except ValueError:
                warnings.warn(
                    "failed to reshape %s to %s"
                    % (result.shape, self.series[series].shape)
                )
                result.shape = (-1,) + pages[0].shape
        else:
            result.shape = (-1,) + pages[0].shape
        return result

    def _omeseries(self):
        """Return image series in OME-TIFF file(s)."""
        root = ElementTree.XML(self.pages[0].tags["image_description"].value)
        uuid = root.attrib.get("UUID", None)
        self._tiffs = {uuid: self}
        modulo = {}
        result = []
        for element in root:
            if element.tag.endswith("BinaryOnly"):
                warnings.warn("not an OME-TIFF master file")
                break
            if element.tag.endswith("StructuredAnnotations"):
                for annot in element:
                    if not annot.attrib.get("Namespace", "").endswith("modulo"):
                        continue
                    for value in annot:
                        for modul in value:
                            for along in modul:
                                if not along.tag[:-1].endswith("Along"):
                                    continue
                                axis = along.tag[-1]
                                newaxis = along.attrib.get("Type", "other")
                                newaxis = AXES_LABELS[newaxis]
                                if "Start" in along.attrib:
                                    labels = range(
                                        int(along.attrib["Start"]),
                                        int(along.attrib["End"]) + 1,
                                        int(along.attrib.get("Step", 1)),
                                    )
                                else:
                                    labels = [
                                        label.text
                                        for label in along
                                        if label.tag.endswith("Label")
                                    ]
                                modulo[axis] = (newaxis, labels)
            if not element.tag.endswith("Image"):
                continue
            for pixels in element:
                if not pixels.tag.endswith("Pixels"):
                    continue
                atr = pixels.attrib
                axes = "".join(reversed(atr["DimensionOrder"]))
                shape = list(int(atr["Size" + ax]) for ax in axes)
                size = numpy.prod(shape[:-2])
                ifds = [None] * size
                for data in pixels:
                    if not data.tag.endswith("TiffData"):
                        continue
                    atr = data.attrib
                    ifd = int(atr.get("IFD", 0))
                    num = int(atr.get("NumPlanes", 1 if "IFD" in atr else 0))
                    num = int(atr.get("PlaneCount", num))
                    idx = [int(atr.get("First" + ax, 0)) for ax in axes[:-2]]
                    idx = numpy.ravel_multi_index(idx, shape[:-2])
                    for uuid in data:
                        if uuid.tag.endswith("UUID"):
                            if uuid.text not in self._tiffs:
                                if not self._multifile:
                                    # abort reading multi file OME series
                                    return []
                                fn = uuid.attrib["FileName"]
                                try:
                                    tf = TiffFile(os.path.join(self.fpath, fn))
                                except (IOError, ValueError):
                                    warnings.warn("failed to read %s" % fn)
                                    break
                                self._tiffs[uuid.text] = tf
                            pages = self._tiffs[uuid.text].pages
                            try:
                                for i in range(num if num else len(pages)):
                                    ifds[idx + i] = pages[ifd + i]
                            except IndexError:
                                warnings.warn("ome-xml: index out of range")
                            break
                    else:
                        pages = self.pages
                        try:
                            for i in range(num if num else len(pages)):
                                ifds[idx + i] = pages[ifd + i]
                        except IndexError:
                            warnings.warn("ome-xml: index out of range")
                result.append(
                    Record(
                        axes=axes,
                        shape=shape,
                        pages=ifds,
                        dtype=numpy.dtype(ifds[0].dtype),
                    )
                )

        for record in result:
            for axis, (newaxis, labels) in modulo.items():
                i = record.axes.index(axis)
                size = len(labels)
                if record.shape[i] == size:
                    record.axes = record.axes.replace(axis, newaxis, 1)
                else:
                    record.shape[i] //= size
                    record.shape.insert(i + 1, size)
                    record.axes = record.axes.replace(axis, axis + newaxis, 1)

        return result

    def __len__(self):
        """Return number of image pages in file."""
        return len(self.pages)

    def __getitem__(self, key):
        """Return specified page."""
        return self.pages[key]

    def __iter__(self):
        """Return iterator over pages."""
        return iter(self.pages)

    def __str__(self):
        """Return string containing information about file."""
        result = [
            self.fname.capitalize(),
            format_size(self._fsize),
            {"<": "little endian", ">": "big endian"}[self.byteorder],
        ]
        if self.is_bigtiff:
            result.append("bigtiff")
        if len(self.pages) > 1:
            result.append("%i pages" % len(self.pages))
        if len(self.series) > 1:
            result.append("%i series" % len(self.series))
        if len(self._tiffs) > 1:
            result.append("%i files" % (len(self._tiffs)))
        return ", ".join(result)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @lazyattr
    def fstat(self):
        try:
            return os.fstat(self._fh.fileno())
        except Exception:  # io.UnsupportedOperation
            return None

    @lazyattr
    def is_bigtiff(self):
        return self.offset_size != 4

    @lazyattr
    def is_rgb(self):
        return all(p.is_rgb for p in self.pages)

    @lazyattr
    def is_palette(self):
        return all(p.is_palette for p in self.pages)

    @lazyattr
    def is_mdgel(self):
        return any(p.is_mdgel for p in self.pages)

    @lazyattr
    def is_mediacy(self):
        return any(p.is_mediacy for p in self.pages)

    @lazyattr
    def is_stk(self):
        return all(p.is_stk for p in self.pages)

    @lazyattr
    def is_lsm(self):
        return self.pages[0].is_lsm

    @lazyattr
    def is_imagej(self):
        return self.pages[0].is_imagej

    @lazyattr
    def is_micromanager(self):
        return self.pages[0].is_micromanager

    @lazyattr
    def is_nih(self):
        return self.pages[0].is_nih

    @lazyattr
    def is_fluoview(self):
        return self.pages[0].is_fluoview

    @lazyattr
    def is_ome(self):
        return self.pages[0].is_ome


class TiffPage(object):
    """A TIFF image file directory (IFD).

    Attributes
    ----------
    index : int
        Index of page in file.
    dtype : str {TIFF_SAMPLE_DTYPES}
        Data type of image, colormapped if applicable.
    shape : tuple
        Dimensions of the image array in TIFF page,
        colormapped and with one alpha channel if applicable.
    axes : str
        Axes label codes:
        'X' width, 'Y' height, 'S' sample, 'P' plane, 'I' image series,
        'Z' depth, 'C' color|em-wavelength|channel, 'E' ex-wavelength|lambda,
        'T' time, 'R' region|tile, 'A' angle, 'F' phase, 'H' lifetime,
        'L' exposure, 'V' event, 'Q' unknown, '_' missing
    tags : TiffTags
        Dictionary of tags in page.
        Tag values are also directly accessible as attributes.
    color_map : numpy array
        Color look up table, if exists.
    mm_uic_tags: Record(dict)
        Consolidated MetaMorph mm_uic# tags, if exists.
    cz_lsm_scan_info: Record(dict)
        LSM scan info attributes, if exists.
    imagej_tags: Record(dict)
        Consolidated ImageJ description and metadata tags, if exists.

    All attributes are read-only.

    """

    def __init__(self, parent):
        """Initialize instance from file."""
        self.parent = parent
        self.index = len(parent.pages)
        self.shape = self._shape = ()
        self.dtype = self._dtype = None
        self.axes = ""
        self.tags = TiffTags()

        self._fromfile()
        self._process_tags()

    def _fromfile(self):
        """Read TIFF IFD structure and its tags from file.

        File cursor must be at storage position of IFD offset and is left at
        offset to next IFD.

        Raises StopIteration if offset (first bytes read) is 0.

        """
        fh = self.parent._fh
        byteorder = self.parent.byteorder
        offset_size = self.parent.offset_size

        fmt = {4: "I", 8: "Q"}[offset_size]
        offset = struct.unpack(byteorder + fmt, fh.read(offset_size))[0]
        if not offset:
            raise StopIteration()

        # read standard tags
        tags = self.tags
        fh.seek(offset)
        fmt, size = {4: ("H", 2), 8: ("Q", 8)}[offset_size]
        try:
            numtags = struct.unpack(byteorder + fmt, fh.read(size))[0]
        except Exception:
            warnings.warn("corrupted page list")
            raise StopIteration()

        tagcode = 0
        for _ in range(numtags):
            try:
                tag = TiffTag(self.parent)
            except TiffTag.Error as e:
                warnings.warn(str(e))
            finally:
                if tagcode > tag.code:
                    warnings.warn("tags are not ordered by code")
                tagcode = tag.code
                if not tag.name in tags:
                    tags[tag.name] = tag
                else:
                    # some files contain multiple IFD with same code
                    # e.g. MicroManager files contain two image_description
                    for ext in ("_1", "_2", "_3"):
                        name = tag.name + ext
                        if not name in tags:
                            tags[name] = tag
                            break

        # read LSM info subrecords
        if self.is_lsm:
            pos = fh.tell()
            for name, reader in CZ_LSM_INFO_READERS.items():
                try:
                    offset = self.cz_lsm_info["offset_" + name]
                except KeyError:
                    continue
                if not offset:
                    continue
                fh.seek(offset)
                try:
                    setattr(self, "cz_lsm_" + name, reader(fh, byteorder))
                except ValueError:
                    pass
            fh.seek(pos)

    def _process_tags(self):
        """Validate standard tags and initialize attributes.

        Raise ValueError if tag values are not supported.

        """
        tags = self.tags
        for code, (name, default, dtype, count, validate) in TIFF_TAGS.items():
            if not (name in tags or default is None):
                tags[name] = TiffTag(
                    code, dtype=dtype, count=count, value=default, name=name
                )
            if name in tags and validate:
                try:
                    if tags[name].count == 1:
                        setattr(self, name, validate[tags[name].value])
                    else:
                        setattr(
                            self,
                            name,
                            tuple(validate[value] for value in tags[name].value),
                        )
                except KeyError:
                    raise ValueError(
                        "%s.value (%s) not supported" % (name, tags[name].value)
                    )

        tag = tags["bits_per_sample"]
        if tag.count == 1:
            self.bits_per_sample = tag.value
        else:
            value = tag.value[: self.samples_per_pixel]
            if any((v - value[0] for v in value)):
                self.bits_per_sample = value
            else:
                self.bits_per_sample = value[0]

        tag = tags["sample_format"]
        if tag.count == 1:
            self.sample_format = TIFF_SAMPLE_FORMATS[tag.value]
        else:
            value = tag.value[: self.samples_per_pixel]
            if any((v - value[0] for v in value)):
                self.sample_format = [TIFF_SAMPLE_FORMATS[v] for v in value]
            else:
                self.sample_format = TIFF_SAMPLE_FORMATS[value[0]]

        if not "photometric" in tags:
            self.photometric = None

        if "image_length" in tags:
            self.strips_per_image = int(
                math.floor(
                    float(self.image_length + self.rows_per_strip - 1)
                    / self.rows_per_strip
                )
            )
        else:
            self.strips_per_image = 0

        key = (self.sample_format, self.bits_per_sample)
        self.dtype = self._dtype = TIFF_SAMPLE_DTYPES.get(key, None)

        if self.is_imagej:
            # consolidate imagej meta data
            if "image_description_1" in self.tags:  # MicroManager
                adict = imagej_description(tags["image_description_1"].value)
            else:
                adict = imagej_description(tags["image_description"].value)
            if "imagej_metadata" in tags:
                try:
                    adict.update(
                        imagej_metadata(
                            tags["imagej_metadata"].value,
                            tags["imagej_byte_counts"].value,
                            self.parent.byteorder,
                        )
                    )
                except Exception as e:
                    warnings.warn(str(e))
            self.imagej_tags = Record(adict)

        if not "image_length" in self.tags or not "image_width" in self.tags:
            # some GEL file pages are missing image data
            self.image_length = 0
            self.image_width = 0
            self.strip_offsets = 0
            self._shape = ()
            self.shape = ()
            self.axes = ""

        if self.is_palette:
            self.dtype = self.tags["color_map"].dtype[1]
            self.color_map = numpy.array(self.color_map, self.dtype)
            dmax = self.color_map.max()
            if dmax < 256:
                self.dtype = numpy.uint8
                self.color_map = self.color_map.astype(self.dtype)
            # else:
            #    self.dtype = numpy.uint8
            #    self.color_map >>= 8
            #    self.color_map = self.color_map.astype(self.dtype)
            self.color_map.shape = (3, -1)

        if self.is_stk:
            # consolidate mm_uci tags
            planes = tags["mm_uic2"].count
            self.mm_uic_tags = Record(tags["mm_uic2"].value)
            for key in ("mm_uic3", "mm_uic4", "mm_uic1"):
                if key in tags:
                    self.mm_uic_tags.update(tags[key].value)
            if self.planar_configuration == "contig":
                self._shape = (
                    planes,
                    1,
                    self.image_length,
                    self.image_width,
                    self.samples_per_pixel,
                )
                self.shape = tuple(self._shape[i] for i in (0, 2, 3, 4))
                self.axes = "PYXS"
            else:
                self._shape = (
                    planes,
                    self.samples_per_pixel,
                    self.image_length,
                    self.image_width,
                    1,
                )
                self.shape = self._shape[:4]
                self.axes = "PSYX"
            if self.is_palette and (
                self.color_map.shape[1] >= 2 ** self.bits_per_sample
            ):
                self.shape = (3, planes, self.image_length, self.image_width)
                self.axes = "CPYX"
            else:
                warnings.warn("palette cannot be applied")
                self.is_palette = False
        elif self.is_palette:
            samples = 1
            if "extra_samples" in self.tags:
                samples += len(self.extra_samples)
            if self.planar_configuration == "contig":
                self._shape = (1, 1, self.image_length, self.image_width, samples)
            else:
                self._shape = (1, samples, self.image_length, self.image_width, 1)
            if self.color_map.shape[1] >= 2 ** self.bits_per_sample:
                self.shape = (3, self.image_length, self.image_width)
                self.axes = "CYX"
            else:
                warnings.warn("palette cannot be applied")
                self.is_palette = False
                self.shape = (self.image_length, self.image_width)
                self.axes = "YX"
        elif self.is_rgb or self.samples_per_pixel > 1:
            if self.planar_configuration == "contig":
                self._shape = (
                    1,
                    1,
                    self.image_length,
                    self.image_width,
                    self.samples_per_pixel,
                )
                self.shape = (
                    self.image_length,
                    self.image_width,
                    self.samples_per_pixel,
                )
                self.axes = "YXS"
            else:
                self._shape = (
                    1,
                    self.samples_per_pixel,
                    self.image_length,
                    self.image_width,
                    1,
                )
                self.shape = self._shape[1:-1]
                self.axes = "SYX"
            if self.is_rgb and "extra_samples" in self.tags:
                extra_samples = self.extra_samples
                if self.tags["extra_samples"].count == 1:
                    extra_samples = (extra_samples,)
                for exs in extra_samples:
                    if exs in ("unassalpha", "assocalpha", "unspecified"):
                        if self.planar_configuration == "contig":
                            self.shape = self.shape[:2] + (4,)
                        else:
                            self.shape = (4,) + self.shape[1:]
                        break
        else:
            self._shape = (1, 1, self.image_length, self.image_width, 1)
            self.shape = self._shape[2:4]
            self.axes = "YX"

        if not self.compression and not "strip_byte_counts" in tags:
            self.strip_byte_counts = numpy.prod(self.shape) * (
                self.bits_per_sample // 8
            )

    def asarray(self, squeeze=True, colormapped=True, rgbonly=True, memmap=False):
        """Read image data from file and return as numpy array.

        Raise ValueError if format is unsupported.
        If any argument is False, the shape of the returned array might be
        different from the page shape.

        Parameters
        ----------
        squeeze : bool
            If True, all length-1 dimensions (except X and Y) are
            squeezed out from result.
        colormapped : bool
            If True, color mapping is applied for palette-indexed images.
        rgbonly : bool
            If True, return RGB(A) image without additional extra samples.
        memmap : bool
            If True, use numpy.memmap to read array if possible.

        """
        fh = self.parent._fh
        if not fh:
            raise IOError("TIFF file is not open")
        if self.dtype is None:
            raise ValueError(
                "data type not supported: %s%i"
                % (self.sample_format, self.bits_per_sample)
            )
        if self.compression not in TIFF_DECOMPESSORS:
            raise ValueError("cannot decompress %s" % self.compression)
        if "ycbcr_subsampling" in self.tags and self.tags[
            "ycbcr_subsampling"
        ].value not in (1, (1, 1)):
            raise ValueError("YCbCr subsampling not supported")
        tag = self.tags["sample_format"]
        if tag.count != 1 and any((i - tag.value[0] for i in tag.value)):
            raise ValueError("sample formats don't match %s" % str(tag.value))

        dtype = self._dtype
        shape = self._shape

        if not shape:
            return None

        image_width = self.image_width
        image_length = self.image_length
        typecode = self.parent.byteorder + dtype
        bits_per_sample = self.bits_per_sample
        byteorder_is_native = {"big": ">", "little": "<"}[
            sys.byteorder
        ] == self.parent.byteorder

        if self.is_tiled:
            if "tile_offsets" in self.tags:
                byte_counts = self.tile_byte_counts
                offsets = self.tile_offsets
            else:
                byte_counts = self.strip_byte_counts
                offsets = self.strip_offsets
            tile_width = self.tile_width
            tile_length = self.tile_length
            tw = (image_width + tile_width - 1) // tile_width
            tl = (image_length + tile_length - 1) // tile_length
            shape = shape[:-3] + (tl * tile_length, tw * tile_width, shape[-1])
            tile_shape = (tile_length, tile_width, shape[-1])
            runlen = tile_width
        else:
            byte_counts = self.strip_byte_counts
            offsets = self.strip_offsets
            runlen = image_width

        try:
            offsets[0]
        except TypeError:
            offsets = (offsets,)
            byte_counts = (byte_counts,)
        if any(o < 2 for o in offsets):
            raise ValueError("corrupted page")

        if not self.is_tiled and (
            self.is_stk
            or (
                not self.compression
                and bits_per_sample in (8, 16, 32, 64)
                and all(
                    offsets[i] == offsets[i + 1] - byte_counts[i]
                    for i in range(len(offsets) - 1)
                )
            )
        ):
            # contiguous data
            if memmap and not (
                self.is_tiled
                or self.predictor
                or ("extra_samples" in self.tags)
                or (colormapped and self.is_palette)
                or (not byteorder_is_native)
            ):
                result = numpy.memmap(fh, typecode, "r", offsets[0], shape)
            else:
                fh.seek(offsets[0])
                result = numpy_fromfile(fh, typecode, numpy.prod(shape))
                result = result.astype("=" + dtype)
        else:
            if self.planar_configuration == "contig":
                runlen *= self.samples_per_pixel
            if bits_per_sample in (8, 16, 32, 64, 128):
                if (bits_per_sample * runlen) % 8:
                    raise ValueError("data and sample size mismatch")

                def unpack(x):
                    return numpy.fromstring(x, typecode)

            elif isinstance(bits_per_sample, tuple):

                def unpack(x):
                    return unpackrgb(x, typecode, bits_per_sample)

            else:

                def unpack(x):
                    return unpackints(x, typecode, bits_per_sample, runlen)

            decompress = TIFF_DECOMPESSORS[self.compression]
            if self.is_tiled:
                result = numpy.empty(shape, dtype)
                tw, tl, pl = 0, 0, 0
                for offset, bytecount in zip(offsets, byte_counts):
                    fh.seek(offset)
                    tile = unpack(decompress(fh.read(bytecount)))
                    tile.shape = tile_shape
                    if self.predictor == "horizontal":
                        numpy.cumsum(tile, axis=-2, dtype=dtype, out=tile)
                    result[0, pl, tl : tl + tile_length, tw : tw + tile_width, :] = tile
                    del tile
                    tw += tile_width
                    if tw >= shape[-2]:
                        tw, tl = 0, tl + tile_length
                        if tl >= shape[-3]:
                            tl, pl = 0, pl + 1
                result = result[..., :image_length, :image_width, :]
            else:
                strip_size = (
                    self.rows_per_strip * self.image_width * self.samples_per_pixel
                )
                result = numpy.empty(shape, dtype).reshape(-1)
                index = 0
                for offset, bytecount in zip(offsets, byte_counts):
                    fh.seek(offset)
                    strip = fh.read(bytecount)
                    strip = unpack(decompress(strip))
                    size = min(result.size, strip.size, strip_size, result.size - index)
                    result[index : index + size] = strip[:size]
                    del strip
                    index += size

        result.shape = self._shape

        if self.predictor == "horizontal" and not self.is_tiled:
            # work around bug in LSM510 software
            if not (self.parent.is_lsm and not self.compression):
                numpy.cumsum(result, axis=-2, dtype=dtype, out=result)

        if colormapped and self.is_palette:
            if self.color_map.shape[1] >= 2 ** bits_per_sample:
                # FluoView and LSM might fail here
                result = numpy.take(self.color_map, result[:, 0, :, :, 0], axis=1)
        elif rgbonly and self.is_rgb and "extra_samples" in self.tags:
            # return only RGB and first alpha channel if exists
            extra_samples = self.extra_samples
            if self.tags["extra_samples"].count == 1:
                extra_samples = (extra_samples,)
            for i, exs in enumerate(extra_samples):
                if exs in ("unassalpha", "assocalpha", "unspecified"):
                    if self.planar_configuration == "contig":
                        result = result[..., [0, 1, 2, 3 + i]]
                    else:
                        result = result[:, [0, 1, 2, 3 + i]]
                    break
            else:
                if self.planar_configuration == "contig":
                    result = result[..., :3]
                else:
                    result = result[:, :3]

        if squeeze:
            try:
                result.shape = self.shape
            except ValueError:
                warnings.warn(
                    "failed to reshape from %s to %s"
                    % (str(result.shape), str(self.shape))
                )

        return result

    def __str__(self):
        """Return string containing information about page."""
        s = ", ".join(
            s
            for s in (
                " x ".join(str(i) for i in self.shape),
                str(numpy.dtype(self.dtype)),
                "%s bit" % str(self.bits_per_sample),
                self.photometric if "photometric" in self.tags else "",
                self.compression if self.compression else "raw",
                "|".join(
                    t[3:]
                    for t in (
                        "is_stk",
                        "is_lsm",
                        "is_nih",
                        "is_ome",
                        "is_imagej",
                        "is_micromanager",
                        "is_fluoview",
                        "is_mdgel",
                        "is_mediacy",
                        "is_reduced",
                        "is_tiled",
                    )
                    if getattr(self, t)
                ),
            )
            if s
        )
        return "Page %i: %s" % (self.index, s)

    def __getattr__(self, name):
        """Return tag value."""
        if name in self.tags:
            value = self.tags[name].value
            setattr(self, name, value)
            return value
        raise AttributeError(name)

    @lazyattr
    def is_rgb(self):
        """True if page contains a RGB image."""
        return "photometric" in self.tags and self.tags["photometric"].value == 2

    @lazyattr
    def is_palette(self):
        """True if page contains a palette-colored image."""
        return "photometric" in self.tags and self.tags["photometric"].value == 3

    @lazyattr
    def is_tiled(self):
        """True if page contains tiled image."""
        return "tile_width" in self.tags

    @lazyattr
    def is_reduced(self):
        """True if page is a reduced image of another image."""
        return bool(self.tags["new_subfile_type"].value & 1)

    @lazyattr
    def is_mdgel(self):
        """True if page contains md_file_tag tag."""
        return "md_file_tag" in self.tags

    @lazyattr
    def is_mediacy(self):
        """True if page contains Media Cybernetics Id tag."""
        return "mc_id" in self.tags and self.tags["mc_id"].value.startswith(b"MC TIFF")

    @lazyattr
    def is_stk(self):
        """True if page contains MM_UIC2 tag."""
        return "mm_uic2" in self.tags

    @lazyattr
    def is_lsm(self):
        """True if page contains LSM CZ_LSM_INFO tag."""
        return "cz_lsm_info" in self.tags

    @lazyattr
    def is_fluoview(self):
        """True if page contains FluoView MM_STAMP tag."""
        return "mm_stamp" in self.tags

    @lazyattr
    def is_nih(self):
        """True if page contains NIH image header."""
        return "nih_image_header" in self.tags

    @lazyattr
    def is_ome(self):
        """True if page contains OME-XML in image_description tag."""
        return "image_description" in self.tags and self.tags[
            "image_description"
        ].value.startswith(b"<?xml version=")

    @lazyattr
    def is_shaped(self):
        """True if page contains shape in image_description tag."""
        return "image_description" in self.tags and self.tags[
            "image_description"
        ].value.startswith(b"shape=(")

    @lazyattr
    def is_imagej(self):
        """True if page contains ImageJ description."""
        return (
            "image_description" in self.tags
            and self.tags["image_description"].value.startswith(b"ImageJ=")
        ) or (
            "image_description_1" in self.tags
            and self.tags["image_description_1"].value.startswith(  # Micromanager
                b"ImageJ="
            )
        )

    @lazyattr
    def is_micromanager(self):
        """True if page contains Micro-Manager metadata."""
        return "micromanager_metadata" in self.tags


class TiffTag(object):
    """A TIFF tag structure.

    Attributes
    ----------
    name : string
        Attribute name of tag.
    code : int
        Decimal code of tag.
    dtype : str
        Datatype of tag data. One of TIFF_DATA_TYPES.
    count : int
        Number of values.
    value : various types
        Tag data as Python object.
    value_offset : int
        Location of value in file, if any.

    All attributes are read-only.

    """

    __slots__ = (
        "code",
        "name",
        "count",
        "dtype",
        "value",
        "value_offset",
        "_offset",
        "_value",
    )

    class Error(Exception):
        pass

    def __init__(self, arg, **kwargs):
        """Initialize instance from file or arguments."""
        self._offset = None
        if hasattr(arg, "_fh"):
            self._fromfile(arg, **kwargs)
        else:
            self._fromdata(arg, **kwargs)

    def _fromdata(self, code, dtype, count, value, name=None):
        """Initialize instance from arguments."""
        self.code = int(code)
        self.name = name if name else str(code)
        self.dtype = TIFF_DATA_TYPES[dtype]
        self.count = int(count)
        self.value = value

    def _fromfile(self, parent):
        """Read tag structure from open file. Advance file cursor."""
        fh = parent._fh
        byteorder = parent.byteorder
        self._offset = fh.tell()
        self.value_offset = self._offset + parent.offset_size + 4

        fmt, size = {4: ("HHI4s", 12), 8: ("HHQ8s", 20)}[parent.offset_size]
        data = fh.read(size)
        code, dtype = struct.unpack(byteorder + fmt[:2], data[:4])
        count, value = struct.unpack(byteorder + fmt[2:], data[4:])
        self._value = value

        if code in TIFF_TAGS:
            name = TIFF_TAGS[code][0]
        elif code in CUSTOM_TAGS:
            name = CUSTOM_TAGS[code][0]
        else:
            name = str(code)

        try:
            dtype = TIFF_DATA_TYPES[dtype]
        except KeyError:
            raise TiffTag.Error("unknown tag data type %i" % dtype)

        fmt = "%s%i%s" % (byteorder, count * int(dtype[0]), dtype[1])
        size = struct.calcsize(fmt)
        if size > parent.offset_size or code in CUSTOM_TAGS:
            pos = fh.tell()
            tof = {4: "I", 8: "Q"}[parent.offset_size]
            self.value_offset = offset = struct.unpack(byteorder + tof, value)[0]
            if offset < 0 or offset > parent._fsize:
                raise TiffTag.Error("corrupt file - invalid tag value offset")
            elif offset < 4:
                raise TiffTag.Error("corrupt value offset for tag %i" % code)
            fh.seek(offset)
            if code in CUSTOM_TAGS:
                readfunc = CUSTOM_TAGS[code][1]
                value = readfunc(fh, byteorder, dtype, count)
                fh.seek(0, 2)  # bug in numpy/Python 3.x ?
                if isinstance(value, dict):  # numpy.core.records.record
                    value = Record(value)
            elif code in TIFF_TAGS or dtype[-1] == "s":
                value = struct.unpack(fmt, fh.read(size))
            else:
                value = read_numpy(fh, byteorder, dtype, count)
                fh.seek(0, 2)  # bug in numpy/Python 3.x ?
            fh.seek(pos)
        else:
            value = struct.unpack(fmt, value[:size])

        if not code in CUSTOM_TAGS:
            if len(value) == 1:
                value = value[0]

        if dtype.endswith("s") and isinstance(value, bytes):
            value = stripnull(value)

        self.code = code
        self.name = name
        self.dtype = dtype
        self.count = count
        self.value = value

    def __str__(self):
        """Return string containing information about tag."""
        return " ".join(str(getattr(self, s)) for s in self.__slots__)


class TiffSequence(object):
    """Sequence of image files.

    Properties
    ----------
    files : list
        List of file names.
    shape : tuple
        Shape of image sequence.
    axes : str
        Labels of axes in shape.

    Examples
    --------
    >>> ims = TiffSequence("test.oif.files/*.tif")
    >>> ims = ims.asarray()
    >>> ims.shape
    (2, 100, 256, 256)

    """

    _axes_pattern = """
        # matches Olympus OIF and Leica TIFF series
        _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))
        _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
        _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
        _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
        _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
        _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
        _?(?:(q|l|p|a|c|t|x|y|z|ch|tp)(\d{1,4}))?
        """

    class _ParseError(Exception):
        pass

    def __init__(self, files, imread=TiffFile, pattern="axes"):
        """Initialize instance from multiple files.

        Parameters
        ----------
        files : str, or sequence of str
            Glob pattern or sequence of file names.
        imread : function or class
            Image read function or class with asarray function returning numpy
            array from single file.
        pattern : str
            Regular expression pattern that matches axes names and sequence
            indices in file names.

        """
        if isinstance(files, basestring):
            files = natural_sorted(glob.glob(files))
        files = list(files)
        if not files:
            raise ValueError("no files found")
        # if not os.path.isfile(files[0]):
        #    raise ValueError("file not found")
        self.files = files

        if hasattr(imread, "asarray"):
            _imread = imread

            def imread(fname, *args, **kwargs):
                with _imread(fname) as im:
                    return im.asarray(*args, **kwargs)

        self.imread = imread

        self.pattern = self._axes_pattern if pattern == "axes" else pattern
        try:
            self._parse()
            if not self.axes:
                self.axes = "I"
        except self._ParseError:
            self.axes = "I"
            self.shape
