'''
Low-level I/O for 2D data

This module contains low-level read/write procedures for two-dimensional data
(scattering patterns, masks etc.). Read functions have the signature::

    def read<what>(filename, <other args>)

and return the loaded data in a structure passing for the original logic of the
file, whereas writer functions look like::

    def write<what>(filename, <data>, <other args>)

and accept the data in a a format as returned by the corresponding reader.

Matrices are usually represented as two-dimensional `numpy.ndarray`-s.

'''
from __future__ import absolute_import

import datetime
import os
import re
import sys

import dateutil.parser
import numpy as np
import scipy.io
import scipy.misc

from . import header
from .c_io import cbfdecompress  # IGNORE:E0611

if sys.version_info[0] == 2:
    from io import open
"""Decompress algorithm for the byte-offset encoding found in CBF files.
Implemented in `Cython` for the sake of speed."""


def readjusifaorg(filename):
    """Read an original ASCII scattering data file (measured at the beamline
    B1/JUSIFA, HASYLAB, Hamburg).

    Inputs
    ------
    filename: string
        the name of the input file

    Outputs
    -------
    the scattering pattern in an N-times-8 shape (rows-times-columns)

    Notes
    -----
    It simply skips the first 133 lines (header data) and loads the rest by
    ``np.loadtxt``. No reshaping is done, it is the responsibility of the
    caller.
    """
    return np.loadtxt(filename, skiprows=133)


def readPAXE(filename):
    """Read an exposure measured at the PAXE instrument of LLB, Saclay, France
    or at the Yellow Submarine SANS instrument of BNC, Budapest, Hungary.

    Inputs
    ------
    filename: string
        the name of the input file

    Outputs
    -------
    header: dict
        the header data
    data: np.ndarray
        the scattering matrix

    Notes
    -----
    The original 16-bit files and 32-bit ones can also be loaded. The type is
    determined by the extension of the filename (``.32`` is the 32-bit type,
    otherwise 16-bit is assumed)
    """
    return header.readPAXE(filename, load_data=True)


def readcbf(name, load_header=False, load_data=True, for_nexus=False):
    """Read a cbf (crystallographic binary format) file from a Dectris PILATUS
    detector.

    Inputs
    ------
    name: string
        the file name
    load_header: bool
        if the header data is to be loaded.
    load_data: bool
        if the binary data is to be loaded.
    for_nexus: bool
        if the array should be opened with NeXus ordering.

    Output
    ------
    a numpy array of the scattering data

    Notes
    -----
    currently only Little endian, "signed 32-bit integer" type and
    byte-offset compressed data are accepted.
    """
    with open(name, 'rb') as f:
        cbfbin = f.read()
    datastart = cbfbin.find(b'\x0c\x1a\x04\xd5') + 4
    hed = [x.strip() for x in cbfbin[:datastart].split(b'\n')]
    header = {}
    readingmode = None
    for i in range(len(hed)):
        if not hed[i]:
            # skip empty header lines
            continue
        elif hed[i] == b';':
            continue
        elif hed[i].startswith(b'_array_data.header_convention'):
            header['CBF_header_convention'] = str(hed[i][
                len(b'_array_data.header_convention'):].strip().replace(b'"', b''), encoding='utf-8')
        elif hed[i].startswith(b'_array_data.header_contents'):
            readingmode = 'PilatusHeader'
        elif hed[i].startswith(b'_array_data.data'):
            readingmode = 'CIFHeader'
        elif readingmode == 'PilatusHeader':
            if not hed[i].startswith(b'#'):
                continue
            line = hed[i].strip()[1:].strip()
            try:
                # try to interpret the line as the date.
                header['CBF_Date'] = dateutil.parser.parse(line)
                header['Date'] = header['CBF_Date']
                continue
            except (ValueError, TypeError):
                # eat exception: if we cannot parse this line as a date, try
                # another format.
                pass
            treated = False
            for sep in (b':', b'='):
                if treated:
                    continue
                if line.count(sep) == 1:
                    name, value = tuple(x.strip() for x in line.split(sep, 1))
                    try:
                        m = re.match(
                            b'^(?P<number>-?(\d+(.\d+)?(e-?\d+)?))\s+(?P<unit>m|s|counts|eV)$', value).groupdict()
                        value = float(m['number'])
                        m['unit'] = str(m['unit'], encoding='utf-8')
                    except AttributeError:
                        # the regex did not match the string, thus re.match()
                        # returned None.
                        pass
                    header[str(name, 'utf-8')] = value
                    treated = True
            if treated:
                continue
            if line.startswith(b'Pixel_size'):
                header['XPixel'], header['YPixel'] = tuple(
                    [float(a.strip().split(b' ')[0]) * 1000 for a in line[len(b'Pixel_size'):].split(b'x')])
            else:
                try:
                    m = re.match(
                        b'^(?P<label>[a-zA-Z0-9,_\.\-!\?\ ]*?)\s+(?P<number>-?(\d+(.\d+)?(e-?\d+)?))\s+(?P<unit>m|s|counts|eV)$', line).groupdict()
                except AttributeError:
                    pass
                else:
                    m['label'] = str(m['label'], 'utf-8')
                    m['unit'] = str(m['unit'], encoding='utf-8')
                    if m['unit'] == b'counts':
                        header[m['label']] = int(m['number'])
                    else:
                        header[m['label']] = float(m['number'])
                    if 'sensor' in m['label'] and 'thickness' in m['label']:
                        header[m['label']] *= 1e6
        elif readingmode == 'CIFHeader':
            line = hed[i]
            for sep in (b':', b'='):
                if line.count(sep) == 1:
                    label, content = tuple(x.strip()
                                           for x in line.split(sep, 1))
                    if b'"' in content:
                        content = content.replace(b'"', b'')
                    try:
                        content = int(content)
                    except ValueError:
                        content = str(content, encoding='utf-8')
                    header['CBF_' + str(label, encoding='utf-8')] = content

        else:
            pass
    ret = []
    if load_data:
        if header['CBF_X-Binary-Element-Type'] != 'signed 32-bit integer':
            raise NotImplementedError(
                'element type is not "signed 32-bit integer" in CBF, but %s.' % header['CBF_X-Binary-Element-Type'])
        if header['CBF_conversions'] != 'x-CBF_BYTE_OFFSET':
            raise NotImplementedError(
                'compression is not "x-CBF_BYTE_OFFSET" in CBF!')
        dim1 = header['CBF_X-Binary-Size-Fastest-Dimension']
        dim2 = header['CBF_X-Binary-Size-Second-Dimension']
        nbytes = header['CBF_X-Binary-Size']
        cbfdata = cbfdecompress(
            bytearray(cbfbin[datastart:datastart + nbytes]), dim1, dim2, for_nexus)
        ret.append(cbfdata)
    if load_header:
        ret.append(header)
    return tuple(ret)


def readbdfv1(filename, bdfext='.bdf', bhfext='.bhf'):
    """Read bdf file (Bessy Data Format v1)

    Input
    -----
    filename: string
        the name of the file

    Output
    ------
    the BDF structure in a dict

    Notes
    -----
    This is an adaptation of the bdf_read.m macro of Sylvio Haas.
    """
    return header.readbhfv1(filename, True, bdfext, bhfext)


def readtif(filename):
    """Read image files (TIFF, JPEG, PNG... supported by PIL).

    Input
    -----
    filename: string
        the name of the file

    Output
    ------
    the image in a ``np.ndarray``

    Notes
    -----
    ``scipy.misc.imread()`` is used, which in turn depends on PIL.
    """
    return scipy.misc.imread(filename, True)


def readint2dnorm(filename):
    """Read corrected intensity and error matrices (Matlab mat or numpy npz
    format for Beamline B1 (HASYLAB/DORISIII))

    Input
    -----
    filename: string
        the name of the file

    Outputs
    -------
    two ``np.ndarray``-s, the Intensity and the Error matrices

    File formats supported:
    -----------------------

    ``.mat``
        Matlab MAT file, with (at least) two fields: Intensity and Error

    ``.npz``
        Numpy zip file, with (at least) two fields: Intensity and Error

    other
        the file is opened with ``np.loadtxt``. The error matrix is tried
        to be loaded from the file ``<name>_error<ext>`` where the intensity was
        loaded from file ``<name><ext>``. I.e. if ``somedir/matrix.dat`` is given,
        the existence of ``somedir/matrix_error.dat`` is checked. If not found,
        None is returned for the error matrix.

    Notes
    -----
    The non-existence of the Intensity matrix results in an exception. If the
    Error matrix does not exist, None is returned for it.
    """
    # the core of read2dintfile
    if filename.upper().endswith('.MAT'):  # Matlab
        m = scipy.io.loadmat(filename)
    elif filename.upper().endswith('.NPZ'):  # Numpy
        m = np.load(filename)
    else:  # loadtxt
        m = {'Intensity': np.loadtxt(filename)}
        name, ext = os.path.splitext(filename)
        errorfilename = name + '_error' + ext
        if os.path.exists(errorfilename):
            m['Error'] = np.loadtxt(errorfilename)
    Intensity = m['Intensity']
    try:
        Error = m['Error']
        return Intensity, Error
    except:
        return Intensity, None


def writeint2dnorm(filename, Intensity, Error=None):
    """Save the intensity and error matrices to a file

    Inputs
    ------
    filename: string
        the name of the file
    Intensity: np.ndarray
        the intensity matrix
    Error: np.ndarray, optional
        the error matrix (can be ``None``, if no error matrix is to be saved)

    Output
    ------
    None
    """
    whattosave = {'Intensity': Intensity}
    if Error is not None:
        whattosave['Error'] = Error
    if filename.upper().endswith('.NPZ'):
        np.savez(filename, **whattosave)
    elif filename.upper().endswith('.MAT'):
        scipy.io.savemat(filename, whattosave)
    else:  # text file
        np.savetxt(filename, Intensity)
        if Error is not None:
            name, ext = os.path.splitext(filename)
            np.savetxt(name + '_error' + ext, Error)


def readmask(filename, fieldname=None):
    """Try to load a maskfile from a matlab(R) matrix file

    Inputs
    ------
    filename: string
        the input file name
    fieldname: string, optional
        field in the mat file. None to autodetect.

    Outputs
    -------
    the mask in a numpy array of type np.uint8
    """
    f = scipy.io.loadmat(filename)
    if fieldname is not None:
        return f[fieldname].astype(np.uint8)
    else:
        validkeys = [
            k for k in list(f.keys()) if not (k.startswith('_') and k.endswith('_'))]
        if len(validkeys) < 1:
            raise ValueError('mask file contains no masks!')
        if len(validkeys) > 1:
            raise ValueError('mask file contains multiple masks!')
        return f[validkeys[0]].astype(np.uint8)


def readedf(filename):
    """Read an ESRF data file (measured at beamlines ID01 or ID02)

    Inputs
    ------
    filename: string
        the input file name

    Output
    ------
    the imported EDF structure in a dict. The scattering pattern is under key
    'data'.

    Notes
    -----
    Only datatype ``FloatValue`` is supported right now.
    """
    edf = header.readehf(filename)
    f = open(filename, 'rb')
    f.read(edf['EDF_HeaderSize'])  # skip header.
    if edf['DataType'] == 'FloatValue':
        dtype = np.float32
    else:
        raise NotImplementedError(
            'Not supported data type: %s' % edf['DataType'])
    edf['data'] = np.fromstring(f.read(edf['EDF_BinarySize']), dtype).reshape(
        edf['Dim_1'], edf['Dim_2'])
    return edf


def readbdfv2(filename, bdfext='.bdf', bhfext='.bhf'):
    """Read a version 2 Bessy Data File

    Inputs
    ------
    filename: string
        the name of the input file. One can give the complete header or datafile
        name or just the base name without the extensions.
    bdfext: string, optional
        the extension of the data file
    bhfext: string, optional
        the extension of the header file

    Output
    ------
    the data structure in a dict. Header is loaded implicitely.

    Notes
    -----
    BDFv2 header and scattering data are stored separately in the header and
    the data files. Given the file name both are loaded.
    """
    datas = header.readbhfv2(filename, True, bdfext, bhfext)
    return datas


def readbdf(filename, bdfext='.bdf', bhfext='.bhf'):
    return header.readbhf(filename, True, bdfext, bhfext)


def readmar(filename):
    """Read a two-dimensional scattering pattern from a MarResearch .image file.
    """
    hed = header.readmarheader(filename)
    with open(filename, 'rb') as f:
        h = f.read(hed['recordlength'])
        data = np.fromstring(
            f.read(2 * hed['Xsize'] * hed['Ysize']), '<u2').astype(np.float64)
        if hed['highintensitypixels'] > 0:
            raise NotImplementedError(
                'Intensities over 65535 are not yet supported!')
        data = data.reshape(hed['Xsize'], hed['Ysize'])
    return data, hed


def writebdfv2(filename, bdf, bdfext='.bdf', bhfext='.bhf'):
    """Write a version 2 Bessy Data File

    Inputs
    ------
    filename: string
        the name of the output file. One can give the complete header or
        datafile name or just the base name without the extensions.
    bdf: dict
        the BDF structure (in the same format as loaded by ``readbdfv2()``
    bdfext: string, optional
        the extension of the data file
    bhfext: string, optional
        the extension of the header file

    Output
    ------
    None

    Notes
    -----
    BDFv2 header and scattering data are stored separately in the header and
    the data files. Given the file name both are saved.
    """
    if filename.endswith(bdfext):
        basename = filename[:-len(bdfext)]
    elif filename.endswith(bhfext):
        basename = filename[:-len(bhfext)]
    else:
        basename = filename
    header.writebhfv2(basename + '.bhf', bdf)
    f = open(basename + '.bdf', 'wb')
    keys = ['RAWDATA', 'RAWERROR', 'CORRDATA', 'CORRERROR', 'NANDATA']
    keys.extend(
        [x for x in list(bdf.keys()) if isinstance(bdf[x], np.ndarray) and x not in keys])
    for k in keys:
        if k not in list(bdf.keys()):
            continue
        f.write('#%s[%d:%d]\n' % (k, bdf['xdim'], bdf['ydim']))
        f.write(np.rot90(bdf[k], 3).astype('float32').tostring(order='F'))
    f.close()


def rebinmask(mask, binx, biny, enlarge=False):
    """Re-bin (shrink or enlarge) a mask matrix.

    Inputs
    ------
    mask: np.ndarray
        mask matrix.
    binx: integer
        binning along the 0th axis
    biny: integer
        binning along the 1st axis
    enlarge: bool, optional
        direction of binning. If True, the matrix will be enlarged, otherwise
        shrinked (this is the default)

    Output
    ------
    the binned mask matrix, of shape ``M/binx`` times ``N/biny`` or ``M*binx``
    times ``N*biny``, depending on the value of ``enlarge`` (if ``mask`` is
    ``M`` times ``N`` pixels).

    Notes
    -----
    one is nonmasked, zero is masked.
    """
    if not enlarge and ((mask.shape[0] % binx) or (mask.shape[1] % biny)):
        raise ValueError(
            'The number of pixels of the mask matrix should be divisible by the binning in each direction!')
    if enlarge:
        return mask.repeat(binx, axis=0).repeat(biny, axis=1)
    else:
        return mask[::binx, ::biny]


def readBerSANSdata(filename):
    hed = header.readBerSANS(filename)
    if hed['Type'] not in ['SANSDraw', 'SANSDAni']:
        raise ValueError('Invalid file type: ' + hed['Type'])
    datasize = int(hed['DataSize'] ** 0.5)
    with open(filename, 'rt') as f:
        l = f.readline()
        counts = []
        errors = []
        while not l.startswith('%Counts') and len(l) > 0:
            l = f.readline()
        l = f.readline()
        while not l.startswith('%') and len(l) > 0:
            counts.append([float(x) for x in l.replace(',', ' ').replace(
                '-', ' -').replace('e -', 'e-').replace('E -', 'E-').split()])
            l = f.readline()
        l = f.readline()
        while len(l) > 0:
            errors.append([float(x) for x in l.replace(',', ' ').replace(
                '-', ' -').replace('e -', 'e-').replace('E -', 'E-').split()])
            l = f.readline()
        c = np.array(counts, dtype=np.float64)
        e = np.array(errors, dtype=np.float64)
        if c.size == hed['DataSize']:
            c = c.reshape(datasize, datasize)
        else:
            raise ValueError('Invalid size of the counts matrix!')
        if e.size == hed['DataSize']:
            e = e.reshape(datasize, datasize)
        elif e.size == 0:
            e = None
        else:
            raise ValueError('Invalid size of the error matrix!')
    return c, e, hed


def readBerSANSmask(filename):
    with open(filename, 'rt') as f:
        l = f.readline()
        while not l.startswith('%Mask') and len(l) > 0:
            l = f.readline()
        m = f.read()
    mask = np.array([[ord(y) for y in x] for x in m.split()], np.uint8)
    mask[mask == 45] = 1
    mask[mask == 35] = 0
    return mask


def writeBerSANSmask(filename, maskmatrix):
    with open(filename, 'wt') as f:
        f.write('%File\n')
        f.write('FileName=%s\n' % filename)
        d = datetime.datetime.now()
        month = [None, 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        f.write('FileDate=%02d-%s-%04d\n' % (d.day, month[d.month], d.year))
        f.write('FileTime=%02d:%02d:%02d\n' % (d.hour, d.minute, d.second))
        f.write('Type=SANSMAni\n')
        f.write('DataSize=%d\n' % maskmatrix.size)
        f.write('%Mask\n')
        maskmatrix = (maskmatrix != 0) * 1
        data = '\n'.join([''.join([str(a) for a in x])
                          for x in maskmatrix.tolist()])
        f.write(data.replace('0', '#').replace('1', '-') + '\n')
