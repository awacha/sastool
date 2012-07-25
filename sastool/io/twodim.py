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

import os
import numpy as np
import scipy.misc
import scipy.io

import header

from _io import cbfdecompress # IGNORE:E0611
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
    return np.loadtxt(filename, skiprows = 133)

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
    return header.readPAXE(filename, load_data = True)

def readcbf(name):
    """Read a cbf (crystallographic binary format) file from a Dectris PILATUS
    detector.
    
    Inputs
    ------
    name: string
        the file name
    
    Output
    ------
    a numpy array of the scattering data
        
    Notes
    -----
    currently only Little endian, "signed 32-bit integer" type and
    byte-offset compressed data are accepted.
    """
    def getvaluefromheader(hed, caption, separator = ':'):
        tmp = [x.split(separator)[1].strip() for x in hed if x.startswith(caption)]
        if len(tmp) == 0:
            raise ValueError ('Caption %s not present in CBF header!' % caption)
        else:
            return tmp[0]
    f = open(name, 'rb')
    cbfbin = f.read()
    f.close()
    datastart = cbfbin.find('\x0c\x1a\x04\xd5') + 4
    hed = [x.strip() for x in cbfbin[:datastart].split('\n')]
    if getvaluefromheader(hed, 'X-Binary-Element-Type') != '"signed 32-bit integer"':
        raise NotImplementedError('element type is not "signed 32-bit integer" in CBF, but %s.' % getvaluefromheader(header, 'X-Binary-Element-Type'))
    if getvaluefromheader(hed, 'conversions', '=') != '"x-CBF_BYTE_OFFSET"':
        raise NotImplementedError('compression is not "x-CBF_BYTE_OFFSET" in CBF!')
    dim1 = long(getvaluefromheader(hed, 'X-Binary-Size-Fastest-Dimension'))
    dim2 = long(getvaluefromheader(hed, 'X-Binary-Size-Second-Dimension'))
    nbytes = long(getvaluefromheader(hed, 'X-Binary-Size'))
    return cbfdecompress(cbfbin[datastart:datastart + nbytes], dim1, dim2)

def readbdfv1(filename, bdfext = '.bdf', bhfext = '.bhf'):
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
    if filename.upper().endswith('.MAT'): #Matlab
        m = scipy.io.loadmat(filename)
    elif filename.upper().endswith('.NPZ'): #Numpy
        m = np.load(filename)
    else: #loadtxt
        m = {'Intensity':np.loadtxt(filename)}
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

def writeint2dnorm(filename, Intensity, Error = None):
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
    whattosave = {'Intensity':Intensity}
    if Error is not None:
        whattosave['Error'] = Error
    if filename.upper().endswith('.NPZ'):
        np.savez(filename, **whattosave)
    elif filename.upper().endswith('.MAT'):
        scipy.io.savemat(filename, whattosave)
    else: #text file
        np.savetxt(filename, Intensity)
        if Error is not None:
            name, ext = os.path.splitext(filename)
            np.savetxt(name + '_error' + ext, Error)

def readmask(filename, fieldname = None):
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
        validkeys = [k for k in f.keys() if not (k.startswith('_') and k.endswith('_'))];
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
        raise NotImplementedError('Not supported data type: %s' % edf['DataType'])
    edf['data'] = np.fromstring(f.read(edf['EDF_BinarySize']), dtype).reshape(edf['Dim_1'], edf['Dim_2'])
    return edf

def readbdfv2(filename, bdfext = '.bdf', bhfext = '.bhf'):
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

def readbdf(filename, bdfext = '.bdf', bhfext = '.bhf'):
    return header.readbhf(filename, True, bdfext, bhfext)

def writebdfv2(filename, bdf, bdfext = '.bdf', bhfext = '.bhf'):
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
    keys.extend([x for x in bdf.keys() if isinstance(bdf[x], np.ndarray) and x not in keys])
    for k in keys:
        if k not in bdf.keys():
            continue
        f.write('#%s[%d:%d]\n' % (k, bdf['xdim'], bdf['ydim']))
        f.write(np.rot90(bdf[k], 3).astype('float32').tostring(order = 'F'))
    f.close()

def rebinmask(mask, binx, biny, enlarge = False):
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
        raise ValueError('The number of pixels of the mask matrix should be divisible by the binning in each direction!')
    if enlarge:
        return mask.repeat(binx, axis = 0).repeat(biny, axis = 1)
    else:
        return mask[::binx, ::biny]

