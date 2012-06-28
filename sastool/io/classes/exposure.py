'''
Created on Jun 15, 2012

@author: andris
'''


import numpy as np
import warnings
import numbers
import matplotlib.pyplot as plt
import matplotlib.colors
import collections
import os
import re
import h5py

from .header import SASHeader
from .common import SASExposureException, _HDF_parse_group
from .mask import SASMask
from ... import misc
from .. import twodim
from ... import utils2d
from ... import dataset
from ...dataset import ArithmeticBase, ErrorValue

import scipy.constants
#Planck constant times speed of light in eV*Angstroem units
HC = scipy.constants.codata.value('Planck constant in eV s') * \
    scipy.constants.codata.value('speed of light in vacuum') * 1e10

class SASExposure(object):
    """A class for holding SAS exposure data, i.e. intensity, error, metadata
    and mask.
    
    A SASExposure has the following special attributes:
    
    Intensity: scattered intensity matrix (np.ndarray).
    Error: error of scattered intensity (np.ndarray).
    header: metadata dictionary (SASHeader instance).
    mask: mask matrix in the form of a SASMask instance. 
    
    any of the above attributes can be missing, a value of None signifies
    this situation.

    
    """
    matrix_names = ['Intensity', 'Error']
    matrices = dict([('Intensity', 'Scattered intensity'),
                                      ('Error', 'Error of intensity')])
    @staticmethod
    def _autoguess_experiment_type(fileformat_or_name):
        if isinstance(fileformat_or_name, basestring):
            fileformat_or_name = os.path.split(fileformat_or_name)[1].upper()
            if fileformat_or_name.endswith('.EDF') or \
                fileformat_or_name.endswith('CCD'):
                return 'read_from_ESRF_ID02'
            elif fileformat_or_name.startswith('ORG'):
                return 'read_from_B1_org'
            elif fileformat_or_name.startswith('INT2DNORM'):
                return 'read_from_B1_int2dnorm'
            elif fileformat_or_name.endswith('.H5') or \
                fileformat_or_name.endswith('.HDF5') or \
                fileformat_or_name.endswith('.HDF'):
                return 'read_from_HDF5'
            elif fileformat_or_name.endswith('.BDF') or \
                fileformat_or_name.endswith('.BHF'):
                return 'read_from_BDF'
            elif fileformat_or_name.startswith('XE') and \
                (fileformat_or_name.endswith('.DAT') or \
                 fileformat_or_name.endswith('.32')):
                return 'read_from_PAXE'
        elif isinstance(fileformat_or_name, h5py.highlevel.Group):
            return 'read_from_HDF5'
        else:
            raise ValueError('Unknown measurement file format')
    @staticmethod
    def _set_default_kwargs_for_readers(kwargs):
        if 'dirs' not in kwargs:
            kwargs['dirs'] = None
        if 'load_mask' not in kwargs:
            kwargs['load_mask'] = True
        if 'estimate_errors' not in kwargs:
            kwargs['estimate_errors'] = True
        if 'maskfile' not in kwargs:
            kwargs['maskfile'] = None
        if 'experiment_type' not in kwargs:
            kwargs['experiment_type'] = None
        if 'error_on_not_found' not in kwargs:
            kwargs['error_on_not_found'] = True
        if 'HDF5_Groupnameformat' not in kwargs:
            kwargs['HDF5_Groupnameformat'] = 'FSN%d'
        if 'HDF5_Intensityname' not in kwargs:
            kwargs['HDF5_Intensityname'] = 'Intensity'
        if 'HDF5_Errorname' not in kwargs:
            kwargs['HDF5_Errorname'] = 'Error'
        return kwargs

    def __init__(self, *args, **kwargs):
        """Initialize the already constructed instance. This is hardly ever
        called directly since `__new__()` is implemented. See there for the
        usual cases of object construction.
        """
        kwargs = SASExposure._set_default_kwargs_for_readers(kwargs)

        if not args: #no positional arguments:
            super(SASExposure, self).__init__()
            self.Intensity = None
            self.Error = None
            self.header = SASHeader()
            self.mask = None
        elif len(args) == 1 and isinstance(args[0], SASExposure):
            self.Intensity = args[0].Intensity.copy()
            self.Error = args[0].Error.copy()
            self.header = SASHeader(args[0].header)
            self.mask = SASMask(args[0].mask)
        elif (len(args) <= 2): #scheme 2) or 3) or 4) with single FSN or direct file load
            self.Intensity = None
            self.Error = None
            self.header = SASHeader()
            self.mask = None
            if len(args) == 1 and isinstance(args[0], basestring):
                filename = args[0]
            elif isinstance(args[0], h5py.highlevel.Group) and len(args) > 1:
                # interpolate FSN
                filename = args[0][kwargs['HDF5_Groupnameformat'] % args[1]]
            elif isinstance(args[0], basestring):
                #and len(args) >1, which is true because the case in which is 
                # false has already been treated.

                try:
                    filename = args[0] % args[1]
                except TypeError:
                    # if args[0] is not a printf-style format string, try to
                    #load it as a HDF5 file
                    with h5py.highlevel.File(args[0]) as hdf_file:
                        self.read_from_HDF5(hdf_file[kwargs['HDF5_Groupnameformat'] % args[1]], **kwargs)
                        filename = None
            else:
                # use this HDF5 group as is.
                filename = args[0]
            if kwargs['experiment_type'] is None:
                #auto-guess from filename
                loadername = SASExposure._autoguess_experiment_type(args[0])
            else:
                loadername = 'read_from_%s' % kwargs['experiment_type']
            if filename is not None:
                try:
                    getattr(self, loadername).__call__(filename, **kwargs)
                except AttributeError as ae:
                    raise AttributeError(str(ae) + '; possibly bad experiment type given')
                #other exceptions such as IOError on read failure are propagated.
            if kwargs['maskfile'] is not None:
                self.set_mask(SASMask(kwargs['maskfile'], dirs = kwargs['dirs']))
        else:
            raise ValueError('Too many positional arguments!')
    def __new__(cls, *args, **kwargs):
        """Load files or just create an empty instance or copy.
        
        This function serves as a general method for instantiating 
        `SASExposure`, i.e. handles calls, like
        ::
            >>> SASExposure(...)
        
        Three ways of working (note that positional arguments *cannot be given
        as keyword ones*!):
        
        0) ``SASExposure()``: create an empty object.
        
        1) ``SASExposure(<instance-of-SASExposure>)``: copy-constructor
            **One positional argument**, which is an instance of `SASExposure`.
            In this case an instance of `SASExposure` is returned. The copy is
            shallow!
        
        2) ``SASExposure(filename, **kwargs)``: load a file
            **One positional argument**, a string
            `filename`:  string
                file name to load (with path if needed)
        
        3) ``SASExposure(fileformat, fsn, **kwargs)``: load
            one or more files. **Two positional arguments**.
            `fileformat`: string
                C-style file format, containing a directive for substitution of
                a number, i.e. ``org_%05d.cbf`` or ``s%07d.bdf`` or
                ``sc3269_0_%04dccd``
            `fsn`: number or a sequence of numbers
                file sequence number. If a scalar, the corresponding file will
                be loaded. If a sequence of numbers, each file will be opened
                and a list of SASExposure instances will be returned.
        
        4) ``SASExposure(hdf_group, [fsn,] **kwargs)``: load exposures from a HDF5 file
            produced by `sastool`. **One positional argument**, a hdf5 filename
            or a hdf5 group (instance of ``h5py.highlevel.Group``). Supported
            keyword arguments:
            `HDF5_Groupnameformat`: a regular expression pattern compatible with
                the `re` module to match and parse group names. Should contain a
                parsed regular expression group for the FSN. For example the
                default value is ``r"FSN(\d+)"``. Typically the pattern should 
                contain a construct like ``(\d+)`` to match file sequence numbers.
            `HDF5_Intensityname`: the name of the dataset in each group which
                corresponds to the scattered intensity. Defaults to
                ``'Intensity'``.
            `HDF5_Errorname`: the name of the dataset in each group which
                corresponds to the error of the scattered intensity. Defaults
                to ``Error``.
        
        For signatures 2), 3) and 4) the following optional keyword arguments
        are supported:
        
            `experiment_type`: string
                the origin of the file, which determines its format. It is
                normally auto-guessed, but in case that fails, one can forcibly
                define it here. See read_from_xxxx() method names for available
                values.
            `error_on_not_found`: `bool`, **only for calling scheme 3) and 4)**
                if an `IOError` is to be raised if the file is not found or not
                readable. Defaults to `True`.
            `maskfile`: string
                name of the mask file, with path if needed. The search
                path in `dirs` is checked. Defaults to `None`.
            `dirs`: list of directories
                None or a list of directories to search files for (uses
                `sastool.misc.findfileindirs()` for this purpose). Can be set
                to `None`, which means the current folder and the `sastool`
                search path. Defaults to `None`
            `load_mask`: if a mask has to be loaded. Defaults to ``True``
            `load_header`: if the metadata are to be loaded. ``True`` by default.
        """
        kwargs = SASExposure._set_default_kwargs_for_readers(kwargs)
        if len(args) < 2: #scheme 0) 1) and 2) can be handled in the same way 
            if isinstance(args[0], h5py.highlevel.Group) or \
              (isinstance(args[0], basestring) and any([args[0].upper().endswith(k) for k in ['.HDF5', '.HDF', '.H5']])) or \
              kwargs['experiment_type'] == 'HDF5':
                pattern = misc.re_from_Cformatstring_numbers(kwargs['HDF5_Groupnameformat'])
                objlist = []
                with _HDF_parse_group(args[0]) as hdf_group:
                    for k in hdf_group.keys():
                        if re.match(pattern, k) is not None:
                            obj = super(SASExposure, cls).__new__(cls)
                            try:
                                obj.__init__(hdf_group[k], **kwargs)
                                objlist.append(obj)
                            except:
                                if kwargs['error_on_not_found']:
                                    raise
                                else:
                                    objlist.append(None)
                return objlist
            else:
                obj = super(SASExposure, cls).__new__(cls)
                return obj # this will call obj.__init__(*args, **kwargs) implicitly
        elif len(args) == 2: # scheme 3) or scheme 4)
            if isinstance(args[0], basestring) or isinstance(args[0], h5py.highlevel.Group):
                fsns = args[1]
                fileformat = args[0]
                if isinstance(fsns, collections.Sequence):
                    #multi-FSN
                    objlist = []
                    if any([fileformat.upper().endswith(k) for k in ['.HDF5', '.HDF', '.H5']]):
                        kwargs['experiment_type'] = 'HDF5'
                        hdf = h5py.highlevel.File(fileformat) # open the file
                    elif isinstance(fileformat, h5py.highlevel.Group):
                        hdf = fileformat
                    else:
                        hdf = None
                    for f in fsns:
                        obj = super(SASExposure, cls).__new__(cls)
                        try:
                            if hdf is not None:
                                obj.__init__(hdf[kwargs['HDF5_Groupnameformat'] % f], **kwargs)
                            else:
                                obj.__init__(fileformat % f, **kwargs)
                        except IOError:
                            del obj
                            if kwargs['error_on_not_found']:
                                if hdf is not None and not isinstance(fileformat, h5py.highlevel.Group):
                                    #we opened, we have to close it.
                                    hdf.close()
                                    hdf = None
                                raise
                            else:
                                obj = None
                        objlist.append(obj)
                    if hdf is not None and not isinstance(fileformat, h5py.highlevel.Group):
                        #we opened, we have to close it.
                        hdf.close()
                        hdf = None
                    return objlist
                else:
                    #single-FSN
                    obj = super(SASExposure, cls).__new__(cls)
                    return obj #delegate the job to the __init__ method
        else:
            raise ValueError('Invalid number of positional arguments.')
    def check_for_mask(self, isfatal = True):
        if self.mask is None:
            if isfatal:
                raise SASExposureException('mask not defined') #IGNORE:W0710
            else:
                return False
        return True
    def check_for_q(self, isfatal = True):
        "Check if needed header elements are present for calculating q values for pixels. If not, raise a SASAverageException."
        #Note, that we check for 'Dist' and 'Energy' instead of 'DistCalibrated'
        # and 'EnergyCalibrated', because if the latter are missing, they
        # default to the values of the uncalibrated ones.
        missing = [x for x in  ['BeamPosX', 'BeamPosY', 'Dist', 'Energy', 'XPixel', 'YPixel'] if x not in self.header]

        if missing:
            if isfatal:
                raise SASExposureException('Fields missing from header: ' + str(missing)) #IGNORE:W0710
            else:
                return missing
    def __del__(self):
        for x in ['Intensity', 'Error', 'header', 'mask']:
            if hasattr(self, x):
                delattr(self, x)
    @property
    def shape(self):
        return self.Intensity.shape

    def __setitem__(self, key, value):
        self.header[key] = value

    def __getitem__(self, key):
        return self.header[key]

    def __delitem__(self, key):
        del self.header[key]
### -------------- Loading routines (new_from_xyz) ------------------------

    @classmethod
    def new_from_hdf5(cls, hdf_or_filename):
        #get a HDF file object
        ret = []
        with _HDF_parse_group(hdf_or_filename) as hpg:
            if hpg.name.startswith('/FSN'):
                hdfgroups = [hpg]
            else:
                hdfgroups = [x for x in hpg.keys() if x.startswith('FSN')]
            for g in hdfgroups:
                ret.append(cls())
                ret[-1].read_from_hdf5(hpg[g], load_mask = False)
            # adding masks later, thus only one copy of each mask will exist in
            # the memory 
            masknames = set([r.header['maskid'] for r in ret])
            masks = dict([(mn, SASMask(hpg, maskid = mn)) for mn in masknames])
            for r in ret:
                r.set_mask(masks[r.header['maskid']])
        return ret

### -------------- reading routines--------------------------

    def read_from_image(self, filename, **kwargs):
        """Read a "bare" image file
        """
        kwargs = SASExposure._set_default_kwargs_for_readers(kwargs)
        filename = misc.findfileindirs(filename, kwargs['dirs'])
        if filename.upper().endswith('CBF'):
            self.Intensity = twodim.readcbf(filename)
        elif filename.upper().endswith('.TIF') or filename.upper().endswith('.TIFF'):
            self.Intensity = twodim.readtif(filename)
        if kwargs['estimate_errors']:
            self.Error = np.sqrt(self.Intensity)
        self.header = SASHeader({'__Origin__':'bare_image'})
        return self

    def read_from_ESRF_ID02(self, filename, **kwargs):
        """Read an EDF file (ESRF beamline ID02 SAXS pattern)
        
        Inputs:
            filename: the name of the file to be loaded
            estimate_errors: error matrices are usually not saved, but they can
                be estimated from the intensity, if they are not present (True
                by default).
            read_mask: try to load the corresponding mask (True by default).
            dirs: folders to look file for.
        """
        kwargs = SASExposure._set_default_kwargs_for_readers(kwargs)
        filename = misc.findfileindirs(filename, kwargs['dirs'])
        edf = twodim.readedf(filename)
        self.header = SASHeader(edf)
        self.Intensity = edf['data'].astype(np.double)
        if kwargs['load_mask']:
            mask = SASMask(misc.findfileindirs(self.header['MaskFileName'], kwargs['dirs']))
            if self.Intensity.shape != mask.mask.shape:
                if all(self.Intensity.shape[i] > mask.mask.shape[i] for i in [0, 1]):
                    xbin, ybin = [self.Intensity.shape[i] / mask.mask.shape[i] for i in [0, 1]]
                    extend = True
                elif all(self.Intensity.shape[i] < mask.mask.shape[i] for i in [0, 1]):
                    xbin, ybin = [mask.mask.shape[i] / self.Intensity.shape[i] for i in [0, 1]]
                    extend = False
                else:
                    raise ValueError('Cannot do simultaneous forward and backward mask binning.')
                warnings.warn('Rebinning mask: %s x %s, direction: %s' % (xbin, ybin, ['shrink', 'enlarge'][extend]))
                mask = mask.rebin(xbin, ybin, extend)
            self.set_mask(mask)
            dummypixels = np.absolute(self.Intensity - self.header['Dummy']) <= self.header['DDummy']
            #self.Intensity[dummypixels]=0
            self.mask.mask &= (-dummypixels).reshape(self.Intensity.shape)
        if kwargs['estimate_errors']:
            sd = edf['SampleDistance']
            ps2 = edf['PSize_1'] * edf['PSize_2']
            I1 = edf['Intensity1']
            self.Error = (0.5 * sd * sd / ps2 / I1 + self.Intensity) * float(sd * sd) / (ps2 * I1)

    def read_from_B1_org(self, filename, **kwargs):
        """Read an original exposition (beamline B1, HASYLAB/DESY, Hamburg)
        
        Inputs:
            filename: the name of the file to be loaded
            dirs: folders to look for files in
        
        Notes:
            We first try to load the header file. If the file name has no
                extension, extensions .header, .DAT,
                .dat, .DAT.gz, .dat.gz are tried in this order.
            If the header has been successfully loaded, we try to load the data.
                Extensions: .cbf, .tif, .tiff, .DAT, .DAT.gz, .dat, .dat.gz are
                tried in this sequence.
            If either the header or the data cannot be loaded, an IOError is
                raised.
            
        """
        kwargs = SASExposure._set_default_kwargs_for_readers(kwargs)
        #try to load header file
        header_extns_default = ['.header', '.DAT', '.dat', '.DAT.gz', '.dat.gz']
        data_extns_default = ['.cbf', '.tif', '.tiff', '.DAT', '.DAT.gz', '.dat', '.dat.gz']

        header_extn = [x for x in header_extns_default if filename.upper().endswith(x.upper())]
        data_extn = [x for x in data_extns_default if filename.upper().endswith(x.upper())]


        # if an extension is found, remove it to get the basename.
        if header_extn + data_extn: # is not empty
            basename = os.path.splitext(filename)[0]
        else:
            basename = filename

        #prepend the already found extension (if any) to the list of possible
        # file extensions, both for header and data.
        header_extn.extend(header_extns_default)
        data_extn.extend(data_extns_default)

        headername = ''

        for extn in header_extn:
            try:
                headername = misc.findfileindirs(basename + extn, kwargs['dirs'])
            except IOError:
                continue
        if not headername:
            raise IOError('Could not find header file')
        dataname = ''
        for extn in data_extn:
            try:
                dataname = misc.findfileindirs(basename + extn, kwargs['dirs'])
            except IOError:
                continue
        if not dataname:
            raise IOError('Could not find 2d org file') #skip this file
        self.header = SASHeader(headername)
        if dataname.lower().endswith('.cbf'):
            self.Intensity = twodim.readcbf(dataname)
        elif dataname.upper().endswith('.DAT') or dataname.upper().endswith('.DAT.GZ'):
            self.Intensity = twodim.readjusifaorg(dataname).reshape(256, 256)
        elif dataname.upper().endswith('.TIF') or dataname.upper().endswith('.TIFF'):
            self.Intensity = twodim.readtif(dataname)
        else:
            raise NotImplementedError(dataname)
        if kwargs['estimate_errors']:
            self.Error = np.sqrt(self.Intensity)
        else:
            self.Error = None
        return self

    def read_from_B1_int2dnorm(self, filename, **kwargs):
        kwargs = SASExposure._set_default_kwargs_for_readers(kwargs)
        if 'fileformat' not in kwargs:
            kwargs['fileformat'] = 'int2dnorm%d'
        if 'logfileformat' not in kwargs:
            kwargs['logfileformat'] = 'intnorm%d'
        if 'logfileextn' not in kwargs:
            kwargs['logfileextn'] = '.log'

        data_extns = ['.npy', '.mat']
        data_extn = [x for x in data_extns if filename.upper().endswith(x.upper())]

        if os.path.isabs(filename):
            if kwargs['dirs'] is None:
                kwargs['dirs'] = []
            elif isinstance(kwargs['dirs'], basestring):
                kwargs['dirs'] = [kwargs['dirs']]
            kwargs['dirs'] = [os.path.split(filename)[0]] + kwargs['dirs']

        if data_extn: # is not empty
            basename = os.path.splitext(filename)[0]
        else:
            basename = filename
        data_extn.extend(data_extns)

        dataname = None
        for extn in data_extn:
            try:
                dataname = misc.findfileindirs(basename + extn, kwargs['dirs'])
            except IOError:
                continue
        if not dataname:
            raise IOError('Cannot find two-dimensional file!')
        m = re.match(kwargs['fileformat'].replace('%d', r'(\d+)'), os.path.split(basename)[1])
        if m is None:
            raise ValueError('Filename %s does not have the format %s, \
therefore the FSN cannot be determined.' % (dataname, kwargs['fileformat']))
        else:
            fsn = int(m.group(1))

        headername = misc.findfileindirs(kwargs['logfileformat'] % fsn + kwargs['logfileextn'], kwargs['dirs'])
        self.header = SASHeader(headername)
        self.Intensity, self.Error = twodim.readint2dnorm(dataname)
        self.header.add_history('Intensity and Error matrices loaded from ' + dataname)
        return self

    def read_from_HDF5(self, filename_or_group, **kwargs):
        kwargs = SASExposure._set_default_kwargs_for_readers(kwargs)
        with _HDF_parse_group(filename_or_group) as hdf_group:
            if kwargs['HDF5_Intensityname'] in hdf_group.keys():
                self.Intensity = hdf_group[kwargs['HDF5_Intensityname']].value
            else:
                raise IOError('No Intensity dataset in HDF5 group ' + hdf_group.filename + ':' + hdf_group.name)
            if kwargs['HDF5_Errorname'] in hdf_group.keys():
                self.Error = hdf_group[kwargs['HDF5_Errorname']].value
            self.header = SASHeader()
            self.header.read_from_hdf5(hdf_group)
            if self.header['maskid'] is not None and kwargs['load_mask']:
                self.mask = SASMask(hdf_group.parent, maskid = self.header['maskid'])

    def read_from_PAXE(self, filename, **kwargs):
        kwargs = SASExposure._set_default_kwargs_for_readers(kwargs)
        paxe = twodim.readPAXE(misc.findfileindirs(filename, dirs = kwargs['dirs']))
        self.header = SASHeader(paxe[0])
        self.Intensity = paxe[1]
        if kwargs['estimate_errors']:
            self.Error = np.sqrt(self.Intensity)
        else:
            self.Error = None
        return self

    def read_from_BDF(self, filename, **kwargs):
        if 'bdfext' not in kwargs:
            kwargs['bdfext'] = '.bdf'
        if 'bhfext' not in kwargs:
            kwargs['bhfext'] = '.bhf'
        if 'read_corrected_if_present' not in kwargs:
            kwargs['read_corrected_if_present'] = True
        kwargs = SASExposure._set_default_kwargs_for_readers(kwargs)
        data = misc.flatten_hierarchical_dict(twodim.readbdf(misc.findfileindirs(filename, kwargs['dirs'])))
        datanames = [k for k in data if isinstance(data[k], np.ndarray)]
        if data['C.bdfVersion'] < 2:
            Intname = 'DATA'
            Errname = 'ERROR'
        elif kwargs['read_corrected_if_present'] and 'CORRDATA' in datanames:
            Intname = 'CORRDATA'
            Errname = 'CORRERROR'
        else:
            Intname = 'RAWDATA'
            Errname = 'RAWERROR'
        self.Intensity = data[Intname]
        self.Error = data[Errname]
        for dn in datanames:
            del data[dn]
        self.header = SASHeader(data)
        #TODO: load mask if needed
        return self
### ------------------- Interface routines ------------------------------------
    def set_mask(self, mask):
        self.mask = SASMask(mask)
        self.header['maskid'] = self.mask.maskid
        self.header.add_history('Mask %s associated to exposure.' % self.mask.maskid)

    def get_matrix(self, name = 'Intensity', othernames = None):
        name = self.get_matrix_name(name, othernames)
        return getattr(self, name)

    def get_matrix_name(self, name = 'Intensity', othernames = None):
        if name in self.matrices.values():
            name = [k for k in self.matrices if self.matrices[k] == name][0]
        if hasattr(self, name) and (getattr(self, name) is not None):
            return name
        if othernames is None:
            othernames = self.matrix_names

        for k in othernames:
            try:
                if getattr(self, k) is not None:
                    return k
            except AttributeError:
                pass
        raise AttributeError('No matrix in this instance of' + str(type(self)))

### ------------------- simple arithmetics ------------------------------------

    def _check_arithmetic_compatibility(self, other):
        if isinstance(other, numbers.Number):
            return ErrorValue(other)
        elif isinstance(other, np.ndarray):
            if other.shape == self.shape:
                return ErrorValue(other)
            else:
                raise ValueError('Incompatible shape!')
        elif isinstance(other, SASExposure):
            if other.shape == self.shape:
                return ErrorValue(other.Intensity, other.Error)
            else:
                raise ValueError('Incompatible shape!')
        else:
            raise NotImplementedError
    def __iadd__(self, other):
        try:
            other = self._check_arithmetic_compatibility(other)
        except NotImplementedError:
            return NotImplemented
        if self.Error is None:
            self.Error = np.zeros_like(self.Intensity)
        self.Intensity = self.Intensity + other.val
        self.Error = np.sqrt(self.Error ** 2 + other.err ** 2)
        return self
    def __neg__(self):
        pass  #TODO
    def _recip(self):
        pass  #TODO
    def __imul__(self):
        pass  #TODO    Also COPY constructors in DATASET-related classes, for ArithmeticBase...
### ------------------- Routines for radial integration -----------------------

    def get_qrange(self, N = None, spacing = 'linear'):
        """Calculate the available q-range.
        
        Inputs:
            N: if integer: the number of bins
               if float: the distance between bins (equidistant bins)
               if None: automatic determination of the number of bins
            spacing: only effective if N is an integer. 'linear' means linearly
                equidistant spacing (as in np.linspace), 'logarithmic' means
                logarithmic spacing (as in np.logspace).
        
        Returns:
            the q-scale in a numpy array.
        """
        self.check_for_mask()
        qrange = utils2d.integrate.autoqscale(self.header['EnergyCalibrated'],
                                            self.header['DistCalibrated'],
                                            self.header['XPixel'],
                                            self.header['YPixel'],
                                            self.header['BeamPosX'],
                                            self.header['BeamPosY'], 1 - self.mask.mask)
        if N is None:
            return qrange
        elif isinstance(N, numbers.Integral):
            if spacing.upper().startswith('LIN'):
                return np.linspace(qrange.min(), qrange.max(), N)
            elif spacing.upper().startswith('LOG'):
                return np.logspace(np.log10(qrange.min()), np.log10(qrange.max()), N)
        elif isinstance(N, numbers.Real):
            return np.arange(qrange.min(), qrange.max(), N)
        else:
            raise NotImplementedError

    @classmethod
    def common_qrange(cls, *exps):
        """Find a common q-range for several exposures
        
        Usage:
        
        >>> SASExposure.common_qrange(exp1, exp2, exp3, ...)
        
        where exp1, exp2, exp3... are instances of the SASExposure class.
        
        Returns:
            the estimated common q-range in a np.ndarray (ndim=1).
        """
        if not all([isinstance(e, cls) for e in exps]):
            raise ValueError('All arguments should be SASExposure instances.')
        qranges = [e.get_qrange() for e in exps]
        qmin = max([qr.min() for qr in qranges])
        qmax = min([qr.max() for qr in qranges])
        N = min([len(qr) for qr in qranges])
        return np.linspace(qmin, qmax, N)

    def radial_average(self, qrange = None, pixel = False, matrix = 'Intensity', errormatrix = 'Error'):
        """Do a radial averaging
        
        Inputs:
            qrange: the q-range. If None, auto-determine.
            pixel: do a pixel-integration (instead of q)
            matrix: matrix to use for averaging
            errormatrix: error matrix to use for averaging (or None)
            
        Outputs:
            the one-dimensional curve as an instance of SASCurve (if pixel is
                False) or DataSet (if pixel is True)
        """
        self.check_for_mask()
        mat = getattr(self, matrix)
        if errormatrix is not None:
            err = getattr(self, errormatrix)
        else:
            err = None
        if not pixel:
            res = utils2d.integrate.radint(mat, err,
                                               self.header['EnergyCalibrated'],
                                               self.header['DistCalibrated'],
                                               (self.header['XPixel'], self.header['YPixel']),
                                               self.header['BeamPosX'],
                                               self.header['BeamPosY'],
                                               1 - self.mask.mask, qrange,
                                               returnavgq = True,
                                               returnpixel = True)
            if err is not None:
                q, I, E, A, p = res
            else:
                q, I, A, p = res
                E = np.zeros_like(q)
            ds = dataset.SASCurve(q, I, E)
            ds.addfield('Pixels', p)
        else:
            res = utils2d.integrate.radintpix(mat, err,
                                                     self.header['BeamPosX'],
                                                     self.header['BeamPosY'],
                                                     1 - self.mask.mask, qrange,
                                                     returnavgpix = True)
            if err is not None:
                p, I, E, A = res
            else:
                p, I, A = res
                E = np.zeros_like(p)
            ds = dataset.DataSet(p, I, E)
        ds.addfield('Area', A)
        ds.header = SASHeader(self.header)
        return ds

    def sector_average(self, phi0, dphi, qrange = None, pixel = False, matrix = 'Intensity', errormatrix = 'Error', symmetric_sector = False):
        """Do a radial averaging restricted to one sector.
        
        Inputs:
            phi0: start of the sector (radians).
            dphi: sector width (radians)
            qrange: the q-range. If None, auto-determine.
            pixel: do a pixel-integration (instead of q)
            matrix: matrix to use for averaging
            errormatrix: error matrix to use for averaging (or None)
            symmetric_sectors: if the sector should be symmetric (phi0+pi needs
                also be taken into account)
        Outputs:
            the one-dimensional curve as an instance of SASCurve (if pixel is
                False) or DataSet (if pixel is True)
    
        Notes:
            x is row direction, y is column. 0 degree is +x, 90 degree is +y.
        """
        self.check_for_mask()
        mat = getattr(self, matrix)
        if errormatrix is not None:
            err = getattr(self, errormatrix)
        else:
            err = None
        if not pixel:
            res = utils2d.integrate.radint(mat, err,
                                               self.header['EnergyCalibrated'],
                                               self.header['DistCalibrated'],
                                               (self.header['XPixel'], self.header['YPixel']),
                                               self.header['BeamPosX'],
                                               self.header['BeamPosY'],
                                               1 - self.mask.mask, qrange,
                                               returnavgq = True,
                                               returnpixel = True,
                                               phi0 = phi0, dphi = dphi, symmetric_sector = symmetric_sector)
            if err is not None:
                q, I, E, A, p = res
            else:
                q, I, A, p = res
                E = np.zeros_like(q)
            ds = dataset.SASCurve(q, I, E)
            ds.addfield('Pixels', p)
        else:
            res = utils2d.integrate.radintpix(mat, err,
                                                     self.header['BeamPosX'],
                                                     self.header['BeamPosY'],
                                                     1 - self.mask.mask, qrange,
                                                     returnavgpix = True, phi0 = phi0,
                                                     dphi = dphi, symmetric_sector = symmetric_sector)
            if err is not None:
                p, I, E, A = res
            else:
                p, I, A = res
                E = np.zeros_like(p)
            ds = dataset.DataSet(p, I, E)
        ds.addfield('Area', A)
        ds.header = SASHeader(self.header)
        return ds

    def azimuthal_average(self, qmin, qmax, Ntheta = 100, pixel = False, matrix = 'Intensity', errormatrix = 'Error'):
        """Do an azimuthal averaging restricted to a ring.
        
        Inputs:
            qmin, qmax: lower and upper bounds of the ring (q or pixel)
            Ntheta: number of points in the output.
            pixel: do a pixel-integration (instead of q)
            matrix: matrix to use for averaging
            errormatrix: error matrix to use for averaging (or None)
        
        Outputs:
            the one-dimensional curve as an instance of DataSet
    
        Notes:
            x is row direction, y is column. 0 degree is +x, 90 degree is +y.
        """
        self.check_for_mask()
        mat = getattr(self, matrix)
        if errormatrix is not None:
            err = getattr(self, errormatrix)
        else:
            err = None
        if not pixel:
            res = utils2d.integrate.azimint(mat, err,
                                               self.header['EnergyCalibrated'],
                                               self.header['DistCalibrated'],
                                               (self.header['XPixel'], self.header['YPixel']),
                                               self.header['BeamPosX'],
                                               self.header['BeamPosY'],
                                               1 - self.mask.mask, Ntheta,
                                               qmin = qmin, qmax = qmax)
            if err is not None:
                theta, I, E, A = res
            else:
                theta, I, A = res
                E = np.zeros_like(theta)
            ds = dataset.DataSet(theta, I, E)
        else:
            res = utils2d.integrate.azimintpix(mat, err,
                                                     self.header['BeamPosX'],
                                                     self.header['BeamPosY'],
                                                     1 - self.mask.mask, Ntheta,
                                                     pixmin = qmin, pixmax = qmax)
            if err is not None:
                theta, I, E, A = res
            else:
                theta, I, A = res
                E = np.zeros_like(theta)
            ds = dataset.DataSet(theta, I, E)
        ds.addfield('Area', A)
        ds.header = SASHeader(self.header)
        return ds


### ---------------------- Plotting -------------------------------------------

    def plot2d(self, **kwargs):
        """Plot the matrix (imshow)
        
        Allowed keyword arguments [and their default values]:
        
        zscale ['linear']: colour scaling of the image. Either a string ('log',
            'log10' or 'linear', case insensitive), or an unary function which
            operates on a numpy array and returns a numpy array of the same
            shape, e.g. np.log, np.sqrt etc.
        crosshair [True]: if a cross-hair marking the beam position is to be
            plotted.
        drawmask [True]: if the mask is to be plotted. If no mask is attached
            to this SASExposure object, it defaults to False.
        qrange_on_axis [True]: if the q-range is to be set to the axis. If no
            mask is attached to this SASExposure instance, defaults to False
        matrix ['Intensity']: the matrix which is to be plotted. If this is not
            present, another one will be chosen quietly
        axis [None]: the axis into which the image should be plotted. If None,
            defaults to the currently active axis (returned by plt.gca())
        invalid_color ['black']: the color for invalid (NaN or infinite) pixels
        mask_opacity [0.8]: the opacity of the overlaid mask (1 is fully opaque,
            0 is fully transparent)
        minvalue: minimal value. All matrix elements below this will be replaced
            by this. Defaults to -infinity.
        maxvalue: maximal value. All matrix elements above this will be replaced
            by this. Defaults to +infinity.
        return_matrix: if the transformed, just-plotted matrix is to be
            returned. False by default.
        All other keywords are forwarded to plt.imshow()
        
        Returns: the image instance returned by imshow()
        """
        kwargs_default = {'zscale':'linear',
                        'crosshair':True,
                        'drawmask':True,
                        'qrange_on_axis':True,
                        'matrix':'Intensity',
                        'axis':None,
                        'invalid_color':'black',
                        'mask_opacity':0.6,
                        'interpolation':'nearest',
                        'origin':'upper',
                        'minvalue':-np.inf,
                        'maxvalue':np.inf,
                        'return_matrix':False}
        my_kwargs = ['zscale', 'crosshair', 'drawmask', 'qrange_on_axis', 'matrix',
                   'axis', 'invalid_color', 'mask_opacity', 'minvalue', 'maxvalue',
                   'return_matrix']
        kwargs_default.update(kwargs)
        return_matrix = kwargs_default['return_matrix'] # save this as this will be removed when kwars_default is fed into imshow()

        kwargs_for_imshow = dict([(k, kwargs_default[k]) for k in kwargs_default if k not in my_kwargs])
        if isinstance(kwargs_default['zscale'], basestring):
            if kwargs_default['zscale'].upper().startswith('LOG10'):
                kwargs_default['zscale'] = np.log10
            elif kwargs_default['zscale'].upper().startswith('LN'):
                kwargs_default['zscale'] = np.log
            elif kwargs_default['zscale'].upper().startswith('LIN'):
                kwargs_default['zscale'] = lambda a:a.copy()
            elif kwargs_default['zscale'].upper().startswith('LOG'):
                kwargs_default['zscale'] = np.log
            else:
                raise ValueError('Invalid value for zscale: %s' % kwargs_default['zscale'])
        mat = self.get_matrix(kwargs_default['matrix']).copy()
        mat[mat < kwargs_default['minvalue']] = kwargs_default['minvalue']
        mat[mat > kwargs_default['maxvalue']] = kwargs_default['maxvalue']
        mat = kwargs_default['zscale'](mat)

        if kwargs_default['drawmask']:
            if not self.check_for_mask(False):
                warnings.warn('Drawing of mask was requested, but mask is not present!')
                kwargs_default['drawmask'] = False
        if kwargs_default['qrange_on_axis']:
            missing = self.check_for_q(False)
            if missing:
                warnings.warn('Q scaling on the axes was requested, but the following fields are missing: ' + repr(missing))
                kwargs_default['qrange_on_axis'] = False

        if kwargs_default['qrange_on_axis']:
            self.check_for_q()
            xmin = 4 * np.pi * np.sin(0.5 * np.arctan((0 - self.header['BeamPosY']) * self.header['YPixel'] / self.header['DistCalibrated'])) * self.header['EnergyCalibrated'] / HC
            xmax = 4 * np.pi * np.sin(0.5 * np.arctan((mat.shape[1] - self.header['BeamPosY']) * self.header['YPixel'] / self.header['DistCalibrated'])) * self.header['EnergyCalibrated'] / HC
            ymin = 4 * np.pi * np.sin(0.5 * np.arctan((0 - self.header['BeamPosX']) * self.header['XPixel'] / self.header['DistCalibrated'])) * self.header['EnergyCalibrated'] / HC
            ymax = 4 * np.pi * np.sin(0.5 * np.arctan((mat.shape[0] - self.header['BeamPosX']) * self.header['XPixel'] / self.header['DistCalibrated'])) * self.header['EnergyCalibrated'] / HC
            if kwargs_for_imshow['origin'].upper() == 'UPPER':
                kwargs_for_imshow['extent'] = [xmin, xmax, ymax, ymin]
            else:
                kwargs_for_imshow['extent'] = [xmin, xmax, ymin, ymax]
            bcx = 0
            bcy = 0
        else:
            if 'BeamPosX' in self.header and 'BeamPosY' in self.header:
                bcx = self.header['BeamPosX']
                bcy = self.header['BeamPosY']
            else:
                bcx = None; bcy = None
            xmin = 0; xmax = mat.shape[1]; ymin = 0; ymax = mat.shape[0]

        if kwargs_default['axis'] is None:
            kwargs_default['axis'] = plt.gca()
        ret = kwargs_default['axis'].imshow(mat, **kwargs_for_imshow) #IGNORE:W0142
        if kwargs_default['drawmask']:
            #workaround: because of the colour-scaling we do here, full one and
            #   full zero masks look the SAME, i.e. all the image is shaded.
            #   Thus if we have a fully unmasked matrix, skip this section.
            #   This also conserves memory.
            if self.mask.mask.sum() != self.mask.mask.size:
                #Mask matrix should be plotted with plt.imshow(maskmatrix, cmap=_colormap_for_mask)
                _colormap_for_mask = matplotlib.colors.ListedColormap(['white', 'white'], '_sastool_%s' % misc.random_str(10))
                _colormap_for_mask._init() #IGNORE:W0212
                _colormap_for_mask._lut[:, -1] = 0 #IGNORE:W0212
                _colormap_for_mask._lut[0, -1] = kwargs_default['mask_opacity'] #IGNORE:W0212
                kwargs_for_imshow['cmap'] = _colormap_for_mask
                kwargs_default['axis'].imshow(self.mask.mask, **kwargs_for_imshow) #IGNORE:W0142
        if kwargs_default['crosshair']:
            if bcx is not None and bcy is not None:
                ax = kwargs_default['axis'].axis()
                kwargs_default['axis'].plot([xmin, xmax], [bcx] * 2, 'w-')
                kwargs_default['axis'].plot([bcy] * 2, [ymin, ymax], 'w-')
                kwargs_default['axis'].axis(ax)
            else:
                warnings.warn('Cross-hair was requested but beam center was not found.')
        #kwargs_default['axis'].set_axis_bgcolor(kwargs_default['invalid_color'])
        kwargs_default['axis'].figure.canvas.draw()
        if return_matrix:
            return ret, mat
        else:
            return ret

###  ------------------------ Beam center finding -----------------------------

    def update_beampos(self, bc, source = None):
        """Update the beam position in the header.
        
        Inputs:
            bc: beam position coordinates (row, col; starting from 0).
            source: name of the beam finding algorithm.
        """
        self.header['BeamPosX'], self.header['BeamPosY'] = bc
        if not source:
            self.header.add_history('Beam position updated to:' + str(tuple(bc)))
        else:
            self.header.add_history('Beam found by *%s*: %s' % (source, str(tuple(bc))))
    def find_beam_semitransparent(self, bs_area, update = True):
        """Find the beam position from the area under the semitransparent
        beamstop.
        
        Inputs:
            bs_area: sequence of the coordinates of the beam-stop area rect.:
                [row_min, row_max, column_min, column_max]
            update: if the new value should be written in the header (default).
                If False, the newly found beam position is only returned.
        
        Outputs:
            the beam position (row,col).
        """
        bs_area = [min(bs_area[2:]), max(bs_area[2:]), min(bs_area[:2]), max(bs_area[:2])]
        bc = utils2d.centering.findbeam_semitransparent(self.get_matrix(), bs_area)
        if update:
            self.update_beampos(bc, source = 'semitransparent')
        return bc
    def find_beam_slices(self, pixmin = 0, pixmax = np.inf, sector_width = np.pi / 9.,
                         update = True, callback = None):
        """Find the beam position by matching diagonal sectors.
        
        Inputs:
            pixmin, pixmax: lower and upper thresholds in the distance from the
                origin in the radial averaging [in pixel units]
            sector_width: width of sectors in radian.
            update: if the new value should be written in the header (default).
                If False, the newly found beam position is only returned.
            callback: a function (accepting no arguments) to be called in each
                iteration of the fitting procedure.
        
        Outputs:
            the beam position (row,col).
        """
        self.check_for_mask()
        bc = utils2d.centering.findbeam_slices(self.get_matrix(),
                                             (self.header['BeamPosX'],
                                              self.header['BeamPosY']),
                                             self.mask.mask, dmin = pixmin,
                                             dmax = pixmax,
                                             sector_width = sector_width,
                                             callback = callback)
        if update:
            self.update_beampos(bc, source = 'slices')
        return bc
    def find_beam_gravity(self, update = True):
        """Find the beam position by finding the center of gravity in each row
        and column.
        
        Inputs:
            update: if the new value should be written in the header (default).
                If False, the newly found beam position is only returned.
        
        Outputs:
            the beam position (row,col).
        """
        self.check_for_mask()
        bc = utils2d.centering.findbeam_gravity(self.get_matrix(), self.mask.mask)
        if update:
            self.update_beampos(bc, source = 'gravity')
        return bc

    def find_beam_azimuthal_fold(self, Ntheta = 50, dmin = 0, dmax = np.inf,
                                 update = True, callback = None):
        """Find the beam position by matching an azimuthal scattering curve
        and its counterpart shifted by pi radians.
        
        Inputs:
            Ntheta: number of bins in the azimuthal scattering curve
            dmin, dmax: lower and upper thresholds in the distance from the
                origin in the radial averaging [in pixel units]
            update: if the new value should be written in the header (default).
                If False, the newly found beam position is only returned.
            callback: a function (accepting no arguments) to be called in each
                iteration of the fitting procedure.
        
        Outputs:
            the beam position (row,col).
        """
        self.check_for_mask()
        bc = utils2d.centering.findbeam_azimuthal_fold(self.get_matrix(),
                                                     (self.header['BeamPosX'],
                                                      self.header['BeamPosY']),
                                                     self.mask.mask,
                                                     Ntheta = Ntheta, dmin = dmin,
                                                     dmax = dmax, callback = callback)
        if update:
            self.update_beampos(bc, source = 'azimuthal_fold')
        return bc

    def find_beam_radialpeak(self, pixmin, pixmax, drive_by = 'amplitude', extent = 10, update = True, callback = None):
        """Find the beam position by optimizing a peak in the radial scattering
        curve.
        
        Inputs:
            pixmin, pixmax: lower and upper thresholds in the distance from the
                origin in the radial averaging [in pixel units]. Should be a
                narrow interval, zoomed onto one peak.
            drive_by: 'amplitude' if the amplitude of the peak has to be maximized
                or 'hwhm' if the hwhm should be minimized.
            extent: expected distance of the true beam position from the current
                one. Just the magnitude counts.
            update: if the new value should be written in the header (default).
                If False, the newly found beam position is only returned.
            callback: a function (accepting no arguments) to be called in each
                iteration of the fitting procedure.
        
        Outputs:
            the beam position (row,col).
        """
        self.check_for_mask()
        bc = utils2d.centering.findbeam_radialpeak(self.get_matrix(),
                                                 (self.header['BeamPosX'],
                                                  self.header['BeamPosY']),
                                                 self.mask.mask, pixmin,
                                                 pixmax, drive_by = drive_by,
                                                 extent = extent, callback = callback)
        if update:
            self.update_beampos(bc, source = 'radialpeak')
        return bc

### ----------------------- Writing routines ----------------------------------

    def write_to_HDF5(self, hdf_or_filename, **kwargs):
        """Save exposure to a HDF5 file or group.
        
        Inputs:
            hdf_or_filename: a file name (string) or an instance of 
                h5py.highlevel.File (equivalent to a HDF5 root group) or
                h5py.highlevel.Group.
            other keyword arguments are passed on as keyword arguments to the
                h5py.highlevel.Group.create_dataset() method.
                
        A HDF5 group will be created with the name FSN<fsn> and the available
        matrices (Intensity, Error) will be saved. Header data is saved
        as attributes to the HDF5 group.
        
        If a mask is associated to this exposure, it is saved as well as a
        sibling group of FSN<fsn> with the name <maskid>.
        """
        if 'compression' not in kwargs:
            kwargs['compression'] = 'gzip'
        with _HDF_parse_group(hdf_or_filename) as hpg:
            groupname = 'FSN%d' % self.header['FSN']
            if groupname in hpg.keys():
                del hpg[groupname]
            hpg.create_group(groupname)
            for k in self.matrix_names:
                if k is None:
                    continue
                hpg[groupname].create_dataset(k, data = getattr(self, k), **kwargs)
            self.header.write_to_hdf5(hpg[groupname])
            if self.mask is not None:
                self.mask.write_to_hdf5(hpg)

    ### ------------------------ Calculation routines -------------------------
    @property
    def Dpix(self):
        """The distance of each pixel from the beam center in pixel units"""
        self.check_for_q()
        col, row = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
        return np.sqrt((col - self.header['BeamPosX']) ** 2 + (row - self.header['BeamPosY']) ** 2)
    @property
    def D(self):
        """The distance of each pixel from the beam center in length units (mm)."""
        self.check_for_q()
        col, row = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
        return np.sqrt(((col - self.header['BeamPosX']) * self.header['XPixel']) ** 2 +
                       ((row - self.header['BeamPosY']) * self.header['YPixel']) ** 2)
    @property
    def q(self):
        """The magnitude of the momentum transfer (q=4*pi*sin(theta)/lambda)
        for each pixel in 1/Angstroem units."""
        self.check_for_q()
        return 4 * np.pi * np.sin(0.5 * \
                                  np.arctan(self.D / \
                                            self.header['DistCalibrated'])) / \
                           HC * self.header['EnergyCalibrated']
    ### ------------------------ Simple arithmetics ---------------------------
