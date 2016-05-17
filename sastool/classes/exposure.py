'''
Created on Jun 15, 2012

@author: andris
'''

import collections
import numbers
import warnings

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid import make_axes_locatable

from .common import SASExposureException
from .curve import SASCurve, SASAzimuthalCurve, SASPixelCurve
from .header import SASHeader
from .mask import SASMask
from .. import misc
from .. import utils2d
from ..misc.arithmetic import ArithmeticBase
from ..misc.errorvalue import ErrorValue

__all__ = ['SASExposure']


class SASExposure(ArithmeticBase):

    """A class for holding SAS exposure data, i.e. intensity, error, metadata
    and mask.

    A SASExposure has the following special attributes:

    Intensity: scattered intensity matrix (np.ndarray).
    Error: error of scattered intensity (np.ndarray).
    header: metadata dictionary (SASHeader instance).
    mask: mask matrix in the form of a SASMask instance.

    any of the above attributes can be missing, a value of None signifies
    this situation.

    This class can be instantiated in one of the following ways:

    ``SASExposure()``
        create an empty object (Intensity, Error, mask, header are None)

    ``SASExposure(other_instance_of_sasexposure)``
        copy constructor: make a deep copy

    ``SASExposure(filename, [other keyword arguments])``
        load a file. The currently supported file (experiment) types and the
        corresponding accepted, not common keyword arguments can be found in the
        doc-strings of the ``read_from_*`` member functions

    ``SASExposure(filename_format, fsn, [other keyword arguments])``
        load multiple files. `filename_format` is a C-style format string, into
        which the file sequence number(s) can be interpolated, e.g. 's%07d.bdf'.
        `fsn` can be a scalar number or a sequence of numbers. In the former
        case a single ``SASExposure`` is returned. In the latter, a list of
        ``SASExposure`` instances is returned.

    ``SASExposure(hdf5_object [, fsn] [, HDF5_Groupnameformat='FSN(\d+)']  \
        [, HDF5_Intensityname='Intensity'] [, HDF5_Errorname='Error'])``
        load exposures from a HDF5 file or group described by `hdf5_object`,
        which should either be the name of a HDF5 file or an instance of
        `h5py.Highlevel.Group`. Exposures reside in HDF5 groups whose name
        conforms the regular expression pattern in HDF5_Groupnameformat. Each
        of these groups can have an intensity and an error matrix (HDF5 datasets
        designated by `HDF5_Intensityname` and `HDF5_Errorname`, respectively.
        The meta-data are stored as attributes to the HDF5 group of the exposure.
        If `fsn` is omitted, all exposures found in the group are loaded. If
        present, only the given file sequence numbers are read (if they are
        present).
    """
    matrix_names = ['Intensity', 'Error']
    matrices = dict([('Intensity', 'Scattered intensity'),
                     ('Error', 'Error of intensity')])
    _plugins = []

    @classmethod
    def register_IOplugin(cls, plugin, idx):
        if idx is None:
            cls._plugins.append(plugin)
        else:
            cls._plugins.insert(idx, plugin)

    @classmethod
    def get_IOplugin(cls, filename, mode='READ', **kwargs):
        plugin = []
        if mode.upper() == 'READ':
            checkmode = lambda a: a.is_read_supported()
        elif mode.upper() == 'WRITE':
            checkmode = lambda a: a.is_write_supported()
        elif mode.upper() == 'BOTH':
            checkmode = lambda a: (
                a.is_read_supported() and a.is_write_supported())
        else:
            raise ValueError('Invalid mode!')
        if 'experiment_type' in kwargs:
            plugin = [p for p in cls._plugins if p.name ==
                      kwargs['experiment_type'] and checkmode(p)]
        if not plugin:
            plugin = [
                p for p in cls._plugins if p.check_if_applies(filename) and checkmode(p)]
        if not plugin:
            raise ValueError('No plugin can handle ' + str(filename))
        return plugin[0]

    @classmethod
    def _autoguess_experiment_type(cls, fileformat_or_name):
        plugin = [
            p for p in cls._plugins if p.check_if_applies(fileformat_or_name)]
        if not plugin:
            raise SASExposureException(
                'Cannot find appropriate IO plugin for file ' + fileformat_or_name)
        return plugin[0]

    @staticmethod
    def _initialize_kwargs(kwargs):
        d = {'dirs': [], 'experiment_type': None, 'load_mask': True, 'maskfile': None,
             'generator': False, 'error_on_not_found': True}
        for k in d:
            if k not in kwargs:
                kwargs[k] = d[k]

    def __init__(self, *args, **kwargs):
        """Initialize the already constructed instance. This is hardly ever
        called directly since `__new__()` is implemented. See there for the
        usual cases of object construction.
        """
        if hasattr(self, '_was_init'):
            # an ugly hack to avoid duplicate __init__()-s, since whenever __new__()
            # returns an instance of this class, the __init__() method is executed
            # with the _SAME_ arguments with which __new__() was originally
            # called.
            return
        self._was_init = True
        ArithmeticBase.__init__(self)
        self._initialize_kwargs(kwargs)
        if not args:
            # no positional arguments: create an empty SASExposure
            self.Intensity = None
            self.Error = None
            self.header = SASHeader()
            self.mask = None
        elif len(args) == 1 and isinstance(args[0], SASExposure):
            # make a deep copy of an existing SASExposure
            if isinstance(args[0].Intensity, np.ndarray):
                self.Intensity = args[0].Intensity.copy()
            else:
                self.Intensity = args[0].Intensity
            if isinstance(args[0].Error, np.ndarray):
                self.Error = args[0].Error.copy()
            else:
                self.Error = args[0].Error
            if isinstance(args[0].header, SASHeader):
                self.header = SASHeader(args[0].header)
            else:
                self.header = args[0].header
            if isinstance(args[0].mask, SASMask):
                self.mask = SASMask(args[0].mask)
            else:
                self.mask = args[0].mask
        elif len(args) == 1 and isinstance(args[0], dict):
            # create SASExposure from a dict
            a = {'Intensity': None, 'Error': None,
                 'header': SASHeader(), 'mask': None}
            a.update(args[0])
            if isinstance(a['Intensity'], np.ndarray):
                self.Intensity = a['Intensity'].copy()
            else:
                self.Intensity = a['Intensity']
            if isinstance(a['Error'], np.ndarray):
                self.Error = a['Error'].copy()
            else:
                self.Error = a['Error']
            if isinstance(a['header'], SASHeader):
                self.header = SASHeader(a['header'])
            else:
                self.header = a['header']
            if isinstance(a['mask'], SASMask):
                self.mask = SASMask(a['mask'])
            else:
                self.mask = a['mask']
        elif isinstance(args[0], np.ndarray):
            # convert a numpy matrix to a SASExposure
            self.Intensity = args[0]
            if len(args) > 1 and isinstance(args[1], np.ndarray):
                self.Error = args[1]
            else:
                self.Error = None
            if len(args) > 2:
                self.header = SASHeader(args[2])
            else:
                self.header = SASHeader()
            if len(args) > 3:
                self.mask = SASMask(args[3])
            else:
                self.mask = None
        if not isinstance(self.header, SASHeader):
            self.header = SASHeader(self.header)
        if kwargs['maskfile'] is not None and kwargs['load_mask']:
            self.set_mask(SASMask(kwargs['maskfile'], dirs=kwargs['dirs']))

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

        2) ``SASExposure(<dict>)``: make a SASExposure from a dictionary. The
            given parameter should contain the field 'Intensity' (np.ndarray)
            and optionally 'Error' (np.ndarray), 'header' and 'mask'. The last two
            must be convertible to SASHeader and SASMask, respectively.

        3) ``SASExposure(Intensity [, Error [, header [, mask]]])`` where
            ``Intensity`` and ``Error`` are np.ndarrays, ``header`` and ``mask``
            are convertible to SASHeader and SASMask.

        4) ``SASExposure(filename_or_other_obj, **kwargs)``: load a file
            **One positional argument**
            `filename_or_other_obj`: file name or other object which can be handled
                by one of the I/O plugins.

        5) ``SASExposure(fileformat_or_other_obj, fsn, **kwargs)``: load
            one or more exposures. **Two positional arguments**.
            `fileformat_or_other_obj`: string or other
                usually a C-style file format, containing a directive for
                substitution of a number, i.e. ``org_%05d.cbf`` or ``s%07d.bdf`` or
                ``sc3269_0_%04dccd``. But can be e.g. a HDF5 Group.

            `fsn`: number or a sequence of numbers
                file sequence number. If a scalar, the corresponding file will
                be loaded. If a sequence of numbers, each file will be opened
                and a list (or a generator, depending on the value of the `generator`
                keyword argument) of SASExposure instances will be returned.

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
        """
        cls._initialize_kwargs(kwargs)
        if not args:
            # no positional arguments are specified: create an empty SASExposure.
            # we cannot call SASExposure(), since this would cause an infinite
            # loop.
            return super(SASExposure, cls).__new__(cls)
        elif (isinstance(args[0], SASExposure) or isinstance(args[0], np.ndarray)
              or isinstance(args[0], dict)):
            # copy an existing SASExposure, make a new SASExposure from a numpy
            # array or from a dict.
            # we cannot call SASExposure(*args,**kwargs), since it will cause
            # an infinite loop
            # will call __init__ implicitly with args and kwargs
            return super(SASExposure, cls).__new__(cls)
        else:
            # Everything else is handled by IO plugins
            plugin = cls.get_IOplugin(args[0], 'READ')
            if len(args) == 2:
                if not isinstance(args[1], collections.Sequence):
                    fsns = [args[1]]
                else:
                    fsns = args[1]
                res = plugin.read_multi(args[0], fsns, **kwargs)
            elif len(args) == 1:
                res = plugin.read(args[0], **kwargs)
            else:
                raise ValueError('Invalid number of positional arguments.')
            if isinstance(res, dict):
                obj = super(SASExposure, cls).__new__(cls)
                # now the _was_init hack comes handy.
                obj.__init__(res, **kwargs)
                return obj
            else:
                gen = cls._read_multi(res, **kwargs)
                if len(args) == 2 and not isinstance(args[1], collections.Sequence):
                    return list(gen)[0]
                elif kwargs['generator']:
                    return gen
                else:
                    return list(gen)

    @classmethod
    def _read_multi(cls, obj, **kwargs):
        for o in obj:
            yield cls(o, **kwargs)  # this will call __new__ with a dict
        return

    def check_for_mask(self, isfatal=True):
        """Check if a valid mask is defined.

        Inputs:
        -------
            isfatal: boolean [True]
                if this is True and self.mask is not an instance of
                SASMask, raises a SASExposureException. If False,
                the error state is returned (True if we have a valid
                mask and False if not).
        """
        if self.mask is None:
            if isfatal:
                raise SASExposureException('mask not defined')  # IGNORE:W0710
            else:
                return False
        if self.mask.shape != self.shape:
            if isfatal:
                raise SASExposureException('mask is of incompatible shape!')
            else:
                return False
        return True

    def check_for_q(self, isfatal=True):
        """Check if needed header elements are present for calculating q values
        for pixels.

        Inputs:
        -------
            isfatal: boolean, True
                if missing header data should raise an exception. If this is
                False, the list of missing field names is returned (an empty
                list signifies that every required field is present).
        """
        # Note, that we check for 'Dist' and 'Wavelength' instead of 'DistCalibrated'
        # and 'WavelengthCalibrated', because if the latter are missing, they
        # default to the values of the uncalibrated ones.
        missing = [x for x in ['BeamPosX', 'BeamPosY', 'Dist', 'Wavelength', 'XPixel', 'YPixel'] if
                   x not in self.header]

        if missing:
            if isfatal:
                raise SASExposureException(
                    'Fields missing from header: ' + str(missing))  # IGNORE:W0710
            else:
                return missing
        return []

    def __del__(self):
        """respond to `del object` calls."""
        for x in ['Intensity', 'Error', 'header', 'mask']:
            if hasattr(self, x):
                delattr(self, x)
                # we do not call the __del__ method of our parent class, since neither it
                # nor its ancestor has one.

    @property
    def shape(self):
        if hasattr(self, 'Intensity') and self.Intensity is not None:
            return self.Intensity.shape
        else:
            return None

    @property
    def size(self):
        if hasattr(self, 'Intensity') and self.Intensity is not None:
            return self.Intensity.size
        else:
            return None

    def __setitem__(self, key, value):
        """respond to `obj[key] = value` calls. Delegate requests to self.header."""
        self.header[key] = value

    def __getitem__(self, key):
        """respond to `obj[key]` requests. The result depends on the type of ``key``:

        1) If ``key`` is a string, the call is delegated to self.header, i.e.
            get a header item.

        2) otherwise a copy is made of this instance with the Intensity, Error and
            mask matrices sliced accordingly.
        """
        if isinstance(key, str):
            return self.header[key]
        else:
            obj = self.__class__()
            obj.Intensity = self.Intensity[key]
            if self.Error is not None:
                obj.Error = self.Error[key]
            if self.mask is not None:
                obj.mask = self.mask[key]
            obj.header = SASHeader(self.header)
            return obj

    def __delitem__(self, key):
        """Respond to `del object[key]` calls. Delegate calls to self.header."""
        del self.header[key]

    def barycenter(self, masked=True, mask=None):
        """Calculate the center of mass of the scattering image.

        Inputs:
            masked: if the mask should be taken into account
            mask: if not None, an overriding mask to be used instead of
                the defined one.
        """
        if mask is None and self.check_for_mask(False):
            mask = self.mask
        if mask is not None and masked:
            indices = np.array(mask) != 0
        else:
            indices = slice(None)
        col, row = np.meshgrid(
            np.arange(self.shape[1]), np.arange(self.shape[0]))
        Imasked = self.Intensity[indices]
        return (row[indices] * Imasked).sum() / Imasked.sum(), (col[indices] * Imasked).sum() / Imasked.sum()

    def sigma(self, masked=True, mask=None):
        """Calculate the RMS extents of the image (or a portion of it, defined by the mask)

        Inputs:
            masked: if the mask should be taken into account
            mask: if not None, an overriding mask to be used instead of
                the defined one.
        """
        if mask is None and self.check_for_mask(False):
            mask = self.mask
        if mask is not None and masked:
            indices = np.array(mask) != 0
        else:
            indices = slice(None)
        col, row = np.meshgrid(
            np.arange(self.shape[1]), np.arange(self.shape[0]))
        Imasked = self.Intensity[indices]
        wt = Imasked / Imasked.sum()
        return (((row[indices] ** 2 * wt).sum() - (row[indices] * wt).sum() ** 2) ** 0.5,
                ((col[indices] ** 2 * wt).sum() - (col[indices] * wt).sum() ** 2) ** 0.5)

    def sum(self, masked=True, mask=None):
        """Calculate the sum of the pixels.

        Inputs:
            masked: if the mask should be taken into account
            mask: if not None, an overriding mask to be used instead of
                the defined one.
        """
        if mask is None and self.check_for_mask(False):
            mask = self.mask
        if mask is not None and masked:
            indices = np.array(mask) != 0
        else:
            indices = slice(None)
        isum = 0
        esum = 0
        try:
            isum = self.Intensity[indices].sum()
            esum = self.Error[indices].sum()
        # NoneType object has no attribute __getitem__: if self.Error is None
        except TypeError:
            return isum
        except ValueError:  # this can occur if everything is masked
            return ErrorValue(0, 0)
        return ErrorValue(isum, esum)

    def median(self, masked=True, mask=None):
        if mask is None and self.check_for_mask(False):
            mask = self.mask
        if mask is not None and masked:
            indices = np.array(mask) != 0
        else:
            indices = slice(None)
        I = self.Intensity[indices]
        E = self.Error[indices]
        try:
            m = np.median(I)
            return ErrorValue(m, E[I == m].max())
        except ValueError:  # this can occur if everything is masked
            return ErrorValue(np.nan, np.nan)

    def max(self, masked=True, mask=None):
        if mask is None and self.check_for_mask(False):
            mask = self.mask
        if mask is not None and masked:
            indices = np.array(mask) != 0
        else:
            indices = slice(None)
        I = self.Intensity[indices]
        E = self.Error[indices]
        try:
            m = I.max()
            return ErrorValue(m, E[I == m].max())
        except ValueError:  # this can occur if everything is masked
            return ErrorValue(np.nan, np.nan)

    def min(self, masked=True, mask=None):
        if mask is None and self.check_for_mask(False):
            mask = self.mask
        if mask is not None and masked:
            indices = np.array(mask) != 0
        else:
            indices = slice(None)
        I = self.Intensity[indices]
        E = self.Error[indices]
        try:
            m = I.min()
            return ErrorValue(m, E[I == m].max())
        except ValueError:  # this can occur if everything is masked
            return ErrorValue(np.nan, np.nan)

    def mean(self, masked=True, mask=None):
        if mask is None and self.check_for_mask(False):
            mask = self.mask
        if mask is not None and masked:
            indices = np.array(mask) != 0
        else:
            indices = slice(None)
        I = self.Intensity[indices]
        E = self.Error[indices]
        try:
            return ErrorValue(I.mean(), ((E ** 2).mean()) ** 0.5)
        except ValueError:  # this can occur if everything is masked
            return ErrorValue(np.nan, np.nan)

    def std(self, masked=True, mask=None):
        if mask is None and self.check_for_mask(False):
            mask = self.mask
        if mask is not None and masked:
            indices = np.array(mask) != 0
        else:
            indices = slice(None)
        I = self.Intensity[indices]
        E = self.Error[indices]
        try:
            return ErrorValue(I.std(), ((E ** 2).std()) ** 0.5)
        except ValueError:  # this can occur if everything is masked
            return ErrorValue(np.nan, np.nan)

    def trimpix(self, rowmin, rowmax, columnmin, columnmax):
        obj = self[rowmin:rowmax, columnmin:columnmax]
        if 'BeamPosX' in obj.header:
            obj.header['BeamPosX'] -= rowmin
        if 'BeamPosY' in obj.header:
            obj.header['BeamPosY'] -= columnmin
        return obj

    def trimq(self, qrowmin, qrowmax, qcolmin, qcolmax):
        rowmin, colmin = self.qtopixel(qrowmin, qcolmin)
        rowmax, colmax = self.qtopixel(qrowmax, qcolmax)
        return self.trimpix(int(np.floor(rowmin)), int(np.ceil(rowmax)), int(np.floor(colmin)), int(np.ceil(colmax)))

    def pixeltoq(self, row, col):
        return (4 * np.pi * np.sin(
            0.5 * np.arctan((row - self.header['BeamPosX']) * self.header['XPixel'] / self.header['DistCalibrated'])) /
            self.header['WavelengthCalibrated'],
            4 * np.pi * np.sin(0.5 * np.arctan(
                (col - self.header['BeamPosY']) * self.header['YPixel'] / self.header['DistCalibrated'])) /
            self.header['WavelengthCalibrated'])

    def qtopixel(self, qrow, qcol):
        return (self.header['DistCalibrated'] * np.tan(
            2 * np.arcsin(qrow * self.header['WavelengthCalibrated'] / (4 * np.pi))) / self.header['XPixel'] +
            self.header['BeamPosX'],
            self.header['DistCalibrated'] * np.tan(
            2 * np.arcsin(qcol * self.header['WavelengthCalibrated'] / (4 * np.pi))) / self.header['YPixel'] +
            self.header['BeamPosY'])

    def qtopixel_radius(self, q):
        return (
            self.header['DistCalibrated'] * np.tan(
                2 * np.arcsin(q * self.header['WavelengthCalibrated'] / (4 * np.pi))) /
            self.header['PixelSize'])

    def pixeltoq_radius(self, pix):
        return 4 * np.pi * np.sin(0.5 * np.arctan(pix * self.header['PixelSize'] / self.header['DistCalibrated'])) / \
            self.header['WavelengthCalibrated']

    # -------------- Loading routines (new_from_xyz) ------------------------
    @classmethod
    def get_valid_experiment_types(cls, mode='read'):
        """Get the available plugins which support 'read', 'write' or 'both'.
        """
        if mode.upper() == 'READ':
            return [p.name for p in cls._plugins if p.is_read_supported()]
        elif mode.upper() == 'WRITE':
            return [p.name for p in cls._plugins if p.is_write_supported()]
        elif mode.upper() == 'BOTH':
            return [p.name for p in cls._plugins if p.is_read_supported() and p.is_write_supported()]
        else:
            raise ValueError('invalid mode')

    # ------------------- Interface routines ---------------------------------
    def set_mask(self, mask=None):
        if mask is None:
            mask1 = SASMask(np.ones(self.shape, dtype=np.uint8))
        else:
            mask1 = SASMask(mask)
        if self.shape != mask1.shape:
            raise SASExposureException(
                'Invalid shape for mask: %s instead of %s.' % (str(mask1.shape), str(self.shape)))
        self.mask = mask1
        self.header['maskid'] = self.mask.maskid
        self.header.add_history(
            'Mask %s associated to exposure.' % self.mask.maskid)

    def get_matrix(self, name='Intensity', othernames=None):
        name = self.get_matrix_name(name, othernames)
        return getattr(self, name)

    def get_matrix_name(self, name='Intensity', othernames=None):
        if name in list(self.matrices.values()):
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

    # ------------------- simple arithmetics ---------------------------------

    def _check_arithmetic_compatibility(self, other):
        if isinstance(other, numbers.Number):
            return ErrorValue(other)
        elif isinstance(other, np.ndarray):
            if len(other) == 1:
                return ErrorValue(other[0])
            elif other.shape == self.shape:
                return ErrorValue(other)
            else:
                raise ValueError('Incompatible shape!')
        elif isinstance(other, SASExposure):
            if other.shape == self.shape:
                return ErrorValue(other.Intensity, other.Error)
            else:
                raise ValueError('Incompatible shape!')
        elif isinstance(other, ErrorValue):
            if isinstance(other.val, numbers.Number):
                return other
            elif isinstance(other.val, np.ndarray):
                if other.val.shape == self.shape and other.err.shape == self.shape:
                    return other
                else:
                    raise ValueError('Incompatible shape!')
        else:
            print('!!!!!!!!!!!!!!! check_arithmetic_compatibility failed with type: %s!!!!!!!!!!!!!' % str(
                type(other)))
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
        obj = SASExposure(self)
        obj.Intensity = -obj.Intensity
        return obj

    def __reciprocal__(self):
        obj = SASExposure(self)
        if obj.Error is None:
            obj.Error = np.zeros_like(self.Intensity)
        obj.Error = obj.Error / (self.Intensity * self.Intensity)
        obj.Intensity = 1.0 / self.Intensity
        return obj

    def __imul__(self, other):
        try:
            other = self._check_arithmetic_compatibility(other)
        except NotImplementedError:
            return NotImplemented
        if self.Error is None:
            self.Error = np.zeros_like(self.Intensity)
        self.Error = np.sqrt(
            (self.Intensity * other.err) ** 2 + (self.Error * other.val) ** 2)
        self.Intensity = self.Intensity * other.val
        return self

    def __pow__(self, other, modulo=None):
        if modulo is not None:
            return NotImplemented
        try:
            other = self._check_arithmetic_compatibility(other)
        except NotImplementedError:
            return NotImplemented
        obj = SASExposure(self)
        if obj.Error is None:
            obj.Error = np.zeros_like(self.Intensity)
        obj.Error = ((obj.Intensity ** (other.val - 1) * other.val * obj.Error) ** 2 + (
            np.log(obj.Intensity) * obj.Intensity ** other.val * other.err) ** 2) ** 0.5
        obj.Intensity = obj.Intensity ** other.val
        return obj

    def __array__(self, dt=None):
        if dt is None:
            return self.Intensity
        else:
            return self.Intensity.astype(dt)

    # ------------------- Routines for radial integration --------------------

    def get_qrange(self, N=None, spacing='linear'):
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
        qrange = utils2d.integrate.autoqscale(self.header['WavelengthCalibrated'],
                                              self.header['DistCalibrated'],
                                              self.header['XPixel'],
                                              self.header['YPixel'],
                                              self.header['BeamPosX'],
                                              self.header['BeamPosY'], 1 - self.mask.mask)
        if N is None:
            N = len(qrange)
        if isinstance(N, numbers.Integral):
            if spacing.upper().startswith('LIN'):
                return np.linspace(qrange.min(), qrange.max(), N)
            elif spacing.upper().startswith('LOG'):
                return np.logspace(np.log10(qrange.min()), np.log10(qrange.max()), N)
        elif isinstance(N, numbers.Real):
            return np.arange(qrange.min(), qrange.max(), N)
        else:
            raise NotImplementedError

    def get_pixrange(self, N=None, spacing='linear'):
        """Calculate the available pixel-range.

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
        dp = self.Dpix[self.mask.mask != 0]
        pixmin, pixmax, npix = dp.min(), dp.max(), np.absolute(
            np.ceil(dp.max() - dp.min()))
        npix = int(npix)
        if N is None:
            N = npix
        if isinstance(N, numbers.Integral):
            if spacing.upper().startswith('LIN'):
                return np.linspace(pixmin, pixmax, N)
            elif spacing.upper().startswith('LOG'):
                return np.logspace(np.log10(pixmin), np.log10(pixmax), N)
        elif isinstance(N, numbers.Real):
            return np.arange(pixmin, pixmax, N)
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

    @classmethod
    def common_pixrange(cls, *exps):
        """Find a common pixel-range for several exposures

        Usage:

        >>> SASExposure.common_pixrange(exp1, exp2, exp3, ...)

        where exp1, exp2, exp3... are instances of the SASExposure class.

        Returns:
            the estimated common pixel-range in a np.ndarray (ndim=1).
        """
        if not all([isinstance(e, cls) for e in exps]):
            raise ValueError('All arguments should be SASExposure instances.')
        pixranges = [e.get_pixrange() for e in exps]
        pixmin = max([pr.min() for pr in pixranges])
        pixmax = min([pr.max() for pr in pixranges])
        N = min([len(pr) for pr in pixranges])
        return np.linspace(pixmin, pixmax, N)

    def radial_average(self, qrange=None, pixel=False, returnmask=False,
                       errorpropagation=3, abscissa_errorpropagation=3):
        """Do a radial averaging

        Inputs:
            qrange: the q-range. If None, auto-determine. If 'linear', auto-determine
                with linear spacing (same as None). If 'log', auto-determineű
                with log10 spacing.
            pixel: do a pixel-integration (instead of q)
            returnmask: if the effective mask matrix is to be returned.
            errorpropagation: the type of error propagation (3: highest of squared or
                std-dev, 2: squared, 1: linear, 0: independent measurements of
                the same quantity)
            abscissa_errorpropagation: the type of the error propagation in the
                abscissa (3: highest of squared or std-dev, 2: squared, 1: linear,
                0: independent measurements of the same quantity)

        Outputs:
            the one-dimensional curve as an instance of SASCurve (if pixel is
                False) or SASPixelCurve (if pixel is True)
            the mask matrix (if returnmask was True)
        """
        self.check_for_mask()
        if isinstance(qrange, str):
            if qrange == 'linear':
                qrange = None
                autoqrange_linear = True
            elif qrange == 'log':
                qrange = None
                autoqrange_linear = False
            else:
                raise NotImplementedError(
                    'Value given for qrange (''%s'') not understood.' % qrange)
        else:
            autoqrange_linear = True  # whatever
        if not pixel:
            res = utils2d.integrate.radint_fullq_errorprop(self.Intensity, self.Error,
                                                           self.header[
                                                               'WavelengthCalibrated'],
                                                           self.header[
                                                               'WavelengthCalibratedError'],
                                                           self.header[
                                                               'DistCalibrated'],
                                                           self.header[
                                                               'DistCalibratedError'],
                                                           self.header['XPixel'], self.header[
                                                               'YPixel'],
                                                           self.header[
                                                               'BeamPosX'], 0,
                                                           self.header[
                                                               'BeamPosY'], 0,
                                                           (self.mask.mask == 0).astype(
                                                               np.uint8),
                                                           qrange, returnmask=returnmask, errorpropagation=errorpropagation,
                                                           autoqrange_linear=autoqrange_linear, abscissa_kind=0,
                                                           abscissa_errorpropagation=abscissa_errorpropagation)
            q, dq, I, E, A = res[:5]
            if returnmask:
                retmask = res[5]
            p = self.qtopixel_radius(q)
            ds = SASCurve(q, I, E, dq, pixel=p, area=A)
        else:
            res = utils2d.integrate.radint_fullq_errorprop(self.Intensity, self.Error, self.header['WavelengthCalibrated'],
                                                           self.header[
                                                               'WavelengthCalibratedError'],
                                                           self.header[
                                                               'DistCalibrated'],
                                                           self.header[
                                                               'DistCalibratedError'],
                                                           self.header['XPixel'], self.header[
                                                               'YPixel'],
                                                           self.header[
                                                               'BeamPosX'], 0,
                                                           self.header[
                                                               'BeamPosY'], 0,
                                                           1 -
                                                           self.mask.mask, qrange,
                                                           returnmask=returnmask, errorpropagation=errorpropagation,
                                                           autoqrange_linear=autoqrange_linear, abscissa_kind=3,
                                                           abscissa_errorpropagation=abscissa_errorpropagation)
            p, dp, I, E, A = res[:5]
            if returnmask:
                retmask = res[5]
            ds = SASPixelCurve(p, I, E, dp, area=A)
        ds.header = SASHeader(self.header)
        if returnmask:
            return ds, retmask
        else:
            return ds

    def radial_average_old(self, qrange=None, pixel=False, matrix='Intensity',
                           errormatrix='Error', q_average=True, returnmask=False,
                           errorpropagation=2):
        """Do a radial averaging

        Inputs:
            qrange: the q-range. If None, auto-determine. If 'linear', auto-determine
                with linear spacing (same as None). If 'log', auto-determineű
                with log10 spacing.
            pixel: do a pixel-integration (instead of q)
            matrix: matrix to use for averaging
            errormatrix: error matrix to use for averaging (or None)
            q_average: if the averaged abscissa values are to be returned
                instead of the nominal ones
            returnmask: if the effective mask matrix is to be returned.
            errorpropagation: the type of error propagation (2: squared,
                1: linear, 0: independent measurements of the same quantity)

        Outputs:
            the one-dimensional curve as an instance of SASCurve (if pixel is
                False) or SASPixelCurve (if pixel is True)
            the mask matrix (if returnmask was True)
        """
        self.check_for_mask()
        mat = getattr(self, matrix).astype(np.double)
        if errormatrix is not None:
            err = getattr(self, errormatrix).astype(np.double)
        else:
            err = None
        if isinstance(qrange, str):
            if qrange == 'linear':
                qrange = None
                autoqrange_linear = True
            elif qrange == 'log':
                qrange = None
                autoqrange_linear = False
            else:
                raise NotImplementedError(
                    'Value given for qrange (''%s'') not understood.' % qrange)
        else:
            autoqrange_linear = True  # whatever
        if not pixel:
            res = utils2d.integrate.radint_fullq(mat, err, self.header['WavelengthCalibrated'],
                                                 self.header['DistCalibrated'],
                                                 (self.header[
                                                  'XPixel'], self.header['YPixel']),
                                                 self.header['BeamPosX'], self.header[
                                                     'BeamPosY'],
                                                 (self.mask.mask == 0).astype(
                                                     np.uint8),
                                                 qrange, returnavgq=q_average,
                                                 returnmask=returnmask, errorpropagation=errorpropagation,
                                                 autoqrange_linear=autoqrange_linear)
            i = 0
            q = res[i]
            i += 1
            I = res[i]
            i += 1
            if err is not None:
                E = res[i]
                i += 1
            else:
                E = np.zeros_like(q)
            A = res[i]
            i += 1
            if returnmask:
                retmask = res[i]
                i += 1
            p = self.qtopixel_radius(q)
            ds = SASCurve(q, I, E, pixel=p, area=A)
        else:
            res = utils2d.integrate.radintpix(mat, err,
                                              self.header['BeamPosX'],
                                              self.header['BeamPosY'],
                                              1 - self.mask.mask, qrange,
                                              returnmask=returnmask,
                                              returnavgpix=q_average, errorpropagation=errorpropagation,
                                              autoqrange_linear=autoqrange_linear)
            i = 0
            p = res[i]
            i += 1
            I = res[i]
            i += 1
            if err is not None:
                E = res[i]
                i += 1
            else:
                E = np.zeros_like(p)
            A = res[i]
            i += 1
            if returnmask:
                retmask = res[i]
                i += 1
            ds = SASPixelCurve(p, I, E, area=A)
        ds.header = SASHeader(self.header)
        if returnmask:
            return ds, retmask
        else:
            return ds

    def sector_average(self, phi0, dphi, qrange=None, pixel=False,
                       matrix='Intensity', errormatrix='Error',
                       symmetric_sector=False, q_average=True,
                       returnmask=False, errorpropagation=2):
        """Do a radial averaging restricted to one sector.

        Inputs:
            phi0: start of the sector (radians).
            dphi: sector width (radians)
            qrange: the q-range. If None, auto-determine. If 'linear', auto-determine
                with linear spacing (same as None). If 'log', auto-determineű
                with log10 spacing.
            pixel: do a pixel-integration (instead of q)
            matrix: matrix to use for averaging
            errormatrix: error matrix to use for averaging (or None)
            symmetric_sectors: if the sector should be symmetric (phi0+pi needs
                also be taken into account)
            q_average: if the averaged abscissa values are to be returned
                instead of the nominal ones
            returnmask: if the effective mask matrix is to be returned.

        Outputs:
            the one-dimensional curve as an instance of SASCurve (if pixel is
                False) or SASPixelCurve (if pixel is True)
            the mask matrix (if returnmask was True)

        Notes:
            x is row direction, y is column. 0 degree is +x, 90 degree is +y.
        """
        self.check_for_mask()
        mat = getattr(self, matrix).astype(np.double)
        if errormatrix is not None:
            err = getattr(self, errormatrix).astype(np.double)
        else:
            err = None
        if isinstance(qrange, str):
            if qrange == 'linear':
                qrange = None
                autoqrange_linear = True
            elif qrange == 'log':
                qrange = None
                autoqrange_linear = False
            else:
                raise NotImplementedError(
                    'Value given for qrange (''%s'') not understood.' % qrange)
        else:
            autoqrange_linear = True  # whatever
        if not pixel:
            res = utils2d.integrate.radint(mat, err,
                                           self.header['WavelengthCalibrated'],
                                           self.header['DistCalibrated'],
                                           (self.header['XPixel'],
                                            self.header['YPixel']),
                                           self.header['BeamPosX'],
                                           self.header['BeamPosY'],
                                           1 - self.mask.mask, qrange,
                                           returnavgq=q_average,
                                           returnpixel=True,
                                           returnmask=returnmask,
                                           phi0=phi0, dphi=dphi, symmetric_sector=symmetric_sector,
                                           errorpropagation=errorpropagation, autoqrange_linear=autoqrange_linear)
            i = 0
            q = res[i]
            i += 1
            I = res[i]
            i += 1
            if err is not None:
                E = res[i]
                i += 1
            else:
                E = np.zeros_like(q)
            A = res[i]
            i += 1
            if returnmask:
                retmask = res[i]
                i += 1
            p = res[i]
            i += 1
            ds = SASCurve(q, I, E, pixel=p, area=A)
        else:
            res = utils2d.integrate.radintpix(mat, err,
                                              self.header['BeamPosX'],
                                              self.header['BeamPosY'],
                                              1 - self.mask.mask, qrange,
                                              returnavgpix=q_average, phi0=phi0,
                                              returnmask=returnmask,
                                              dphi=dphi, symmetric_sector=symmetric_sector,
                                              errorpropagation=errorpropagation, autoqrange_linear=autoqrange_linear)
            i = 0
            p = res[i]
            i += 1
            I = res[i]
            i += 1
            if err is not None:
                E = res[i]
                i += 1
            else:
                E = np.zeros_like(p)
            A = res[i]
            i += 1
            if returnmask:
                retmask = res[i]
                i += 1
            ds = SASPixelCurve(p, I, E, area=A)
        ds.header = SASHeader(self.header)
        if returnmask:
            return ds, retmask
        else:
            return ds

    def slice_average(self, phi0, width, qrange=None, pixel=False,
                      matrix='Intensity', errormatrix='Error',
                      symmetric_slice=False, q_average=True,
                      returnmask=False, errorpropagation=2):
        """Do a radial averaging restricted to one sector.

        Inputs:
            phi0: direction of the slice (radians).
            width: slice width (pixels)
            qrange: the q-range. If None, auto-determine. If 'linear', auto-determine
                with linear spacing (same as None). If 'log', auto-determineű
                with log10 spacing.
            pixel: do a pixel-integration (instead of q)
            matrix: matrix to use for averaging
            errormatrix: error matrix to use for averaging (or None)
            symmetric_slice: if the slice should be symmetric (phi0+pi needs
                also be taken into account)
            q_average: if the averaged abscissa values are to be returned
                instead of the nominal ones
            returnmask: if the effective mask matrix is to be returned.

        Outputs:
            the one-dimensional curve as an instance of SASCurve (if pixel is
                False) or SASPixelCurve (if pixel is True)
            the mask matrix (if returnmask was True)

        Notes:
            x is row direction, y is column. 0 degree is +x, 90 degree is +y.
        """
        self.check_for_mask()
        mat = getattr(self, matrix).astype(np.double)
        if errormatrix is not None:
            err = getattr(self, errormatrix).astype(np.double)
        else:
            err = None
        if isinstance(qrange, str):
            if qrange == 'linear':
                qrange = None
                autoqrange_linear = True
            elif qrange == 'log':
                qrange = None
                autoqrange_linear = False
            else:
                raise NotImplementedError(
                    'Value given for qrange (''%s'') not understood.' % qrange)
        else:
            autoqrange_linear = True  # whatever
        if not pixel:
            res = utils2d.integrate.radint(mat, err,
                                           self.header['WavelengthCalibrated'],
                                           self.header['DistCalibrated'],
                                           (self.header['XPixel'],
                                            self.header['YPixel']),
                                           self.header['BeamPosX'],
                                           self.header['BeamPosY'],
                                           1 - self.mask.mask, qrange,
                                           returnavgq=q_average,
                                           returnpixel=True,
                                           returnmask=returnmask,
                                           phi0=phi0, dphi=width,
                                           doslice=True,
                                           symmetric_sector=symmetric_slice,
                                           errorpropagation=errorpropagation, autoqrange_linear=autoqrange_linear)
            i = 0
            q = res[i]
            i += 1
            I = res[i]
            i += 1
            if err is not None:
                E = res[i]
                i += 1
            else:
                E = np.zeros_like(q)
            A = res[i]
            i += 1
            if returnmask:
                retmask = res[i]
                i += 1
            p = res[i]
            i += 1
            ds = SASCurve(q, I, E, pixel=p, area=A)
        else:
            res = utils2d.integrate.radintpix(mat, err,
                                              self.header['BeamPosX'],
                                              self.header['BeamPosY'],
                                              1 - self.mask.mask, qrange,
                                              returnavgpix=q_average, phi0=phi0,
                                              dphi=width, symmetric_sector=symmetric_slice,
                                              returnmask=returnmask,
                                              doslice=True,
                                              errorpropagation=errorpropagation, autoqrange_linear=autoqrange_linear)
            i = 0
            p = res[i]
            i += 1
            I = res[i]
            i += 1
            if err is not None:
                E = res[i]
                i += 1
            else:
                E = np.zeros_like(p)
            A = res[i]
            i += 1
            if returnmask:
                retmask = res[i]
                i += 1
            ds = SASPixelCurve(p, I, E, area=A)
        ds.header = SASHeader(self.header)
        if returnmask:
            return ds, retmask
        else:
            return ds

    def azimuthal_average(self, qmin, qmax, Ntheta=100, pixel=False,
                          matrix='Intensity', errormatrix='Error', returnmask=False,
                          errorpropagation=2):
        """Do an azimuthal averaging restricted to a ring.

        Inputs:
            qmin, qmax: lower and upper bounds of the ring (q or pixel)
            Ntheta: number of points in the output.
            pixel: do a pixel-integration (instead of q)
            matrix: matrix to use for averaging
            errormatrix: error matrix to use for averaging (or None)
            returnmask: if the effective mask matrix is to be returned.

        Outputs:
            the one-dimensional curve as an instance of SASAzimuthalCurve
            the mask matrix (if returnmask was True)

        Notes:
            x is row direction, y is column. 0 degree is +x, 90 degree is +y.
        """
        self.check_for_mask()
        mat = getattr(self, matrix).astype(np.double)
        if errormatrix is not None:
            err = getattr(self, errormatrix).astype(np.double)
        else:
            err = None
        if not pixel:
            res = utils2d.integrate.azimint(mat, err,
                                            self.header[
                                                'WavelengthCalibrated'],
                                            self.header['DistCalibrated'],
                                            (self.header[
                                             'XPixel'], self.header['YPixel']),
                                            self.header['BeamPosX'],
                                            self.header['BeamPosY'],
                                            (self.mask.mask == 0).astype(
                                                np.uint8), Ntheta,
                                            returnmask=returnmask,
                                            qmin=qmin, qmax=qmax,
                                            errorpropagation=errorpropagation)
        else:
            res = utils2d.integrate.azimintpix(mat, err,
                                               self.header['BeamPosX'],
                                               self.header['BeamPosY'],
                                               self.mask.mask == 0, Ntheta,
                                               returnmask=returnmask,
                                               pixmin=qmin, pixmax=qmax,
                                               errorpropagation=errorpropagation)
        i = 0
        theta = res[i]
        i += 1
        I = res[i]
        i += 1
        if err is not None:
            E = res[i]
            i += 1
        else:
            E = np.zeros_like(theta)
        A = res[i]
        i += 1
        if returnmask:
            retmask = res[i]
            i += 1
        ds = SASAzimuthalCurve(theta, I, E, area=A)
        ds.header = SASHeader(self.header)
        if returnmask:
            return ds, retmask
        else:
            return ds

    # ---------------------- Plotting ----------------------------------------

    def imshow(self, **kwargs):
        """Create a two-dimensional plot of the scattering pattern

        The behaviour can be fine-tuned by the following keyword arguments:

        crosshair (True): If a beam-centre cross-hair is to be drawn.
        mask (True): If the mask is to be drawn.
        mask_alpha (0.7): The opacity of the mask (1 is fully opaque, 0 is fully
            transparent)
        pixel (True): If the axes are to be represented in pixel scale.
        colorbar (True): If the color-bar is to be drawn.
        colorbar_destination (None): a matplotlib.Axes2D instance to put the
            colorbar in.
        destination (None): a matplotlib.Axes2D instance to plot the matrix to.
        """

    def plot2d(self, **kwargs):
        """Plot the matrix (imshow)

        Allowed keyword arguments [and their default values]:

        zscale ['linear']: colour scaling of the image. Either a string ('log',
            'log10', 'sqrt' or 'linear', case insensitive), or an unary function
            which operates on a numpy array and returns a numpy array of the same
            shape, e.g. np.log, np.sqrt etc.
        crosshair [True]: if a cross-hair marking the beam position is to be
            plotted.
        drawmask [True]: if the mask is to be plotted. If no mask is attached
            to this SASExposure object, it defaults to False.
        qrange_on_axis [True]: if the q-range is to be set to the axis. If no
            mask is attached to this SASExposure instance, defaults to False
        matrix ['Intensity']: the matrix which is to be plotted. If this is not
            present, another one will be chosen quietly. If None, the scattering
            image is not plotted.
        axes [None]: the axes into which the image should be plotted. If None,
            defaults to the currently active axes (returned by plt.gca())
        invalid_color ['black']: the color for invalid (NaN or infinite) pixels
        mask_opacity [0.8]: the opacity of the overlaid mask (1 is fully opaque,
            0 is fully transparent)
        minvalue: minimal value. All matrix elements below this will be replaced
            by this. Defaults to -infinity.
        maxvalue: maximal value. All matrix elements above this will be replaced
            by this. Defaults to +infinity.
        return_matrix: if the transformed, just-plotted matrix is to be
            returned. False by default.
        vmin, vmax: these are keywords for plt.imshow(). However, they are changed
            if defined, according to the value of `zscale`.
        fallback [True]: if errors such as qrange not found or mask not found or the
            like should be suppressed.
        drawcolorbar [True]: if a colorbar is to be added. Can be a boolean value
            (True or False) or an instance of matplotlib.axes.Axes, into which the
            color bar should be drawn. This cannot be other than False if argument
            'matrix' is None.

        All other keywords are forwarded to plt.imshow()

        Returns: the image instance returned by imshow()

        Notes:
            Using minvalue and maxvalue results in a clipping of the matrix before
            z-scaling and imshow()-ing. On the other hand, vmin and vmax do the
            luminance scaling in imshow(). However, vmin and vmax are adjusted with
            the same z-scaling as the matrix before invoking imshow().
        """
        kwargs_default = {'zscale': 'linear',
                          'crosshair': True,
                          'drawmask': True,
                          'qrange_on_axis': True,
                          'matrix': 'Intensity',
                          'axes': None,
                          'invalid_color': 'black',
                          'mask_opacity': 0.6,
                          'interpolation': 'nearest',
                          'origin': 'upper',
                          'minvalue': -np.inf,
                          'maxvalue': np.inf,
                          'return_matrix': False,
                          'fallback': True,
                          'drawcolorbar': True}
        my_kwargs = ['zscale', 'crosshair', 'drawmask', 'qrange_on_axis', 'matrix',
                     'axes', 'invalid_color', 'mask_opacity', 'minvalue', 'maxvalue',
                     'return_matrix', 'fallback', 'drawcolorbar']
        kwargs_default.update(kwargs)
        return_matrix = kwargs_default[
            'return_matrix']  # save this as this will be removed when kwars_default is fed into imshow()

        kwargs_for_imshow = dict(
            [(k, kwargs_default[k]) for k in kwargs_default if k not in my_kwargs])
        if isinstance(kwargs_default['zscale'], str):
            if kwargs_default['zscale'].upper().startswith('LOG10'):
                kwargs_default['zscale'] = np.log10
            elif kwargs_default['zscale'].upper().startswith('LN'):
                kwargs_default['zscale'] = np.log
            elif kwargs_default['zscale'].upper().startswith('LIN'):
                kwargs_default['zscale'] = lambda a: a * 1.0
            elif kwargs_default['zscale'].upper().startswith('LOG'):
                kwargs_default['zscale'] = np.log
            elif kwargs_default['zscale'].upper().startswith('SQRT'):
                kwargs_default['zscale'] = np.sqrt
            else:
                raise ValueError(
                    'Invalid value for zscale: %s' % kwargs_default['zscale'])
        if kwargs_default['zscale'] is np.log10:
            kwargs_for_imshow['norm'] = matplotlib.colors.LogNorm()
            kwargs_default['zscale'] = lambda a: a * 1.0
        for v in ['vmin', 'vmax']:
            if v in kwargs_for_imshow:
                kwargs_for_imshow[v] = kwargs_default[
                    'zscale'](float(kwargs_for_imshow[v]))
        if kwargs_default['matrix'] is not None:
            mat = self.get_matrix(kwargs_default['matrix']).copy()
            mat[mat < kwargs_default['minvalue']] = kwargs_default['minvalue']
            mat[mat > kwargs_default['maxvalue']] = kwargs_default['maxvalue']
            mat = kwargs_default['zscale'](mat)
        else:
            mat = None

        if kwargs_default['drawmask']:
            try:
                self.check_for_mask()
            except SASExposureException:
                if kwargs_default['fallback']:
                    warnings.warn(
                        'Drawing of mask was requested, but mask is not present!')
                    kwargs_default['drawmask'] = False
        if kwargs_default['qrange_on_axis']:
            try:
                self.check_for_q()
            except SASExposureException:
                if kwargs_default['fallback']:
                    missing = self.check_for_q(False)
                    if missing:
                        warnings.warn(
                            'Q scaling on the axes was requested, but the following fields are missing: ' + repr(
                                missing))
                        kwargs_default['qrange_on_axis'] = False

        if kwargs_default['qrange_on_axis']:
            xmin = 4 * np.pi * np.sin(0.5 * np.arctan(
                (0 - self.header['BeamPosY']) * self.header['YPixel'] / self.header['DistCalibrated'])) / self.header[
                'WavelengthCalibrated']
            xmax = 4 * np.pi * np.sin(0.5 * np.arctan(
                (mat.shape[1] - self.header['BeamPosY']) * self.header['YPixel'] / self.header['DistCalibrated'])) / \
                self.header['WavelengthCalibrated']
            ymin = 4 * np.pi * np.sin(0.5 * np.arctan(
                (0 - self.header['BeamPosX']) * self.header['XPixel'] / self.header['DistCalibrated'])) / self.header[
                'WavelengthCalibrated']
            ymax = 4 * np.pi * np.sin(0.5 * np.arctan(
                (mat.shape[0] - self.header['BeamPosX']) * self.header['XPixel'] / self.header['DistCalibrated'])) / \
                self.header['WavelengthCalibrated']
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
                bcx = None
                bcy = None
            xmin = 0
            xmax = self.shape[1]
            ymin = 0
            ymax = self.shape[0]

        if kwargs_default['axes'] is None:
            kwargs_default['axes'] = plt.gca()
        if mat is not None:
            ret = kwargs_default['axes'].imshow(
                mat, **kwargs_for_imshow)  # IGNORE:W0142
        else:
            ret = None
        if 'norm' in kwargs_for_imshow:
            del kwargs_for_imshow['norm']
        if kwargs_default['drawmask']:
            # workaround: because of the colour-scaling we do here, full one and
            #   full zero masks look the SAME, i.e. all the image is shaded.
            #   Thus if we have a fully unmasked matrix, skip this section.
            #   This also conserves memory.
            if self.mask.mask.sum() != self.mask.mask.size:
                # Mask matrix should be plotted with plt.imshow(maskmatrix,
                # cmap=_colormap_for_mask)
                _colormap_for_mask = matplotlib.colors.ListedColormap(['white', 'white'],
                                                                      '_sastool_%s' % misc.random_str(10))
                _colormap_for_mask._init()  # IGNORE:W0212
                _colormap_for_mask._lut[:, -1] = 0  # IGNORE:W0212
                _colormap_for_mask._lut[
                    0, -1] = kwargs_default['mask_opacity']  # IGNORE:W0212
                kwargs_for_imshow['cmap'] = _colormap_for_mask
                # print "kwargs_for_imshow while plotting mask: ",
                # repr(kwargs_for_imshow)
                kwargs_default['axes'].imshow(
                    self.mask.mask, **kwargs_for_imshow)  # IGNORE:W0142
        if kwargs_default['crosshair']:
            if bcx is not None and bcy is not None:
                ax = kwargs_default['axes'].axis()
                kwargs_default['axes'].plot([xmin, xmax], [bcx] * 2, 'w-')
                kwargs_default['axes'].plot([bcy] * 2, [ymin, ymax], 'w-')
                kwargs_default['axes'].axis(ax)
            else:
                warnings.warn(
                    'Cross-hair was requested but beam center was not found.')
        kwargs_default['axes'].set_axis_bgcolor(
            kwargs_default['invalid_color'])
        if kwargs_default['drawcolorbar'] and mat is not None:
            if isinstance(kwargs_default['drawcolorbar'], matplotlib.axes.Axes):
                kwargs_default['axes'].figure.colorbar(
                    ret, cax=kwargs_default['drawcolorbar'])
            else:
                # try to find a suitable colorbar axes: check if the plot target axes already
                # contains some images, then check if their colorbars exist as
                # axes.
                cax = [i.colorbar[1]
                       for i in kwargs_default['axes'].images if i.colorbar is not None]
                cax = [c for c in cax if c in c.figure.axes]
                if cax:
                    cax = cax[0]
                else:
                    
                    cax = make_axes_locatable(kwargs_default['axes']).append_axes('right',size="5%",pad=0.05)
                kwargs_default['axes'].figure.colorbar(
                    ret, cax=cax, ax=kwargs_default['axes'])
        kwargs_default['axes'].figure.canvas.draw()
        if return_matrix:
            return ret, mat
        else:
            return ret

    def get_q_extent(self):
        xmin = 4 * np.pi * np.sin(
            0.5 * np.arctan((0 - self.header['BeamPosY']) * self.header['YPixel'] / self.header['DistCalibrated'])) / \
            self.header['WavelengthCalibrated']
        xmax = 4 * np.pi * np.sin(0.5 * np.arctan(
            (self.shape[1] - 1 - self.header['BeamPosY']) * self.header['YPixel'] / self.header['DistCalibrated'])) / \
            self.header['WavelengthCalibrated']
        ymin = 4 * np.pi * np.sin(
            0.5 * np.arctan((0 - self.header['BeamPosX']) * self.header['XPixel'] / self.header['DistCalibrated'])) / \
            self.header['WavelengthCalibrated']
        ymax = 4 * np.pi * np.sin(0.5 * np.arctan(
            (self.shape[0] - 1 - self.header['BeamPosX']) * self.header['XPixel'] / self.header['DistCalibrated'])) / \
            self.header['WavelengthCalibrated']
        return (xmin, xmax, ymin, ymax)

    # ------------------------ Beam center finding ---------------------------

    def update_beampos(self, bc, source=None):
        """Update the beam position in the header.

        Inputs:
            bc: beam position coordinates (row, col; starting from 0).
            source: name of the beam finding algorithm.
        """
        self.header['BeamPosX'], self.header['BeamPosY'] = bc
        if not source:
            self.header.add_history(
                'Beam position updated to:' + str(tuple(bc)))
        else:
            self.header.add_history(
                'Beam found by *%s*: %s' % (source, str(tuple(bc))))

    def find_beam_semitransparent(self, bs_area, threshold=0.05, update=True, callback=None):
        """Find the beam position from the area under the semitransparent
        beamstop.

        Inputs:
            bs_area: sequence of the coordinates of the beam-stop area rect.:
                [row_min, row_max, column_min, column_max]
            threshold: threshold value for eliminating low-intensity pixels. Set
                it to None to skip this refinement.
            update: if the new value should be written in the header (default).
                If False, the newly found beam position is only returned.
            callback: dummy parameter, kept for similar appearence as the other
                find_beam_*() functions.

        Outputs:
            the beam position (row,col).
        """
        bs_area = [min(bs_area[2:]), max(bs_area[2:]), min(
            bs_area[:2]), max(bs_area[:2])]
        bc = utils2d.centering.findbeam_semitransparent(
            self.get_matrix(), bs_area, threshold)
        if update:
            self.update_beampos(bc, source='semitransparent')
        return bc

    def find_beam_slices(self, pixmin=0, pixmax=np.inf, sector_width=np.pi / 9.,
                         extent=10, update=True, callback=None):
        """Find the beam position by matching diagonal sectors.

        Inputs:
            pixmin, pixmax: lower and upper thresholds in the distance from the
                origin in the radial averaging [in pixel units]
            sector_width: width of sectors in radian.
            extent: expected distance of the true beam position from the current
                one. Just the magnitude of it counts.
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
                                               self.mask.mask, dmin=pixmin,
                                               dmax=pixmax,
                                               sector_width=sector_width,
                                               extent=extent,
                                               callback=callback)
        if update:
            self.update_beampos(bc, source='slices')
        return bc

    def find_beam_gravity(self, update=True, callback=None):
        """Find the beam position by finding the center of gravity in each row
        and column.

        Inputs:
            update: if the new value should be written in the header (default).
                If False, the newly found beam position is only returned.
            callback: dummy parameter, not used. It is only there to have a
                similar signature of all find_beam_*() functions

        Outputs:
            the beam position (row,col).
        """
        self.check_for_mask()
        bc = utils2d.centering.findbeam_gravity(
            self.get_matrix(), self.mask.mask)
        if update:
            self.update_beampos(bc, source='gravity')
        return bc

    def find_beam_azimuthal_fold(self, Ntheta=50, dmin=0, dmax=np.inf,
                                 extent=10, update=True, callback=None):
        """Find the beam position by matching an azimuthal scattering curve
        and its counterpart shifted by pi radians.

        Inputs:
            Ntheta: number of bins in the azimuthal scattering curve
            dmin, dmax: lower and upper thresholds in the distance from the
                origin in the radial averaging [in pixel units]
            extent: expected distance of the true beam position from the current
                one. Just the magnitude of it counts.
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
                                                       Ntheta=Ntheta, dmin=dmin,
                                                       dmax=dmax,
                                                       extent=extent, callback=callback)
        if update:
            self.update_beampos(bc, source='azimuthal_fold')
        return bc

    def find_beam_radialpeak(self, pixmin, pixmax, drive_by='amplitude', extent=10,
                             update=True, callback=None):
        """Find the beam position by optimizing a peak in the radial scattering
        curve.

        Inputs:
            pixmin, pixmax: lower and upper thresholds in the distance from the
                origin in the radial averaging [in pixel units]. Should be a
                narrow interval, zoomed onto one peak.
            drive_by: 'amplitude' if the amplitude of the peak has to be maximized
                or 'hwhm' if the hwhm should be minimized.
            extent: expected distance of the true beam position from the current
                one. Just the magnitude of it counts.
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
                                                   pixmax, drive_by=drive_by,
                                                   extent=extent, callback=callback)
        if update:
            self.update_beampos(bc, source='radialpeak')
        return bc

    def find_beam_Guinier(self, pixmin, pixmax, extent=10,
                          update=True, callback=None):
        """Find the beam position by maximizing of a radius of gyration in the
        radial scattering curve (i.e. minimizing the FWHM of a Gaussian centered
        at the origin).

        Inputs:
            pixmin, pixmax: lower and upper thresholds in the distance from the
                origin in the radial averaging [in pixel units]. Should be a
                narrow interval, zoomed onto a Guinier range.
            extent: expected distance of the true beam position from the current
                one. Just the magnitude of it counts.
            update: if the new value should be written in the header (default).
                If False, the newly found beam position is only returned.
            callback: a function (accepting no arguments) to be called in each
                iteration of the fitting procedure.

        Outputs:
            the beam position (row,col).
        """
        self.check_for_mask()
        bc = utils2d.centering.findbeam_Guinier(self.get_matrix(),
                                                (self.header['BeamPosX'],
                                                 self.header['BeamPosY']),
                                                self.mask.mask, pixmin,
                                                pixmax,
                                                extent=extent, callback=callback)
        if update:
            self.update_beampos(bc, source='Guinier')
        return bc

    def find_beam_powerlaw(self, pixmin, pixmax, drive_by='R2',
                           update=True, callback=None):
        """Find the beam position by power-law fitting. Beam position is where the parameter
        designated by ``drive_by`` is optimal. 
        """

        self.check_for_mask()
        bc = utils2d.centering.findbeam_powerlaw(self.get_matrix(),
                                                 (self.header['BeamPosX'],
                                                  self.header['BeamPosY']),
                                                 self.mask.mask, pixmin,
                                                 pixmax, drive_by=drive_by,
                                                 callback=callback)
        if update:
            self.update_beampos(bc, source='PowerLaw')
        return bc
    # ----------------------- Writing routines -------------------------------

    def write(self, writeto, plugin=None, **kwargs):
        if plugin is None:
            plugin = self.get_IOplugin(writeto, 'WRITE')
        else:
            plugin = [p for p in self._plugins if p.name == plugin][0]
        plugin.write(writeto, self, **kwargs)

    # ------------------------ Calculation routines -------------------------
    @property
    def Dpix(self):
        """The distance of each pixel from the beam center in pixel units"""
        self.check_for_q()
        col, row = np.meshgrid(
            np.arange(self.shape[1]), np.arange(self.shape[0]))
        return np.sqrt((col - self.header['BeamPosY']) ** 2 + (row - self.header['BeamPosX']) ** 2)

    @property
    def D(self):
        """The distance of each pixel from the beam center in length units (mm)."""
        self.check_for_q()
        col, row = np.meshgrid(
            np.arange(self.shape[1]), np.arange(self.shape[0]))
        return np.sqrt(((col - self.header['BeamPosY']) * self.header['YPixel']) ** 2 +
                       ((row - self.header['BeamPosX']) * self.header['XPixel']) ** 2)

    @property
    def q(self):
        """The magnitude of the momentum transfer (q=4*pi*sin(theta)/lambda)
        for each pixel. Units are defined by the dimension of WavelengthCalibrated."""
        self.check_for_q()
        return 4 * np.pi * np.sin(0.5 *
                                  np.arctan(self.D /
                                            self.header['DistCalibrated'])) / \
            self.header['WavelengthCalibrated']

    @property
    def dq(self):
        """Error of the magnitude of the momentum transfer (q=4*pi*sin(theta)/lambda)
        for each pixel. Units are defined by the dimension of WavelengthCalibrated.
        Currently only the uncertainty in the sample-to-detector distance is taken
        into account."""
        self.check_for_q()
        D = self.D
        return (2 * np.pi * D * np.cos(np.arctan(D / self.header['DistCalibrated']) / 2.)) / (
            self.header['WavelengthCalibrated'] * (D ** 2 + self.header['DistCalibrated'] ** 2))

    @property
    def tth(self):
        """Two-theta matrix"""
        return np.arctan(self.D / self.header['DistCalibrated'])

    @property
    def dtth(self):
        """Error of the two-theta matrix. Currently only the error in the sample-to-detector
        distance is taken into account."""
        D = self.D
        return np.abs(D / (self.header['DistCalibrated'] ** 2 + D ** 2) * self.header['DistCalibratedError'])

    # ------------------------ Simple arithmetics ---------------------------

    def __str__(self):
        return str(self.header)

    def __unicode__(self):
        return str(self.header)
