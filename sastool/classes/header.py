'''
Created on Jun 15, 2012

@author: andris
'''

# pylint: disable=E0611

import collections
import itertools
import datetime
import math
import re
import warnings
import copyreg
from .headerfields import SASHeaderField, SASHeaderFieldLink
from .history import SASHistory
from functools import reduce
__all__ = ['SASHeader']
from ..libconfig import HC
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import scipy.constants

NEUTRON_WAVELENGTH_CONVERTOR = scipy.constants.codata.value('Planck constant') ** 2 * 0.5 / \
    (scipy.constants.codata.value('neutron mass')) / \
    scipy.constants.codata.value(
        'electron volt-joule relationship') * 1e20  # J


class SASHeaderException(Exception):
    pass


class SASHeader(dict):
    """A class for holding measurement meta-data, such as sample-detector
    distance, photon energy, wavelength, beam position etc.

    This is a subclass of the Python dictionary class. It supports therefore
    dict-style indexing (header['BeamPosX']), but has several extensions:

    1) The main intent of this class is to offer a uniform interface for common
    header data (beam position, etc.), while keeping the original logic of
    various instruments. To accomplish that, we have key aliasing under the
    hood: the dict _key_aliases can define alias names for each key, e.g. EDF
    files contain the beam position as Center_1 and Center_2. Aliasing ensures
    that these can be accessed under the general fieldnames BeamPosX and
    BeamPosY. Either is read or modified, the underlying field gets read or
    modified.

    2) safe default values are supplied for undefined fields, e.g. default FSN
    is zero, default Title is "untitled" etc. See __missing__() for details

    3) summing and averaging of multiple headers is supported, compatibility is
    taken care of. See fields _fields_to_sum, _fields_to_average,
    _fields_to_collect and __add__, __iadd__, isequiv, summarize() methods.

    4) Loading/saving: if you want to load a header file, use the new_from_*()
    class methods. If you want to extend (update) an instance with another file,
    use the read_from_*() instance methods. Saving is supported as generalized
    B1 logfiles (text) and as attributes of a HDF5 group or dataset.

    Reader routines can be called only with a single filename (or corresponding
    data dict loaded elsewhere). New_from_*() methods have two possibilities of
    invocation:

    >>> header = new_from_xyz(filename_or_dict_object)

    or

    >>> header = new_from_xyz(fsn, fileformat=<default_name>, dirs=['.'])

    where fsn is either a scalar number or a sequence of scalar numbers (the
    so-called file sequence number) and fileformat is a C-style format string
    into which the fsn(s) is/are interpolated, e.g. 'org_%05d.header' for B1
    original header files. This argument ALWAYS has a default value. dirs is a
    list of folders to look for the file in. From the first two arguments,
    a full filename will be constructed and fed to sastool.misc.findfileindirs
    along with dirs. If fsn is a scalar, all exceptions raised by that function
    are propagated to the caller. If fsn is a sequence, exceptions are eaten
    and a sequence of only the successfully loaded headers will be returned.

    5) History: operations on the dataset can (and should) update the history.

    Examples:

    Load a B1 original header file:
    >>> h=SASHeader.new_from_B1_org('path/to/datafile/org_00015.header')

    Load an EDF file:
    >>> h=SASHeader.new_from_ESRF_ID02('path/to/datafile/sc3269_0_0015ccd')

    Supported keys (these are obligatory. Part of them have safe default
    values):
    :FSN:     File sequence number (integer)
    :FSNs:    List of file sequence numbers (defaults to an empty list)
    :Title:   Sample name
    :Dist:    Sample-to-detector distance in mm
    :DistCalibrated: Calibrated sample-to-detector distance in mm
    :Thickness:      Sample thickness (cm)
    :ThicknessError: Absolute error of sample thickness (cm)
    :Transm:         Sample transmission (e^{-mu*d})
    :TransmError:    Error of sample transmission
    :PosSample:      Sample position
    :Temperature:    Sample temperature
    :MeasTime:       Measurement time
    :FSNref1:        File sequence number for absolute intensity reference
    :Thicknessref1:  Thickness of the absolute intensity reference (cm)
    :Thicknessref1Error: Error of the thickness of the absolute intensity reference (cm)
    :Energy:         Nominal (apparent) photon energy (in case of neutrons, the
            "equivalent" photon energy h*c/lambda)
    :EnergyCalibrated: The calibrated photon energy
    :BeamPosX:       Position of the beam on the detector: row index (indexing
        starts from 0).
    :BeamPosY:       Position of the beam on the detector: column index
        (indexing starts from 0)
    :BeamsizeX:      Size of the beam at the sample, row direction (mm)
    :BeamsizeY:      Size of the beam at the sample, column direction (mm)
    :XPixel:         Pixel size of the detector, row direction (mm)
    :YPixel:         Pixel size of the detector, column direction (mm)
    :PixelSize:      Average pixel size of the detector: 0.5*(XPixel+YPixel)
    :Monitor:        Monitor counts
    :MonitorError:   Error of the monitor counts

    Some keys are treated specially:

    keys ending with ``Calibrated``
        If this key is not found, it is created with the value of the
        corresponding key without the ``Calibrated`` suffix. E.g. if
        ``EnergyCalibrated`` does not exist, it is created automatically and
        its value will be that of ``Energy``.

    keys ending with ``Error``
        if this key is not present, it is created with 0 value. Upon averaging
        multiple headers, these and the corresponding un-suffixed keys are
        treated specially. Namely, those found in SASHeader._fields_to_sum will
        be added (and the corresponding Error key will be calculated through
        Gaussian error-propagation), those found in SASHeader._fields_to_average
        will be averaged (weighted by the corresponding Error keys). I.e. the
        former are assumed to be *independent measurements of different
        quantities*, the latter are assumed to be *independent measurements of
        the same quantity*.

    """
    _plugins = []
    # the following define fields treated specially when adding one or more
    # headers together.
    # _fields_to_sum: these fields are to be added. The corresponding 'Error'
    # fields are used for error propagation. Field names can be strings or
    # compiled regular expression patterns.
    _fields_to_sum = ['MeasTime', 'Anode', re.compile('Monitor\w*')]
    # _fields_to_average: these fields will be averaged. Error propagation is
    # done.
    _fields_to_average = ['Transm', 'Temperature', 'BeamPosX', 'BeamPosY',
                          'Thickness', 'Thicknessref1']
    # _fields_to_collect: these will be collected.
    _fields_to_collect = {'FSN': 'FSNs'}
    # Testing equivalence. Two SASHeaders are deemed to be equivalent (i.e. different
    # exposures of the same sample under the same conditions are equivalent), if all
    # criteria are fulfilled. Criteria can be given as tuples:
    #  ('fieldname', tolerance)
    # where tolerance can be any nonnegative real number or None. In the former case
    # equivalence is the value of (abs(header1['fieldname']-header2['fieldname'])<tolerance.
    # In the latter case header1['fieldname']==header2['fieldname'] is checked.
    _equiv_tests = [('Dist', 1),
                    ('Energy', 1),
                    ('Temperature', 0.5),
                    ('Title', None),
                    ]
    # dictionary of key aliases. Note that multi-level aliases are not allowed!
    # This is a
    _key_aliases = None
    _protectedfields_to_copy = ['_protectedfields_to_copy', '_key_aliases',
                                '_fields_to_sum', '_fields_to_average',
                                '_fields_to_collect', '_equiv_tests']
    _needs_init = True
    # -- Housekeeping methods: __init__, iterators, __missing__ etc. ----------

    @staticmethod
    def _set_default_kwargs_for_readers(kwargs):
        if 'dirs' not in kwargs:
            kwargs['dirs'] = None
        if 'experiment_type' not in kwargs:
            kwargs['experiment_type'] = None
        if 'error_on_not_found' not in kwargs:
            kwargs['error_on_not_found'] = True
        if 'generator' not in kwargs:
            kwargs['generator'] = False
        return kwargs

    def __new__(cls, *args, **kwargs):
        """Load one or more header structure. This function serves as a general
        entry point. It handles the calls
        ::

            >>> SASHeader(...)

        The ways of calling are:

        0) ``SASHeader()``: empty constructor

        1) ``SASHeader(<instance-of-SASHeader>)``: copy constructor

        2) ``SASHeader(<instance-of-dict>)``: casting constructor.

        3) ``SASHeader(<filename>, **kwargs)``: direct loading of a file

        4) ``SASHeader(<fileformat>, <fsn>, **kwargs)``: loading possibly more
        files.
        """
        kwargs = SASHeader._set_default_kwargs_for_readers(kwargs)
        if not args:
            return super(SASHeader, cls).__new__(cls)
        elif (isinstance(args[0], SASHeader) or isinstance(args[0], dict)) or isinstance(args[0], tuple):
            return super(SASHeader, cls).__new__(cls)
        else:
            # everything from now on is handled by plug-ins.
            # Everything else is handled by IO plugins
            plugin = cls.get_IOplugin(args[0], 'READ', **kwargs)
            logger.debug(
                'Using header plugin %s for loading file %s.' % (plugin.name, args[0]))
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
            if isinstance(res, tuple):
                obj = super(SASHeader, cls).__new__(cls)
                obj.__init__(res)
                return obj
            else:
                gen = cls._read_multi(res)
                if len(args) == 2 and not isinstance(args[1], collections.Sequence):
                    return list(gen)[0]
                elif kwargs['generator']:
                    return gen
                else:
                    return list(gen)

    @classmethod
    def _read_multi(cls, lis):
        for l in lis:
            yield cls(l)
        return

    @classmethod
    def _autoguess_experiment_type(cls, file_or_dict):
        plugin = [p for p in cls._plugins if p.check_if_applies(file_or_dict)]
        if not plugin:
            raise SASHeaderException(
                'No plugin can handle ' + str(file_or_dict))
        return plugin[0]

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

    def __init__(self, *args, **kwargs):
        """This constructor behaves identically to that of the superclass. If
        the first positional argument is a SASHeader, this copies over the
        protected parameters whose names are found in _protectedfields_to_copy.
        """
        if hasattr(self, '_was_init'):
            return
        self._was_init = True
        self._key_aliases = {}
        kwargs = SASHeader._set_default_kwargs_for_readers(kwargs)
        if not args:
            super(SASHeader, self).__init__()
        elif isinstance(args[0], SASHeader):
            super(SASHeader, self).__init__(args[0])
            # copy over protected attributes
            for fn in args[0]._protectedfields_to_copy:  # IGNORE:W0212
                attr = getattr(args[0], fn)
                if hasattr(attr, 'copy'):
                    # if the attribute has a copy method, use that. E.g. dicts.
                    setattr(self, fn, attr.copy())
                elif isinstance(attr, collections.Sequence):
                    # if the attribute is a sequence, use the [:] construct.
                    setattr(self, fn, attr[:])
                else:
                    # call the constructor to copy. Note that this can raise an
                    # exception, which is forwarded to the upper level.
                    setattr(self, fn, attr.type(attr))
        elif isinstance(args[0], tuple):
            if args[0] and args[0][0] == 'PICKLED_SASHEADER':
                self.update(args[0][1]['data'])
                for f in list(args[0][1].keys()):
                    if f == 'data':
                        continue
                    self.__setattr__(f, args[0][1][f])
            else:
                self.update(args[0][0])
                self._key_aliases = args[0][1]
        else:
            # search for a plugin to handle this
            try:
                plugin = self.get_IOplugin(args[0], 'READ', **kwargs)
            except ValueError:
                raise NotImplementedError(
                    '__init__() not supported with args[0]==' + str(args[0]))
            else:
                d, ka = plugin.read(args[0], **kwargs)
                del self._was_init
                self.__init__((d, ka))

    def copy(self, *args, **kwargs):
        """Make a copy of this header structure"""
        d = super(SASHeader, self).copy(*args, **kwargs)
        return SASHeader(d)

    def __missing__(self, key, dry_run=False):
        """Create default values for missing fields"""
        if key in ['FSNs']:
            val = []
        elif key == 'ErrorFlags':
            val = ''
        elif key.endswith('Error'):
            if 'Calibrated' in key:
                val = self[key.replace('Calibrated', '')]
            else:
                val = 0
        elif key.startswith('Monitor'):
            val = 1
        elif key == '__particle__':
            val = 'photon'
        elif key == 'Energy':
            if not self.__contains__('Wavelength', False):
                raise KeyError(key)
            if self['__particle__'] == 'photon':
                val = HC() / self['Wavelength']
            elif self['__particle__'] == 'neutron':
                val = NEUTRON_WAVELENGTH_CONVERTOR / self['Wavelength'] ** 2
            else:
                raise ValueError(
                    'Invalid particle type: ' + self['__particle__'])
        elif key == 'Wavelength':
            if not self.__contains__('Energy', False):
                raise KeyError(key)
            if self['__particle__'] == 'photon':
                val = HC() / self['Energy']
            elif self['__particle__'] == 'neutron':
                val = (NEUTRON_WAVELENGTH_CONVERTOR / self['Energy']) ** 0.5
            else:
                raise ValueError(
                    'Invalid particle type: ' + self['__particle__'])
        elif key in ['maskid']:
            val = None
        elif key.startswith('FSN'):
            val = 0
        elif key == 'Title':
            val = '<untitled>'
        elif 'Calibrated' in key:
            val = self[key.replace('Calibrated', '')]
        # elif key in ['Dist', 'Energy', 'BeamPosX', 'BeamPosY', 'PixelSize']:
        #    val = np.NAN
        elif key in ['XPixel', 'YPixel']:
            val = self['PixelSize']
        else:
            raise KeyError(key)
        if not dry_run:
            super(SASHeader, self).__setitem__(key, val)
        return val

    def __str__(self, *args):
        """Print a short summary of this header"""
        if 'FSN' in self:
            fsn = self['FSN']
        else:
            fsn = '<no FSN>'
        if 'Title' in self:
            title = self['Title']
        else:
            title = '<no title>'
        if 'Dist' in self:
            dist = '%.2f mm' % self['Dist']
        else:
            dist = '<no dist>'
        if 'Energy' in self:
            energy = '%.2f eV' % self['Energy']
        else:
            energy = '<no energy>'
        if 'MeasTime' in self:
            meastime = '%.3f s' % self['MeasTime']
        else:
            meastime = '<no exptime>'
        return "FSN %s; %s; %s; %s; %s" % (fsn, title, dist, energy, meastime)

    def __repr__(self):
        return "<SASHeader: " + str(self) + '>'

    def __getitem__(self, key):
        """ respond to header[key] requests, implements key aliasing."""
        if key in self._key_aliases:
            return super(SASHeader, self).__getitem__(self._key_aliases[key])
        else:
            return super(SASHeader, self).__getitem__(key)

    def __setitem__(self, key, value, notricks=False):
        """ respond to header[key]=value requests, implements key aliasing."""
        if key.startswith('Energy') and not notricks:
            # set the wavelength as well.
            if self.__getitem__('__particle__') == 'photon':
                self.__setitem__(
                    key.replace('Energy', 'Wavelength'), HC() / value, notricks=True)
            elif self.__getitem__('__particle__') == 'neutron':
                self.__setitem__(key.replace(
                    'Energy', 'Wavelength'), (NEUTRON_WAVELENGTH_CONVERTOR / value) ** 0.5, notricks=True)
            else:
                warnings.warn('Particle type not defined in header')
            self.__setitem__(key, value, notricks=True)
        elif key.startswith('Wavelength') and not notricks:
            if self.__getitem__('__particle__') == 'photon':
                self.__setitem__(
                    key.replace('Wavelength', 'Energy'), HC() / value, notricks=True)
            elif self.__getitem__('__particle__') == 'neutron':
                self.__setitem__(key.replace(
                    'Wavelength', 'Energy'), NEUTRON_WAVELENGTH_CONVERTOR / value ** 2, notricks=True)
            else:
                warnings.warn('Particle type not defined in header')
            self.__setitem__(key, value, notricks=True)
        if key in self._key_aliases:
            return self.__setitem__(self._key_aliases[key], value)
        else:
            return super(SASHeader, self).__setitem__(key, value)

    def __delitem__(self, key):
        """ respond to del header[key] requests, implements key aliasing."""
        if key in self:
            return super(SASHeader, self).__delitem__(key)
        elif key in self._key_aliases:
            return self.__delitem__(self._key_aliases[key])
        else:
            raise KeyError(key)

    def __contains__(self, key, generate_missing=True):
        """ respond to 'key' in header requests, implements key aliasing."""
        if key in self._key_aliases:
            return super(SASHeader, self).__contains__(self._key_aliases[key])
        else:
            ret = super(SASHeader, self).__contains__(key)
            # try if the key can be auto-generated by __missing__()
            if not ret and generate_missing:
                try:
                    self.__missing__(key, dry_run=True)
                except KeyError:
                    return False
                return True
            else:
                return ret

    def __iter__(self):
        """ Return an iterator. This is used e.g. in for k in header constructs.
        """
        return iter(self.keys())

    def keys(self):
        """ Iterator version of keys()."""
        return itertools.chain(super(SASHeader, self).keys(), filter(lambda x: x in self, self._key_aliases.keys()))

    def values(self):
        """ Iterator version of values()."""
        return itertools.chain(super(SASHeader, self).values(),
                               map(lambda x: self[self._key_aliases[x]], filter(lambda x: x in self, self._key_aliases.keys())))

    def items(self):
        """ Iterator version of items()."""
        return zip(self.keys(), self.values())

    @classmethod
    def register_IOplugin(cls, plugin, idx=None):
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
        if 'plugin' in kwargs:
            kwargs['experiment_type'] = kwargs['plugin']
        if 'experiment_type' in kwargs:
            plugin = [p for p in cls._plugins if p.name ==
                      kwargs['experiment_type'] and checkmode(p)]
        if not plugin:
            plugin = [
                p for p in cls._plugins if p.check_if_applies(filename) and checkmode(p)]
        if not plugin:
            raise ValueError('No plugin can handle ' + str(filename))
        return plugin[0]

    def write(self, writeto, **kwargs):
        plugin = self.get_IOplugin(writeto, 'WRITE', **kwargs)
        plugin.write(writeto, self, **kwargs)

    # ------------------------ History manipulation ---------------------------

    def add_history(self, text, time=None):
        """Add a new entry to the history.

        Inputs:
            text: history text
            time: time of the event. If None, the current time will be used.
        """
        if 'History' not in self:
            self['History'] = SASHistory()
        self['History'].add(text, time)

    def get_history(self):
        """Return the history in a human-readable format"""
        return str(self['History'])


# --------------------- Summarizing, averaging and equivalence -------------

    def __iadd__(self, other):
        """Add in-place. The actual work is done by the SASHeader.summarize()
        classmethod."""
        obj = SASHeader.summarize(self, other)
        for k in list(obj.keys()):
            self[k] = obj[k]
        return self

    def __add__(self, other):
        """Add two headers. The actual work is done by the SASHeader.summarize()
        classmethod."""
        return SASHeader.summarize(self, other)

    @classmethod
    def summarize(cls, *args):
        """Summarize several headers. Calling convention:

        summed=SASHeader.summarize(header1,header2,header3,...)

        Several fields are treated specially. Values of fields in
        SASHeader._fields_to_sum are summed (error is calculated by taking the
        corresponding 'Error' fields into account). Values of fields in
        SASHeader._fields_to_average get averaged. Values of fields in
        SASHeader._fields_to_collect get collected to a corresponding list field.
        """
        if any([not isinstance(a, SASHeader) for a in args]):
            raise NotImplementedError('Only SASHeaders can be averaged!')
        obj = cls()
        fields_treated = []
        allfieldnames = set(
            reduce(lambda a_b: a_b[0] + a_b[1], [list(a.keys()) for a in args], []))
        for k in allfieldnames.intersection(cls._fields_to_sum):
            dk = k + 'Error'
            obj[k] = sum([a[k] for a in args])
            obj[dk] = math.sqrt(sum([a[dk] ** 2 for a in args]))
            fields_treated.extend([k, dk])
        allfieldnames = allfieldnames.difference(fields_treated)
        for k in allfieldnames.intersection(cls._fields_to_average):
            dk = k + 'Error'
            sumweight = sum([1 / a[dk] ** 2 for a in args])
            obj[k] = sum([a[k] / a[dk] ** 2 for a in args]) / sumweight
            obj[dk] = 1 / math.sqrt(sumweight)
            fields_treated.extend([k, dk])
        allfieldnames = allfieldnames.difference(fields_treated)
        for k in allfieldnames.intersection(list(cls._fields_to_collect.keys())):
            k_c = cls._fields_to_collect[k]
            obj[k_c] = sum([a[k_c] for a in args])
            obj[k_c].extend([a[k] for a in args])
            obj[k_c] = list(set(obj[k_c]))
            obj[k] = obj[k_c][0]  # take the first one.
            fields_treated.extend([k, k_c])
        allfieldnames = allfieldnames.difference(fields_treated)
        for k in allfieldnames:
            # find the first occurrence of the field
            obj[k] = [a for a in args if k in list(a.keys())][0][k]
        obj.add_history('Summed from: ' + ' and '.join([str(a) for a in args]))
        return obj

    def isequiv(self, other):
        """Test if the two headers are equivalent. The information found in
        SASHeader._equiv_tests is used to decide equivalence.
        """
        return all([self._equiv_tests[k](self[k], other[k]) for k in self._equiv_tests] +
                   [other._equiv_tests[k](self[k], other[k]) for k in other._equiv_tests])  # IGNORE:W0212

    # ---------------------------- HDF5 I/O ----------------------------------

    def __reduce__(self):
        d = {'data': dict(self)}
        d['_protectedfields_to_copy'] = self._protectedfields_to_copy
        for f in self._protectedfields_to_copy:
            d[f] = self.__getattribute__(f)
        return (SASHeader, (('PICKLED_SASHEADER', d),))
