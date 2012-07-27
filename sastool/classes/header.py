'''
Created on Jun 15, 2012

@author: andris
'''

# pylint: disable=E0611

import collections
import numpy as np
import itertools
import os
import datetime
import math
import numbers
import re
import h5py

from ..io import header
from .. import misc


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
    # Testing equivalence
    _equiv_tests = {'Dist': lambda a, b: (abs(a - b) < 1),
                    'Energy': lambda a, b: (abs(a - b) < 1),
                    'Temperature': lambda a, b: (abs(a - b) < 0.5),
                    'Title': lambda a, b: (a == b),
                   }
    # information on HDF5 reading: not all Python datatypes have their HDF5
    # equivalents. These help to convert them to/from HDF5.
    # depending on the type: list of (type, converter_function) tuples
    _HDF5_read_postprocess_type = [(np.generic, lambda x:x.tolist()), ]
    # depending on the key name: dictionary of 'key':converter_function pairs
    _HDF5_read_postprocess_name = {'FSNs':lambda x:x.tolist(), 'History':header._delinearize_history}
    # dictionary of key aliases. Note that multi-level aliases are not allowed!
    # This is a 
    _key_aliases = None
    _protectedfields_to_copy = ['_protectedfields_to_copy', '_key_aliases',
                              '_HDF5_read_postprocess_type',
                              '_fields_to_sum', '_fields_to_average',
                              '_fields_to_collect', '_equiv_tests']

    # -- Housekeeping methods: __init__, iterators, __missing__ etc. ----------
    @staticmethod
    def _set_default_kwargs_for_readers(kwargs):
        if 'dirs' not in kwargs:
            kwargs['dirs'] = None
        if 'experiment_type' not in kwargs:
            kwargs['experiment_type'] = None
        if 'error_on_not_found' not in kwargs:
            kwargs['error_on_not_found'] = True
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

        if len(args) < 2:
            obj = super(SASHeader, cls).__new__(cls)
            return obj #this will call obj.__init__(*args, **kwargs) implicitly
        elif len(args) == 2:
            fsns = args[1]
            fileformat = args[0]
            if isinstance(fsns, collections.Sequence):
                objlist = []
                for f in fsns:
                    obj = super(SASHeader, cls).__new__(cls)
                    try:
                        obj.__init__(fileformat % f, **kwargs)
                    except IOError as ioerr:
                        if kwargs['error_on_not_found']:
                            raise ioerr
                        else:
                            obj = None
                    #all other exceptions pass through
                    objlist.append(obj)
                return objlist
            else:
                obj = super(SASHeader, cls).__new__(cls)
                return obj #this will call obj.__init__(*args, **kwargs) implicitly
        else:
            raise ValueError('Invalid number of positional arguments!')
    @staticmethod
    def _autoguess_experiment_type(file_or_dict):
        if isinstance(file_or_dict, basestring):
            file_or_dict = os.path.split(file_or_dict)[1].upper()
            if file_or_dict.endswith('.EDF') or \
                file_or_dict.endswith('CCD'):
                return 'read_from_ESRF_ID02'
            elif file_or_dict.startswith('ORG'):
                return 'read_from_B1_org'
            elif file_or_dict.startswith('INTNORM'):
                return 'read_from_B1_int2dnorm'
            elif file_or_dict.endswith('.H5') or \
                file_or_dict.endswith('.HDF5') or \
                file_or_dict.endswith('.HDF'):
                return 'read_from_hdf5'
            elif file_or_dict.endswith('.BDF') or \
                file_or_dict.endswith('.BHF'):
                return 'read_from_BDF'
            elif file_or_dict.startswith('XE') and \
                (file_or_dict.endswith('.DAT') or \
                 file_or_dict.endswith('.32')):
                return 'read_from_PAXE'
        elif isinstance(file_or_dict, h5py.highlevel.Group):
            return 'read_from_HDF5'
        elif isinstance(file_or_dict, dict):
            if '__Origin__' not in file_or_dict:
                raise ValueError('Cannot determine measurement file format from this dict: no \'__Origin__\' field.')
            elif file_or_dict['__Origin__'] == 'B1 original header':
                return 'read_from_B1_org'
            elif file_or_dict['__Origin__'] == 'B1 log':
                return 'read_from_B1_int2dnorm'
            elif file_or_dict['__Origin__'] == 'EDF ID02':
                return 'read_from_ESRF_ID02'
            elif file_or_dict['__Origin__'] == 'PAXE':
                return 'read_from_PAXE'
            elif file_or_dict['__Origin__'] == 'BDFv1':
                return 'read_from_BDF'
            elif file_or_dict['__Origin__'] == 'BDFv2':
                return 'read_from_BDFv2'
            else:
                raise ValueError('Unknown header dictionary')
        else:
            raise ValueError('Unknown measurement file format')

    def __init__(self, *args, **kwargs):
        """This constructor behaves identically to that of the superclass. If
        the first positional argument is a SASHeader, this copies over the
        protected parameters whose names are found in _protectedfields_to_copy.
        """
        self._key_aliases = {}
        kwargs = SASHeader._set_default_kwargs_for_readers(kwargs)
        if len(args) == 1:
            # expect a single argument, an instance of `dict`. Copy over all
            # its keys

            if isinstance(args[0], SASHeader):
                super(SASHeader,self).__init__(args[0])
                # copy over protected attributes
                for fn in args[0]._protectedfields_to_copy: #IGNORE:W0212
                    attr = getattr(args[0], fn)
                    if hasattr(attr, 'copy'):
                        # if the attribute has a copy method, use that. E.g. dicts.
                        setattr(self, fn, attr.copy())
                    elif isinstance(attr, collections.Sequence):
                        # if the attribute is a sequence, use the [:] construct.
                        setattr(self, fn, attr[:])
                    else:
                        #call the constructor to copy. Note that this can raise an
                        # exception, which is forwarded to the upper level.
                        setattr(self, fn, attr.type(attr))
            elif isinstance(args[0], basestring) or isinstance(args[0], dict):
                # we have to call a read_from_*() method
                if kwargs['experiment_type'] is None:
                    #auto-guess from filename
                    try:
                        loadername = SASHeader._autoguess_experiment_type(args[0])
                    except ValueError:
                        #no special dict, just copy the data and set __Origin__
                        # to 'unknown'
                        super(SASHeader, self).__init__(args[0])
                        self['__Origin__'] = 'unknown'
                        return
                else:
                    loadername = 'read_from_%s' % kwargs['experiment_type']
                try:
                    getattr(self, loadername).__call__(args[0], **kwargs)
                except AttributeError as ae:
                    raise AttributeError(str(ae) + '; possibly bad experiment type given')
                #other exceptions such as IOError on read failure are propagated.
        elif len(args) == 2:
            # file format and fsn is given, fsn should be a scalar number
            fileformat = args[0]
            fsn = args[1]
            if not isinstance(args[1], numbers.Number):
                raise ValueError('Invalid fsn: should be a scalar number')
            if kwargs['experiment_type'] is None:
                #auto-guess from filename
                loadername = SASHeader._autoguess_experiment_type(fileformat % fsn)
            else:
                loadername = 'read_from_%s' % kwargs['experiment_type']
            try:
                getattr(self, loadername).__call__(fileformat % fsn, **kwargs)
            except AttributeError as ae:
                raise AttributeError(str(ae) + '; possibly bad experiment type given')
            #other exceptions such as IOError on read failure are propagated.
    def copy(self, *args, **kwargs):
        """Make a copy of this header structure"""
        d = super(SASHeader, self).copy(*args, **kwargs)
        return SASHeader(d)
    def __missing__(self, key, dry_run = False):
        """Create default values for missing fields"""
        if key in ['FSNs']:
            val = []
        elif key.endswith('Error'):
            val = 0
        elif key.startswith('Monitor'):
            val = 0
        elif key in ['maskid']:
            val = None
        elif key.startswith('FSN'):
            val = 0
        elif key == 'Title':
            val = '<untitled>'
        elif key.endswith('Calibrated'):
            val = self[key[:-len('Calibrated')]]
        #elif key in ['Dist', 'Energy', 'BeamPosX', 'BeamPosY', 'PixelSize']:
        #    val = np.NAN
        elif key in ['XPixel', 'YPixel']:
            val = self['PixelSize']
        else:
            raise KeyError(key)
        if not dry_run:
            super(SASHeader, self).__setitem__(key, val)
        return val
    def __unicode__(self):
        """Print a short summary of this header"""
        return "FSN %s; %s; %s mm; %s eV" % (self['FSN'], self['Title'], self['Dist'], self['Energy'])
    __str__ = __unicode__
    def __repr__(self):
        return "<SASHeader: " + unicode(self) + '>'
    def __getitem__(self, key):
        """ respond to header[key] requests, implements key aliasing."""
        if key in self._key_aliases:
            return super(SASHeader, self).__getitem__(self._key_aliases[key])
        else:
            return super(SASHeader, self).__getitem__(key)
    def __setitem__(self, key, value):
        """ respond to header[key]=value requests, implements key aliasing."""
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
    def __contains__(self, key):
        """ respond to 'key' in header requests, implements key aliasing."""
        if key in self._key_aliases:
            return super(SASHeader, self).__contains__(self._key_aliases[key])
        else:
            ret = super(SASHeader, self).__contains__(key)
            if not ret: # try if the key can be auto-generated by __missing__()
                try:
                    self.__missing__(key, dry_run = True)
                except KeyError:
                    return False
                return True
            else:
                return True
    def __iter__(self):
        """ Return an iterator. This is used e.g. in for k in header constructs.
        """
        return self.iterkeys()
    def keys(self):
        """ Return the keys present. Alias names are also listed."""
        return [k for k in self.iterkeys()]
    def values(self):
        """ Return values. Aliases are listed more than one times."""
        return [v for v in self.itervalues()]
    def items(self):
        """ Return (key,value) pairs. Aliases are listed more than one times.
        """
        return [i for i in self.iteritems()]
    def iterkeys(self):
        """ Iterator version of keys()."""
        return itertools.chain(super(SASHeader, self).iterkeys(), self._key_aliases.iterkeys())
    def itervalues(self):
        """ Iterator version of values()."""
        return itertools.chain(super(SASHeader, self).itervalues(),
                 itertools.imap(lambda x:self[self._key_aliases[x]], self._key_aliases.iterkeys()))
    def iteritems(self):
        """ Iterator version of items()."""
        return itertools.izip(self.iterkeys(), self.itervalues())

    # -------------------- Reader methods (read_from*)

    def read_from_PAXE(self, filename_or_paxe, **kwargs):
        """Read header data from a PAXE (Saclay, France or Budapest, Hungary)
        measurement file.
        
        Inputs:
            filename_or_paxe: the file name (usually XE????.DAT) or a dict
                loaded by readPAXE().
                
        Outputs: the updated header structure. Fields not present in the file
            are kept unchanged.
        """
        kwargs = SASHeader._set_default_kwargs_for_readers(kwargs)
        if isinstance(filename_or_paxe, basestring):
            paxe = header.readPAXE(misc.findfileindirs(filename_or_paxe, kwargs['dirs']))
        else:
            paxe = filename_or_paxe

        self.update(paxe)
        if isinstance(filename_or_paxe, basestring):
            self.add_history('Loaded from PAXE file ' + filename_or_paxe)
        return self

    def read_from_ESRF_ID02(self, filename_or_edf, **kwargs):
        """Read header data from an ESRF ID02 EDF file.
        
        Inputs:
            filename_or_edf: the full name of the file or an edf structure read
                by readehf()
                
        Outputs: the updated header structure. Fields not present in the file
            are kept unchanged.
        """
        kwargs = SASHeader._set_default_kwargs_for_readers(kwargs)
        if isinstance(filename_or_edf, basestring):
            filename_or_edf = header.readehf(misc.findfileindirs(filename_or_edf, kwargs['dirs']))
        self.update(filename_or_edf)
        self._key_aliases['FSN'] = 'HMRunNumber'
        self._key_aliases['BeamPosX'] = 'Center_2'
        self._key_aliases['BeamPosY'] = 'Center_1'
        self._key_aliases['MeasTime'] = 'ExposureTime'
        self._key_aliases['Monitor'] = 'Intensity0'
        self._key_aliases['Detector'] = 'DetectorInfo'
        self._key_aliases['Date'] = 'HMStartTime'
        self._key_aliases['Wavelength'] = 'WaveLength'
        self['Transm'] = self['Intensity1'] / self['Intensity0']
        self['Energy'] = 12398.419 / (self['WaveLength'] * 1e10)
        self['Dist'] = self['SampleDistance'] * 1000
        self['XPixel'] = (self['PSize_1'] * 1000)
        self['YPixel'] = (self['PSize_2'] * 1000)
        self['PixelSize'] = 0.5 * (self['XPixel'] + self['YPixel'])
        self['Title'] = self['TitleBody']
        self['maskid'] = os.path.splitext(self['MaskFileName'])[0]
        for k in sorted([k for k in self if k.startswith('History')]):
            self.add_history(self[k], self['HMStartTime'])
        self.add_history('Loaded EDF header from file ' + filename_or_edf['FileName'])
        return self

    def read_from_B1_org(self, filename, **kwargs):
        kwargs = SASHeader._set_default_kwargs_for_readers(kwargs)
        if isinstance(filename, basestring):
            hed = header.readB1header(misc.findfileindirs(filename, kwargs['dirs']))
        else:
            hed = filename
        self.update(hed)
        if isinstance(filename, basestring):
            self.add_history('Original header loaded: ' + filename)
        return self

    def read_from_B1_int2dnorm(self, filename, **kwargs):
        kwargs = SASHeader._set_default_kwargs_for_readers(kwargs)
        if isinstance(filename, basestring):
            hed = header.readB1logfile(misc.findfileindirs(filename, kwargs['dirs']))
        else:
            hed = filename
        self.update(hed)
        if isinstance(filename, basestring):
            self.add_history('B1 logfile loaded: ' + filename)

    def read_from_BDF(self, filename, **kwargs):
        kwargs = SASHeader._set_default_kwargs_for_readers(kwargs)
        if isinstance(filename, basestring):
            hed = misc.flatten_hierarchical_dict(header.readbhf(misc.findfileindirs(filename, kwargs['dirs'])))
        else:
            hed = misc.flatten_hierarchical_dict(filename)
        self.update(hed)
        if self['C.bdfVersion'] < 2:
            self['BeamPosX'] = self['C.xcen'] - 1
            self['BeamPosY'] = self['C.ycen'] - 1
            for h in self['his']:
                self.add_history('BDF: ' + h)
        elif self['C.bdfVersion'] >= 2:
            if 'CORR.CenterX' in self:
                self['BeamPosX'] = self['CORR.CenterX'] - 1
            if 'CORR.CenterY' in self:
                self['BeamPosY'] = self['CORR.CenterY'] - 1
            if 'CORR.EnergyReal' in self:
                self._key_aliases['EnergyCalibrated'] = 'CORR.EnergyReal'
            if 'CORR.PixelSizeX' in self:
                self['XPixel'] = self['CORR.PixelSizeX'] * 10
            if 'CORR.PixelSizeY' in self:
                self['YPixel'] = self['CORR.PixelSizeY'] * 10
            for h in self['HIS']:
                self.add_history('BDF: ' + h)
            if 'CORR.SampleThickness' in self:
                self._key_aliases['Thickness'] = 'CORR.SampleThickness'
            if 'CORR.SampleThicknessError' in self:
                self._key_aliases['ThicknessError'] = 'CORR.SampleThicknessError'
        #common to BDFv1 and v2
        self.add_history('History imported from BDF file')
        self._key_aliases['Energy'] = 'M.Energy'
        self._key_aliases['Dist'] = 'M.SD'
        self._key_aliases['Title'] = 'C.Sample'
        self._key_aliases['Temperature'] = 'C.isTemp'
        self._key_aliases['MeasTime'] = 'CS.Seconds'
        self._key_aliases['Monitor'] = 'CS.Monitor'
        self._key_aliases['Anode'] = 'CS.Anode'
        self._key_aliases['PosSample'] = 'M.VacSampleX'
        self._key_aliases['PosRef'] = 'M.RefSampleX'
        self._key_aliases['Transm'] = 'CT.trans'
        self._key_aliases['TransmError'] = 'CT.transerr'
        try:
            self['FSN'] = int(re.search('\d+', self['C.Frame']).group())
        except AttributeError:
            self['FSN'] = self['C.Frame']
        return self
    read_from_BDFv2 = read_from_BDF
    read_from_BDFv1 = read_from_BDF

    def write_B1_log(self, filename):
        header.writeB1logfile(filename, self)
    # ------------------------ History manipulation ---------------------------

    def add_history(self, text, time = None):
        """Add a new entry to the history.
        
        Inputs:
            text: history text
            time: time of the event. If None, the current time will be used.
        """
        if time is None:
            time = datetime.datetime.now()
        if 'History' not in self:
            self['History'] = []
        deltat = time - datetime.datetime.fromtimestamp(0, time.tzinfo)
        deltat_seconds = deltat.seconds + deltat.days * 24 * 3600 + deltat.microseconds * 1e-6
        self['History'].append((deltat_seconds, text))

    def get_history(self):
        """Return the history in a human-readable format"""
        return '\n'.join([str(h[0]) + ': ' + h[1] for h in self['History']])


# --------------------- Summarizing, averaging and equivalence -------------

    def __iadd__(self, other):
        """Add in-place. The actual work is done by the SASHeader.summarize()
        classmethod."""
        obj = SASHeader.summarize(self, other)
        for k in obj.keys():
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
        allfieldnames = set(reduce(lambda (a, b):a + b, [a.keys() for a in args], []))
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
        for k in allfieldnames.intersection(cls._fields_to_collect.keys()):
            k_c = cls._fields_to_collect[k]
            obj[k_c] = sum([a[k_c] for a in args])
            obj[k_c].extend([a[k] for a in args])
            obj[k_c] = list(set(obj[k_c]))
            obj[k] = obj[k_c][0]  # take the first one.
            fields_treated.extend([k, k_c])
        allfieldnames = allfieldnames.difference(fields_treated)
        for k in allfieldnames:
            # find the first occurrence of the field
            obj[k] = [a for a in args if k in a.keys()][0][k]
        obj.add_history('Summed from: ' + ' and '.join([unicode(a) for a in args]))
        return obj

    def isequiv(self, other):
        """Test if the two headers are equivalent. The information found in
        SASHeader._equiv_tests is used to decide equivalence.
        """
        return all([self._equiv_tests[k](self[k], other[k]) for k in self._equiv_tests] +
                   [other._equiv_tests[k](self[k], other[k]) for k in other._equiv_tests]) #IGNORE:W0212

    # ---------------------------- HDF5 I/O ----------------------------------

    def write_to_hdf5(self, hdf_entity):
        """Write the parameter structure to a HDF entity (group or dataset) as
        attributes. hdf_entity should be an instance of h5py.highlevel.Dataset
        or h5py.highlevel.Group or h5py.highlevel.File."""
        try:
            self.add_history('Written to HDF:' + hdf_entity.file.filename + hdf_entity.name)
            for k in self.keys():
                if k == 'History':
                    hdf_entity.attrs[k] = header._linearize_history(self[k]).encode('utf-8')
                elif isinstance(self[k], bool):
                    hdf_entity.attrs[k] = int(self[k])
                elif isinstance(self[k], numbers.Number):
                    hdf_entity.attrs[k] = self[k]
                elif isinstance(self[k], basestring):
                    hdf_entity.attrs[k] = self[k].encode('utf-8')
                elif isinstance(self[k], collections.Sequence):
                    hdf_entity.attrs[k] = self[k]
                elif isinstance(self[k], datetime.datetime):
                    hdf_entity.attrs[k] = str(self[k])
                else:
                    raise ValueError('Invalid field type: ' + str(k) + ', ', repr(type(self[k])))
        finally:
            del self['History'][-1]

    def read_from_hdf5(self, hdf_entity):
        """Read the parameter structure from the attributes of a HDF entity 
        (group or dataset). hdf_entity should be an instance of
        h5py.highlevel.Dataset or h5py.highlevel.Group or h5py.highlevel.File.
        """
        for k in hdf_entity.attrs.keys():
            attr = hdf_entity.attrs[k]
            if k in self._HDF5_read_postprocess_name:
                self[k] = self._HDF5_read_postprocess_name[k](attr)
            else:
                typematch = [x for x in self._HDF5_read_postprocess_type if isinstance(attr, x[0]) ]
                if typematch:
                    self[k] = typematch[0][1](attr)
                else:
                    self[k] = attr
        self.add_history('Header read from HDF:' + hdf_entity.file.filename + hdf_entity.name)
