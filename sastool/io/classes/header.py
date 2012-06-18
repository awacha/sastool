'''
Created on Jun 15, 2012

@author: andris
'''

import functools
import collections
import numpy as np
import itertools
import os
import datetime
import math
import numbers

from .. import header
from ... import misc

def _linearize_bool(b):
    return ['n', 'y'][bool(b)]


def _delinearize_bool(b):
    if isinstance(b, basestring):
        b = b.lower()
        if b.startswith('y') or b == 'true':
            return True
        elif b.startswith('n') or b == 'false':
            return False
        else:
            raise ValueError(b)
    else:
        return bool(b)


def _linearize_list(l, pre_converter = lambda a:a, post_converter = lambda a:a):
    return post_converter(' '.join([unicode(pre_converter(x)) for x in l]))


def _delinearize_list(l, pre_converter = lambda a:a, post_converter = list):
    return post_converter([misc.parse_number(x) for x in \
                           pre_converter(l).replace(',', ' ').replace(';', ' ').split()])


def _linearize_history(history):
    history_text = [str(x[0]) + ': ' + x[1] for x in history]
    history_text = [a.replace(';', ';;') for a in history_text]
    return '; '.join(history_text)

def _delinearize_history(history_oneliner):
    history_oneliner = history_oneliner.replace(';;',
                                                '<*doublesemicolon*>')
    history_list = [a.strip().replace('<*doublesemicolon*>', ';') \
                    for a in history_oneliner.split(';')]
    history = [a.split(':', 1) for a in history_list]
    history = [(float(a[0]), a[1].strip()) for a in history]
    return history

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
    
    """
    # the following define fields treated specially when adding one or more
    # headers together.
    # _fields_to_sum: these fields are to be added. The corresponding 'Error'
    # fields are used for error propagation
    _fields_to_sum = ['MeasTime', 'Anode', 'Monitor', 'MonitorDORIS',
                      'MonitorPIEZO']
    # _fields_to_average: these fields will be averaged. Error propagation is
    # done.
    _fields_to_average = ['Transm', 'Temperature', 'BeamPosX', 'BeamPosY']
    # _fields_to_collect: these will be collected.
    _fields_to_collect = {'FSN': 'FSNs'}
    # Testing equivalence
    _equiv_tests = {'Dist': lambda a, b: (abs(a - b) < 1),
                    'Energy': lambda a, b: (abs(a - b) < 1),
                    'Temperature': lambda a, b: (abs(a - b) < 0.5),
                    'Title': lambda a, b: (a == b),
                   }

    # information on how to store the param structure. Each sub-list
    # corresponds to a line in the param structure and should be of the form
    # [<linestart>,<field name(s)>,<formatter function>,<reader function>]
    #
    # Required are the first and second.
    #
    # linestart: the beginning of the line in the file, up to the colon.
    #
    # field name(s): field name can be a string or a tuple of strings.
    #
    # formatter function: can be (1) a function accepting a single argument
    #     (the value of the field) or (2) a tuple of functions or (3) None. In
    #     the latter case and when omitted, unicode() will be used.
    #
    # reader function: can be (1) a function accepting a string and returning
    #     as many values as the number of field names is. Or if omitted,
    #     unicode() will be used.
    #
    _logfile_data = [('FSN', 'FSN', None, int),
                     ('FSNs', 'FSNs', _linearize_list, _delinearize_list),
                     ('Sample name', 'Title'),
                     ('Sample title', 'Title'),
                     ('Sample-to-detector distance (mm)', 'Dist', None, float),
                     ('Sample thickness (cm)', 'Thickness', None, float),
                     ('Sample transmission', 'Transm', None, float),
                     ('Sample position (mm)', 'PosSample', None, float),
                     ('Temperature', 'Temperature', None, float),
                     ('Measurement time (sec)', 'MeasTime', None, float),
                     ('Scattering on 2D detector (photons/sec)',
                      'ScatteringFlux', None, float),
                     ('Dark current subtracted (cps)', 'dclevel', None, float),
                     ('Dark current FSN', 'FSNdc', None, int),
                     ('Empty beam FSN', 'FSNempty', None, int),
                     ('Injection between Empty beam and sample measurements?',
                      'InjectionEB', _linearize_bool, _delinearize_bool),
                     ('Glassy carbon FSN', 'FSNref1', None, int),
                     ('Glassy carbon thickness (cm)', 'Thicknessref1', None,
                      float),
                     ('Injection between Glassy carbon and sample measurements?',
                      'InjectionGC', _linearize_bool, _delinearize_bool),
                     ('Energy (eV)', 'Energy', None, float),
                     ('Calibrated energy (eV)', 'EnergyCalibrated', None, float),
                     ('Calibrated energy', 'EnergyCalibrated', None, float),
                     ('Beam x y for integration', ('BeamPosX', 'BeamPosY'),
                      functools.partial(_linearize_list, pre_converter = lambda a:a + 1),
                      functools.partial(_delinearize_list,
                                        post_converter = lambda a:tuple([x - 1 for x in a]))),
                     ('Normalisation factor (to absolute units)', 'NormFactor',
                      None, float),
                     ('Relative error of normalisation factor (percentage)',
                      'NormFactorRelativeError', None, float),
                     ('Beam size X Y (mm)', ('BeamsizeX', 'BeamsizeY'), _linearize_list,
                      functools.partial(_delinearize_list, post_converter = tuple)),
                     ('Pixel size of 2D detector (mm)', 'PixelSize', None, float),
                     ('Primary intensity at monitor (counts/sec)', 'Monitor', None,
                      float),
                     ('Primary intensity calculated from GC (photons/sec/mm^2)',
                      'PrimaryIntensity', None, float),
                     ('Sample rotation around x axis', 'RotXsample', None, float),
                     ('Sample rotation around y axis', 'RotYsample', None, float),
                     ('History', 'History', _linearize_history, _delinearize_history),
                    ]
    # information on HDF5 reading: not all Python datatypes have their HDF5
    # equivalents. These help to convert them to/from HDF5.
    # depending on the type: list of (type, converter_function) tuples
    _HDF5_read_postprocess_type = [(np.generic, lambda x:x.tolist()), ]
    # depending on the key name: dictionary of 'key':converter_function pairs
    _HDF5_read_postprocess_name = {'FSNs':lambda x:x.tolist(), 'History':_delinearize_history}
    # dictionary of key aliases. Note that multi-level aliases are not allowed!
    # This is a 
    _key_aliases = None
    _protectedfields_to_copy = ['_protectedfields_to_copy', '_key_aliases',
                              '_HDF5_read_postprocess_type', '_logfile_data',
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
                        obj.__init__(fileformat % f, *kwargs)
                    except IOError as ioerr:
                        if kwargs['error_on_not_found']:
                            raise
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
    def _autoguess_experiment_type(filename_or_name):
        if isinstance(fileformat_or_name, basestring):
            fileformat_or_name = os.path.split(fileformat_or_name)[1].upper()
            if fileformat_or_name.endswith('.EDF') or \
                fileformat_or_name.endswith('CCD'):
                return 'read_from_ESRF_ID02'
            elif fileformat_or_name.startswith('ORG'):
                return 'read_from_B1_org'
            elif fileformat_or_name.startswith('INTNORM'):
                return 'read_from_B1_log'
            elif fileformat_or_name.endswith('.H5') or \
                fileformat_or_name.endswith('.HDF5') or \
                fileformat_or_name.endswith('.HDF'):
                return 'read_from_hdf5'
            elif fileformat_or_name.endswith('.BDF') or \
                fileformat_or_name.endswith('.BHF'):
                return 'read_from_BDF'
            elif fileformat_or_name.startswith('XE') and \
                (fileformat_or_name.endswith('.DAT') or \
                 fileformat_or_name.endswith('.32')):
                return 'read_from_PAXE'
        elif isinstance(fileformat_or_name, h5py.highlevel.Group):
            return 'read_from_HDF5'
        elif isinstance(fileformat_or_name, dict):
            #TODO
        else:
            raise ValueError('Unknown measurement file format')

    def __init__(self, *args, **kwargs):
        """This constructor behaves identically to that of the superclass. If
        the first positional argument is a SASHeader, this copies over the
        protected parameters whose names are found in _protectedfields_to_copy.
        """
        self._key_aliases = {}
        if len(args) == 1:
            # expect a single argument, an instance of `dict`. Copy over all
            # its keys
            if isinstance(args[0], dict):
                super(SASHeader, self).__init__(self, *args, **kwargs)
                #if this is an instance of `SASHeader` as well, do some fine-tuning
                if isinstance(args[0], SASHeader):
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
            elif isinstance(args[0], basestring): # we have to open a file
                if kwargs['experiment_type'] is None:
                    #auto-guess from filename
                    loadername = SASHeader._autoguess_experiment_type(args[0])
                else:
                    loadername = 'read_from_%s' % kwargs['experiment_type']
                try:
                    getattr(self, loadername).__call__(filename, **kwargs)
                except AttributeError as ae:
                    raise AttributeError(str(ae) + '; possibly bad experiment type given')
                #other exceptions such as IOError on read failure are propagated.
            # copy over protected attributes

        elif len(args) == 2:
    def copy(self, *args, **kwargs):
        """Make a copy of this header structure"""
        d = super(SASHeader, self).copy(*args, **kwargs)
        return SASHeader(d)
    def __missing__(self, key):
        """Create default values for missing fields"""
        if key in ['FSNs']:
            val = []
        elif key.endswith('Error'):
            val = 0
        elif key in ['maskid']:
            val = None
        elif key.startswith('FSN'):
            val = 0
        elif key == 'Title':
            val = '<untitled>'
        elif key in ['Dist', 'Energy', 'EnergyCalibrated', 'BeamPosX', 'BeamPosY', 'PixelSize']:
            val = np.NAN
        else:
            raise KeyError(key)
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
            return super(SASHeader, self).__contains__(key)
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

    def read_from_PAXE(self, filename_or_paxe):
        """Read header data from a PAXE (Saclay, France or Budapest, Hungary)
        measurement file.
        
        Inputs:
            filename_or_paxe: the file name (usually XE????.DAT) or a dict
                loaded by readPAXE().
                
        Outputs: the updated header structure. Fields not present in the file
            are kept unchanged.
        """
        if isinstance(filename_or_paxe, basestring):
            filename_or_paxe = header.readPAXE(filename_or_paxe)[0]
        self.update(filename_or_paxe)
        self._key_aliases['EnergyCalibrated'] = 'Energy'
        return self

    def read_from_ESRF_ID02(self, filename_or_edf):
        """Read header data from an ESRF ID02 EDF file.
        
        Inputs:
            filename_or_edf: the full name of the file or an edf structure read
                by readehf()
                
        Outputs: the updated header structure. Fields not present in the file
            are kept unchanged.
        """
        if isinstance(filename_or_edf, basestring):
            filename_or_edf = header.readehf(filename_or_edf)
        self.update(filename_or_edf)
        self._key_aliases['FSN'] = 'HMRunNumber'
        self._key_aliases['BeamPosX'] = 'Center_2'
        self._key_aliases['BeamPosY'] = 'Center_1'
        self._key_aliases['MeasTime'] = 'ExposureTime'
        self._key_aliases['Monitor'] = 'Intensity0'
        self._key_aliases['Detector'] = 'DetectorInfo'
        self._key_aliases['Date'] = 'HMStartTime'
        self._key_aliases['Wavelength'] = 'WaveLength'
        self._key_aliases['EnergyCalibrated'] = 'Energy'
        self['Hour'] = self['HMStartTime'].hour
        self['Minutes'] = self['HMStartTime'].minute
        self['Month'] = self['HMStartTime'].month
        self['Day'] = self['HMStartTime'].day
        self['Year'] = self['HMStartTime'].year
        self['Transm'] = self['Intensity1'] / self['Intensity0']
        self['Energy'] = 12398.419 / (self['WaveLength'] * 1e10)
        self['Dist'] = self['SampleDistance'] * 1000
        self['XPixel'] = (self['PSize_1'] * 1000)
        self['YPixel'] = (self['PSize_2'] * 1000)
        self['PixelSize'] = 0.5 * (self['XPixel'] + self['YPixel'])
        self['Title'] = self['TitleBody']
        self['Origin'] = 'ESRF_ID02'
        self['maskid'] = os.path.splitext(self['MaskFileName'])[0]
        for k in sorted([k for k in self if k.startswith('History')]):
            self.add_history(self[k], self['HMStartTime'])
        self.add_history('Loaded EDF header from file ' + filename_or_edf['FileName'])
        return self

    def read_from_B1_org(self, filename):
        self.update(header.readB1header(filename))

        self.add_history('Original header loaded: ' + filename)
        return self

    def read_from_B1_log(self, filename):
        """Read B1 logfile (*.log)
        
        Inputs:
            filename: the file name
                
        Output: the fields of this header are updated, old fields are kept
            unchanged. The header instance is returned as well.
        """
        fid = open(filename, 'r') #try to open. If this fails, an exception is raised
        for l in fid:
            try:
                ld = [ld for ld in self._logfile_data if l.split(':')[0].strip() == ld[0]][0]
            except IndexError:
                #line is not recognized.
                continue
            if len(ld) < 4:
                reader = unicode
            else:
                reader = ld[3]
            vals = reader(l.split(':')[1].strip())
            if isinstance(ld[1], tuple):
                #more than one field names. The reader function should return a 
                # tuple here, a value for each field.
                if len(vals) != len(ld[1]):
                    raise ValueError('Cannot read %d values from line %s in file!' % (len(ld[1]), l))
                self.update(dict(zip(ld[1], vals)))
            else:
                self[ld[1]] = vals
        fid.close()
        self.add_history('B1 logfile loaded: ' + filename)
        return self

    def write_B1_log(self, filename):
        """Write this header structure into a B1 logfile.
        
        Inputs:
            filename: name of the file.
            
        Notes:
            exceptions pass through to the caller.
        """
        allkeys = self.keys()
        f = open(filename, 'wt')
        for ld in self._logfile_data: #process each line
            linebegin = ld[0]
            fieldnames = ld[1]
            #set the default formatter if it is not given
            if len(ld) < 3:
                formatter = unicode
            elif ld[2] is None:
                formatter = unicode
            else:
                formatter = ld[2]
            #this will contain the formatted values.
            formatted = ''
            print fieldnames
            if isinstance(fieldnames, basestring):
                #scalar field name, just one field. Formatter should be a callable.
                if fieldnames not in allkeys:
                    #this field has already been processed
                    continue
                try:
                    formatted = formatter(self[fieldnames])
                except KeyError:
                    #field not found in param structure
                    continue
            elif isinstance(fieldnames, tuple):
                #more than one field names in a tuple. In this case, formatter can
                # be a tuple of callables...
                if all([(fn not in allkeys) for fn in fieldnames]):
                    #if all the fields have been processed:
                    continue
                if isinstance(formatter, tuple) and len(formatter) == len(fieldnames):
                    formatted = ' '.join([ft(self[fn]) for ft, fn in zip(formatter, fieldnames)])
                #...or a single callable...
                elif not isinstance(formatter, tuple):
                    formatted = formatter([self[fn] for fn in fieldnames])
                #...otherwise raise an exception.
                else:
                    raise SyntaxError('Programming error: formatter should be a scalar or a tuple\
of the same length as the field names in logfile_data.')
            else: #fieldnames is neither a string, nor a tuple.
                raise SyntaxError('Invalid syntax (programming error) in logfile_data in writeparamfile().')
            #try to get the values
            f.write(linebegin + ':\t' + formatted + '\n')
            if isinstance(fieldnames, tuple):
                for fn in fieldnames: #remove the params treated.
                    if fn in allkeys:
                        allkeys.remove(fn)
            else:
                if fieldnames in allkeys:
                    allkeys.remove(fieldnames)
        #write untreated params
        for k in allkeys:
            f.write(k + ':\t' + unicode(self[k]) + '\n')
        f.close()

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

    #--------------------------- new_from_*() loader methods ------------------
    @classmethod
    def _new_from_general(cls, args, kwargs, readfunction_name, defaults = None,
                          argnames = None):
        """General helper for new_from_*() class methods.
        
        Inputs:
            args, kwargs: the arguments given to the caller function.
            argnames: the names of the arguments of the caller function. Do not
                touch these, as one can break things seriously (yes, badly
                written code, I admit). Defaults to ['fsn', 'filename_format',
                'dirs'].
            defaults: default value for each argument (dict). If None, it will
                be empty.
            readfunction_name: the read function which has to be called (will
                be called with a filename)
        """
        if defaults is None:
            defaults = {}
        if argnames is None:
            argnames = ['fsn', 'filename_format', 'dirs']
        if len(args) == 1 and isinstance(args[0], basestring) and not kwargs:
            #filename supplied as a positional argument
            filenames = [args[0]]; scalar = True
        elif 'filename' in kwargs and not args:
            # filename supplied as a keyword argument
            filenames = [kwargs['filename']]; scalar = True
        else:
            # fsn, dirs etc. are supported as positional and keyword arguments.
            # we have to decode them and get a dictionary of keyword arguments.
            for i, argname in zip(itertools.count(0), argnames):
                # check if the ith positional argument is given. It is, put it
                # in the kwargs dict with the appropriate name.
                if len(args) > i:
                    if argname in kwargs:
                        raise ValueError('Argument %s defined twice!' % argname)
                    else:
                        kwargs[argname] = args[i]

            # get default values.
            for i, argname in zip(itertools.count(0), argnames):
                if argname in kwargs:
                    continue
                if argname not in defaults:
                    raise ValueError('Argument %s is not defined!' % argname)
                else:
                    kwargs[argname] = defaults[argname]

            # normalize the FSN argument.
            filenames = []
            if np.isscalar(kwargs['fsn']):
                kwargs['fsn'] = [kwargs['fsn']]
                scalar = True
            else:
                scalar = False
            # find the filenames.
            for f in kwargs['fsn']:
                try:
                    fn = kwargs['filename_format'] % f
                    filenames.append(misc.findfileindirs(fn, dirs = kwargs['dirs']))
                except IOError:
                    if scalar:
                        raise
                    continue
        # Now we have a filenames list. Try to load each file in this list.
        instances = []
        for fn in filenames:
            inst = cls()
            getattr(inst, readfunction_name)(fn) # this should not raise an exception, since the file exists. Hopefully it is readable too.
            instances.append(inst)
        if scalar:
            return instances[0]
        else:
            return instances

    @classmethod
    def new_from_ESRF_ID02(cls, *args, **kwargs):
        """Load ESRF ID02 headers.
        
        Reading just one file with full path:
        >>> new_from_ESRF_ID02(filename) # returns a SASHeader 
        
        Reading one file but searching in directories (fsn is scalar):
        >>> new_from_ESRF_ID02(fsn, filename_format, dirs) #returns a SASHeader
        
        Reading more files in several directories (fsn is a sequence):
        >>> new_from_ESRF_ID02(fsn, filename_format, dirs) #returns a list of
        ...                                                #SASHeaders
        """
        return cls._new_from_general(args, kwargs, 'read_from_ESRF_ID02',
                                     {'filename_format':'sc3269_0_%04dccd', 'dirs':['.']})

    @classmethod
    def new_from_B1_org(cls, *args, **kwargs):
        """Load a header from an org_?????.header file, beamline B1, HASYLAB.
        
        Reading just one file with full path:
        >>> new_from_B1_org(filename) # returns a SASHeader 
        
        Reading one file but searching in directories (fsn is scalar):
        >>> new_from_B1_org(fsn, filename_format, dirs) #returns a SASHeader
        
        Reading more files in several directories (fsn is a sequence):
        >>> new_from_B1_org(fsn, filename_format, dirs) #returns a list of
        ...                                             #SASHeaders
        """
        return cls._new_from_general(args, kwargs, 'read_from_B1_org',
                                     {'filename_format':'org_%05d.header', 'dirs':['.']})

    @classmethod
    def new_from_B1_log(cls, *args, **kwargs):
        """Load a logfile, beamline B1, HASYLAB
        
        Reading just one file with full path:
        >>> new_from_B1_log(filename) # returns a SASHeader 
        
        Reading one file but searching in directories (fsn is scalar):
        >>> new_from_B1_log(fsn, filename_format, dirs) #returns a SASHeader
        
        Reading more files in several directories (fsn is a sequence):
        >>> new_from_B1_log(fsn, filename_format, dirs) #returns a list of
        ...                                             #SASHeaders
        """
        return cls._new_from_general(args, kwargs, 'read_from_B1_log',
                                     {'filename_format':'intnorm%d.log', 'dirs':['.']})

    @classmethod
    def new_from_PAXE(cls, *args, **kwargs):
        """Load a PAXE file (BNC, Yellow Submarine)
        
        Reading just one file with full path:
        >>> new_from_PAXE(filename) # returns a SASHeader 
        
        Reading one file but searching in directories (fsn is scalar):
        >>> new_from_PAXE(fsn, filename_format, dirs) #returns a SASHeader
        
        Reading more files in several directories (fsn is a sequence):
        >>> new_from_PAXE(fsn, filename_format, dirs) #returns a list of
        ...                                             #SASHeaders
        
        """
        return cls._new_from_general(args, kwargs, 'read_from_PAXE',
                                     {'filename_format':'XE%04d.DAT', 'dirs':['.']})

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
                    hdf_entity.attrs[k] = _linearize_history(self[k]).encode('utf-8')
                elif isinstance(self[k], bool):
                    hdf_entity.attrs[k] = int(self[k])
                elif isinstance(self[k], numbers.Number):
                    hdf_entity.attrs[k] = self[k]
                elif isinstance(self[k], basestring):
                    hdf_entity.attrs[k] = self[k].encode('utf-8')
                elif isinstance(self[k], collections.Sequence):
                    hdf_entity.attrs[k] = self[k]
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
