'''
Basic classes to represent expositions and their metadata.

Created on Apr 5, 2012

@author: andris
'''

import datetime
import gzip
import math
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import numbers
import h5py
import functools
import scipy.io
import os
import warnings
import itertools
import matplotlib.nxutils  #this contains pnpoly

from .. import dataset
from .. import utils2d
from .. import misc
import twodim

def debug(*args, **kwargs):
    for a in args:
        print repr(a)
    for k in kwargs:
        print k, ':', repr(kwargs[k])


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


def _linearize_list(l,pre_converter=lambda a:a,post_converter=lambda a:a):
    return post_converter(' '.join([unicode(pre_converter(x)) for x in l]))


def _delinearize_list(l, pre_converter=lambda a:a, post_converter=list):
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

class _HDF_parse_group(object):
    def __init__(self,hdf_argument):
        self.hdf_argument=hdf_argument
    def __enter__(self):
        if isinstance(self.hdf_argument,basestring):
            self.hdf_file=h5py.highlevel.File(self.hdf_argument)
            self.hdf_group=self.hdf_file
        elif isinstance(self.hdf_argument,h5py.highlevel.File):
            self.hdf_file=self.hdf_argument
            self.hdf_group=self.hdf_file
        elif isinstance(self.hdf_argument,h5py.highlevel.Group):
            self.hdf_file=self.hdf_argument.file
            self.hdf_group=self.hdf_argument
        else:
            raise ValueError
        return self.hdf_group
    def __exit__(self,exc_type,exc_value,exc_traceback):
        if isinstance(self.hdf_argument,basestring):
            self.hdf_file.close()
            
class SASMaskException(Exception):
    pass

class SASAverageException(Exception):
    pass

class General_new_from():
    @classmethod
    def _new_from_general(cls, args, kwargs, readfunction_name, defaults={},
                          argnames=['fsn','filename_format','dirs']):
        """General helper for new_from_*() class methods.
        
        Inputs:
            args, kwargs: the arguments given to the caller function.
            argnames: the names of the arguments of the caller function. Do not
                touch these, as one can break things seriously (yes, badly
                written code, I admit).
            defaults: default value for each argument (dict)
            readfunction_name: the read function which has to be called (will
                be called with a filename)
        """
        if len(args)==1 and isinstance(args[0],basestring) and not kwargs:
            #filename supplied as a positional argument
            filenames=[args[0]]; scalar=True
        elif 'filename' in kwargs and not args:
            # filename supplied as a keyword argument
            filenames=[kwargs['filename']]; scalar=True
        else:
            # fsn, dirs etc. are supported as positional and keyword arguments.
            # we have to decode them and get a dictionary of keyword arguments.
            for i,argname in zip(itertools.count(0),argnames):
                # check if the ith positional argument is given. It is, put it
                # in the kwargs dict with the appropriate name.
                if len(args)>i:
                    if argname in kwargs:
                        raise ValueError('Argument %s defined twice!'%argname)
                    else:
                        kwargs[argname]=args[i]
            
            # get default values.
            for i, argname in zip(itertools.count(0),argnames):
                if argname in kwargs:
                    continue
                if argname not in defaults:
                    raise ValueError('Argument %s is not defined!'%argname)
                else:
                    kwargs[argname]=defaults[argname]
            
            # normalize the FSN argument.
            filenames=[]
            if np.isscalar(kwargs['fsn']):
                kwargs['fsn']=[kwargs['fsn']]
                scalar=True
            else:
                scalar=False
            # find the filenames.
            for f in kwargs['fsn']:
                try:
                    fn=kwargs['filename_format']%f
                    filenames.append(misc.findfileindirs(fn,dirs=kwargs['dirs']))
                except IOError:
                    if scalar:
                        raise
                    continue
        # Now we have a filenames list. Try to load each file in this list.
        instances=[]
        for fn in filenames:
            inst=cls()
            getattr(inst,readfunction_name)(fn) # this should not raise an exception, since the file exists. Hopefully it is readable too.
            instances.append(inst)
        if scalar:
            return instances[0]
        else:
            return instances
    

class SASHeader(dict, General_new_from):
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
                      functools.partial(_linearize_list, pre_converter=lambda a:a+1),
                      functools.partial(_delinearize_list,
                                        post_converter = lambda a:tuple([x-1 for x in a]))),
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
    _protectedfields_to_copy=['_protectedfields_to_copy','_key_aliases',
                              '_HDF5_read_postprocess_type','_logfile_data',
                              '_fields_to_sum','_fields_to_average',
                              '_fields_to_collect','_equiv_tests']
    
    # -- Housekeeping methods: __init__, iterators, __missing__ etc. ----------
    
    def __init__(self, *args, **kwargs):
        """This constructor behaves identically to that of the superclass. If
        the first positional argument is a SASHeader, this copies over the
        protected parameters whose names are found in _protectedfields_to_copy.
        """
        self._key_aliases={}
        super(SASHeader, self).__init__(self,*args, **kwargs)
        if args and isinstance(args[0],SASHeader):
            # copy over protected attributes
            for fn in args[0]._protectedfields_to_copy:
                attr=getattr(args[0],fn)
                if hasattr(attr,'copy'):
                    # if the attribute has a copy method, use that. E.g. dicts.
                    setattr(self,fn,attr.copy())
                elif isinstance(attr,collections.Sequence):
                    # if the attribute is a sequence, use the [:] construct.
                    setattr(self,fn,attr[:])
                else:
                    try:
                        # try to use a copy constructor
                        setattr(self,fn,attr.type(attr))
                    except:
                        # set it as is and hope for the best.
                        setattr(self,fn,attr)
    def copy(self, *args, **kwargs):
        """Make a copy of this header structure"""
        d = super(SASHeader, self).copy(*args, **kwargs)
        return SASHeader(d)
    def __missing__(self,key):
        """Create default values for missing fields"""
        if key in ['FSNs']:
            val=[]
        elif key.endswith('Error'):
            val=0
        elif key in ['maskid']:
            val=None
        elif key.startswith('FSN'):
            val=0
        elif key=='Title':
            val='<untitled>'
        elif key in ['Dist','Energy','EnergyCalibrated','BeamPosX','BeamPosY','PixelSize']:
            val=np.NAN
        else:
            raise KeyError(key)
        val=self._default_factory(key)
        super(SASHeader,self).__setitem__(key,val)
        return val
    def __unicode__(self):
        """Print a short summary of this header"""
        return "FSN %s; %s; %s mm; %s eV" % (self['FSN'],self['Title'],self['Dist'],self['Energy'])
    __str__ = __unicode__
    def __repr__(self):
        return "<SASHeader: "+unicode(self)+'>'
    def __getitem__(self, key):
        """ respond to header[key] requests, implements key aliasing."""
        if key in self._key_aliases:
            return super(SASHeader,self).__getitem__(self._key_aliases[key])
        else:
            return super(SASHeader,self).__getitem__(key)
    def __setitem__(self, key, value):
        """ respond to header[key]=value requests, implements key aliasing."""
        if key in self._key_aliases:
            return self.__setitem__(self._key_aliases[key], value)
        else:
            return super(SASHeader,self).__setitem__(key, value)
    def __delitem__(self, key):
        """ respond to del header[key] requests, implements key aliasing."""
        if key in self:
            return super(SASHeader,self).__delitem__(key)
        elif key in self._key_aliases:
            return self.__delitem__(self._key_aliases[key])
        else:
            raise KeyError(key)
    def __contains__(self, key):
        """ respond to 'key' in header requests, implements key aliasing."""
        if key in self._key_aliases:
            return super(SASHeader,self).__contains__(self._key_aliases[key])
        else:
            return super(SASHeader,self).__contains__(key)
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
        return itertools.chain(super(SASHeader,self).iterkeys(),self._key_aliases.iterkeys())
    def itervalues(self):
        """ Iterator version of values()."""
        return itertools.chain(super(SASHeader,self).itervalues(),
                 itertools.imap(lambda x:self[self._key_aliases[x]],self._key_aliases.iterkeys() ))
    def iteritems(self):
        """ Iterator version of items()."""
        return itertools.izip(self.iterkeys(),self.itervalues())
    
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
        if isinstance(filename_or_paxe,basestring):
            filename_or_paxe=twodim.readPAXE(filename_or_paxe)[0]
        self.update(filename_or_paxe)
        self._key_aliases['EnergyCalibrated']='Energy'
        return self
    
    def read_from_ESRF_ID02(self, filename_or_edf):
        """Read header data from an ESRF ID02 EDF file.
        
        Inputs:
            filename_or_edf: the full name of the file or an edf structure read
                by readehf()
                
        Outputs: the updated header structure. Fields not present in the file
            are kept unchanged.
        """
        if isinstance(filename_or_edf,basestring):
            filename_or_edf=twodim.readehf(filename_or_edf)
        self.update(filename_or_edf)
        self._key_aliases['FSN']='HMRunNumber'
        self._key_aliases['BeamPosX']='Center_2'
        self._key_aliases['BeamPosY']='Center_1'
        self._key_aliases['MeasTime']='ExposureTime'
        self._key_aliases['Monitor']='Intensity0'
        self._key_aliases['Detector']='DetectorInfo'
        self._key_aliases['Date']='HMStartTime'
        self._key_aliases['Wavelength']='WaveLength'
        self._key_aliases['EnergyCalibrated']='Energy'
        self['Hour']=self['HMStartTime'].hour
        self['Minutes']=self['HMStartTime'].minute
        self['Month']=self['HMStartTime'].month
        self['Day']=self['HMStartTime'].day
        self['Year']=self['HMStartTime'].year
        self['Transm']=self['Intensity1']/self['Intensity0']
        self['Energy']=12398.419/(self['WaveLength']*1e10)
        self['Dist']=self['SampleDistance']*1000;
        self['XPixel']=(self['PSize_1']*1000)
        self['YPixel']=(self['PSize_2']*1000)
        self['PixelSize']=0.5*(self['XPixel']+self['YPixel'])
        self['Title']=self['TitleBody']
        self['Origin']='ESRF_ID02'
        self['maskid']=os.path.splitext(self['MaskFileName'])[0]
        for k in sorted([k for k in self if k.startswith('History')]):
            self.add_history(self[k],self['HMStartTime'])
        self.add_history('Loaded EDF header from file '+filename_or_edf['FileName'])
        return self
    
    def read_from_B1_org(self, filename):
        #Planck's constant times speed of light: incorrect
        # constant in the old program on hasjusi1, which was
        # taken over by the measurement program, to keep
        # compatibility with that.
        jusifaHC = 12396.4
        if filename.upper().endswith('.GZ'):
            fid = gzip.GzipFile(filename, 'r')
        else:
            fid = open(filename, 'rt')
        lines = fid.readlines()
        fid.close()
        self['FSN'] = int(lines[0].strip())
        self['Hour'] = int(lines[17].strip())
        self['Minutes'] = int(lines[18].strip())
        self['Month'] = int(lines[19].strip())
        self['Day'] = int(lines[20].strip())
        self['Year'] = int(lines[21].strip()) + 2000
        self['FSNref1'] = int(lines[23].strip())
        self['FSNdc'] = int(lines[24].strip())
        self['FSNsensitivity'] = int(lines[25].strip())
        self['FSNempty'] = int(lines[26].strip())
        self['FSNref2'] = int(lines[27].strip())
        self['Monitor'] = float(lines[31].strip())
        self['Anode'] = float(lines[32].strip())
        self['MeasTime'] = float(lines[33].strip())
        self['Temperature'] = float(lines[34].strip())
        self['BeamPosX'] = float(lines[36].strip())
        self['BeamPosY'] = float(lines[37].strip())
        self['Transm'] = float(lines[41].strip())
        self['Wavelength'] = float(lines[43].strip())
        self['Energy'] = jusifaHC / self['Wavelength']
        self['Dist'] = float(lines[46].strip())
        self['XPixel'] = 1 / float(lines[49].strip())
        self['YPixel'] = 1 / float(lines[50].strip())
        self['Title'] = lines[53].strip().replace(' ', '_').replace('-', '_')
        self['MonitorDORIS'] = float(lines[56].strip())  # aka. DORIS counter
        self['Owner'] = lines[57].strip()
        self['RotXSample'] = float(lines[59].strip())
        self['RotYSample'] = float(lines[60].strip())
        self['PosSample'] = float(lines[61].strip())
        self['DetPosX'] = float(lines[62].strip())
        self['DetPosY'] = float(lines[63].strip())
        self['MonitorPIEZO'] = float(lines[64].strip())  # aka. PIEZO counter
        self['BeamsizeX'] = float(lines[66].strip())
        self['BeamsizeY'] = float(lines[67].strip())
        self['PosRef'] = float(lines[70].strip())
        self['Monochromator1Rot'] = float(lines[77].strip())
        self['Monochromator2Rot'] = float(lines[78].strip())
        self['Heidenhain1'] = float(lines[79].strip())
        self['Heidenhain2'] = float(lines[80].strip())
        self['Current1'] = float(lines[81].strip())
        self['Current2'] = float(lines[82].strip())
        self['Detector'] = 'Unknown'
        self['PixelSize'] = (self['XPixel'] + self['YPixel']) / 2.0
        
        self['AnodeError'] = math.sqrt(self['Anode'])
        self['TransmError'] = 0
        self['MonitorError'] = math.sqrt(self['Monitor'])
        self['MonitorPIEZOError'] = math.sqrt(self['MonitorPIEZO'])
        self['MonitorDORISError'] = math.sqrt(self['MonitorDORIS'])
        self['Date'] = datetime.datetime(self['Year'], self['Month'], self['Day'], self['Hour'], self['Minutes'])
        self['Origin'] = 'B1 original header'
        self.add_history('Original header loaded: ' + filename)
        return self
    
    def read_from_B1_log(self, filename):
        """Read B1 logfile (*.log)
        
        Inputs:
            filename: the file name
                
        Output: the fields of this header are updated, old fields are kept
            unchanged. The header instance is returned as well.
        """
        fid = open(filename, 'r'); #try to open. If this fails, an exception is raised
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
        return self;
    
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
    
    def add_history(self, text, time=None):
        """Add a new entry to the history.
        
        Inputs:
            text: history text
            time: time of the event. If None, the current time will be used.
        """
        if time is None:
            time=datetime.datetime.now()
        if 'History' not in self:
            self['History'] = []
        deltat=time-datetime.datetime.fromtimestamp(0,time.tzinfo)
        deltat_seconds=deltat.seconds+deltat.days*24*3600+deltat.microseconds*1e-6
        self['History'].append((deltat_seconds, text))
        
    def get_history(self):
        """Return the history in a human-readable format"""
        return '\n'.join([str(h[0])+': '+h[1] for h in self['History']])
    
    #--------------------------- new_from_*() loader methods ------------------
    
    @classmethod
    def new_from_ESRF_ID02(cls,*args, **kwargs):
        """Load ESRF ID02 headers.
        
        Reading just one file with full path:
        >>> new_from_ESRF_ID02(filename) # returns a SASHeader 
        
        Reading one file but searching in directories (fsn is scalar):
        >>> new_from_ESRF_ID02(fsn, filename_format, dirs) #returns a SASHeader
        
        Reading more files in several directories (fsn is a sequence):
        >>> new_from_ESRF_ID02(fsn, filename_format, dirs) #returns a list of
        ...                                                #SASHeaders
        """
        return cls._new_from_general(args,kwargs,'read_from_ESRF_ID02',
                                     {'filename_format':'sc3269_0_%04dccd','dirs':['.']})
        
    @classmethod
    def new_from_B1_org(cls, *args,**kwargs):
        """Load a header from an org_?????.header file, beamline B1, HASYLAB.
        
        Reading just one file with full path:
        >>> new_from_B1_org(filename) # returns a SASHeader 
        
        Reading one file but searching in directories (fsn is scalar):
        >>> new_from_B1_org(fsn, filename_format, dirs) #returns a SASHeader
        
        Reading more files in several directories (fsn is a sequence):
        >>> new_from_B1_org(fsn, filename_format, dirs) #returns a list of
        ...                                             #SASHeaders
        """
        return cls._new_from_general(args,kwargs,'read_from_B1_org',
                                     {'filename_format':'org_%05d.header','dirs':['.']})
        
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
        return cls._new_from_general(args,kwargs,'read_from_B1_log',
                                     {'filename_format':'intnorm%d.log','dirs':['.']})
        
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
        return cls._new_from_general(args,kwargs,'read_from_PAXE',
                                     {'filename_format':'XE%04d.DAT','dirs':['.']})
        
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
            sumweight = sum([1 / a[dk] ** 2 for a in args]);
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
                   [other._equiv_tests[k](self[k], other[k]) for k in other._equiv_tests])
    
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

class SASExposure(object,General_new_from):
    """A class for holding SAS exposure data, i.e. intensity, error, metadata
    and mask.
    
    A SASExposure has the following special attributes:
    
    Intensity: (corrected) scattered intensity matrix (np.ndarray).
    Error: error of (corrected) scattered intensity (np.ndarray).
    Image: raw scattering image (np.ndarray).
    header: metadata dictionary (SASHeader instance).
    mask: mask matrix in the form of a SASMask instance. 
    
    any of the above attributes can be missing, a value of None signifies
    this situation.

    
    """
    matrix_names=['Image','Intensity','Error']
    matrices=dict([('Image','Detector Image'),
                                      ('Intensity','Corrected intensity'),
                                      ('Error','Error of intensity')])
    def __init__(self):
        super(SASExposure, self).__init__()
        self.Intensity=None
        self.Error=None
        self.Image=None
        self.header=SASHeader()
        self.mask=None
    def check_for_mask(self):
        if self.mask is None:
            raise SASMaskException('mask not defined')
    def check_for_q(self):
        missing=[x for x in  ['BeamPosX','BeamPosY','Dist','EnergyCalibrated','PixelSize'] if x not in self.header]
        if missing:
            raise SASAverageException('Fields missing from header: '+str(missing))
    def __del__(self):
        del self.Intensity
        del self.Error
        del self.Image
        del self.header
        del self.mask

### -------------- Loading routines (new_from_xyz) ------------------------

    @classmethod
    def new_from_ESRF_ID02(cls,*args,**kwargs):
        """Load ESRF ID02 data files.
        
        Reading just one file with full path:
        >>> new_from_ESRF_ID02(filename) # returns a SASExposure 
        
        Reading one file but searching in directories (fsn is scalar):
        >>> new_from_ESRF_ID02(fsn, filename_format, dirs) #returns a SASExposure
        
        Reading more files in several directories (fsn is a sequence):
        >>> new_from_ESRF_ID02(fsn, filename_format, dirs) #returns a list of
        ...                                                #SASExposures
        """
        obj=cls()
        obj.read_from_ESRF_ID02(*args,**kwargs)
        return obj
    @classmethod
    def new_from_B1_org(cls,*args,**kwargs):
        obj=cls()
        obj.read_from_B1_org(*args,**kwargs)
        return obj
    @classmethod
    def new_from_B1_int2dnorm(cls, *args, **kwargs):
        obj = cls()
        obj.read_from_B1_int2dnorm(*args, **kwargs)
        return obj
    @classmethod
    def new_from_PAXE(cls, *args, **kwargs):
        obj=cls()
        obj.read_from_PAXE(*args,**kwargs)
        return obj
    @classmethod
    def new_from_BDF(cls, *args, **kwargs):
        obj=cls()
        obj.read_from_BDF(*args,**kwargs)
        return obj
    @classmethod
    def new_from_hdf5(cls, hdf_or_filename):
        #get a HDF file object
        ret=[]
        with _HDF_parse_group(hdf_or_filename) as hpg:
            if hpg.name.startswith('/FSN'):
                hdfgroups=[hpg]
            else:
                hdfgroups = [x for x in hpg.keys() if x.startswith('FSN')]
            for g in hdfgroups:
                ret.append(cls())
                ret[-1].read_from_hdf5(hpg[g],load_mask=False)
            # adding masks later, thus only one copy of each mask will exist in
            # the memory 
            masknames=set([r.header['maskid'] for r in ret])
            masks=dict([(mn,SASMask.new_from_hdf5(hpg,mn)) for mn in masknames])
            for r in ret:
                r.set_mask(masks[r.header['maskid']])
        return ret

### -------------- reading routines--------------------------

    def read_from_ESRF_ID02(self,fsn,fileformat,estimate_errors=True,dirs=[]):
        """Read an EDF file (ESRF beamline ID02 SAXS pattern)
        
        Inputs:
            fsn: file sequence number
            fileformat: c-style file format, e.g. sc3269_0_%04dccd
            estimate_errors: error matrices are usually not saved, but they can
                be estimated from the intensity.
            dirs: folders to look file for.
        """
        filename=misc.findfileindirs(fileformat%fsn,dirs)
        edf=twodim.readedf(filename)
        self.header=SASHeader.new_from_ESRF_ID02(edf)
        self.Intensity=edf['data'].astype(np.double)
        mask=SASMask(misc.findfileindirs(self.header['MaskFileName'],dirs))
        if self.Intensity.shape!=mask.mask.shape:
            if all(self.Intensity.shape[i]>mask.mask.shape[i] for i in [0,1]):
                xbin,ybin=[self.Intensity.shape[i]/mask.mask.shape[i] for i in [0,1]]
                extend=True
            elif all(self.Intensity.shape[i]<mask.mask.shape[i] for i in [0,1]):
                xbin,ybin=[mask.mask.shape[i]/self.Intensity.shape[i] for i in [0,1]]
                extend=False
            else:
                raise ValueError('Cannot do simultaneous forward and backward mask binning.')
            warnings.warn('Rebinning mask: %s x %s, direction: %s'%(xbin,ybin,['shrink','enlarge'][extend]))
            mask=mask.rebin(xbin,ybin,extend)
        self.set_mask(mask)
        dummypixels=np.absolute(self.Intensity-self.header['Dummy'])<=self.header['DDummy']
        #self.Intensity[dummypixels]=0
        self.mask.mask&=(-dummypixels).reshape(self.Intensity.shape)
        if estimate_errors:
            sd=edf['SampleDistance']
            ps2=edf['PSize_1']*edf['PSize_2']
            I1=edf['Intensity1']
            self.Error=(0.5*sd*sd/ps2/I1+self.Intensity)*float(sd*sd)/(ps2*I1)

    def read_from_B1_org(self,fsn,fileformat='org_%05d',dirs=['.']):
        """Read an original exposition (beamline B1, HASYLAB/DESY, Hamburg)
        
        Inputs:
            fsn: file sequence number
            fileformat: C-style file format (without extension)
            dirs: folders to look for files in
        
        Notes:
            We first try to load the header file. Extensions .header, .DAT,
                .dat, .DAT.gz, .dat.gz are tried in this order.
            If the header has been successfully loaded, we try to load the data.
                Extensions: .cbf, .tif, .tiff, .DAT, .DAT.gz, .dat, .dat.gz are
                tried in this sequence.
            If either the header or the data cannot be loaded, an IOError is
                raised.
            
        """
        #try to load header file
        headername=''
        for extn in ['.header','.DAT','.dat','.DAT.gz','.dat.gz']:
            try:
                headername=misc.findfileindirs(fileformat%fsn+extn,dirs);
            except IOError:
                continue
        if not headername:
            raise IOError('Could not find header file')
        dataname=''
        for extn in ['.cbf','.tif','.tiff','.DAT','.DAT.gz','.dat','.dat.gz']:
            try:
                dataname=misc.findfileindirs(fileformat%fsn+extn,dirs)
            except IOError:
                continue
        if not dataname:
            raise IOError('Could not find 2d org file') #skip this file
        self.header=SASHeader.new_from_B1_org(headername)
        if dataname.lower().endswith('.cbf'):
            self.Image=twodim.readcbf(dataname)
        elif dataname.upper().endswith('.DAT') or dataname.upper().endswith('.DAT.GZ'):
            self.Image=twodim.readjusifaorg(dataname).reshape(256,256)
        elif dataname.upper().endswith('.TIF') or dataname.upper().endswith('.TIFF'):
            self.Image=twodim.readtif(dataname)
        else:
            raise NotImplementedError(dataname)
        return self

    def read_from_B1_int2dnorm(self, fsn, fileformat = 'int2dnorm%d', logfileformat = 'intnorm%d.log', dirs = ['.']):
        dataname = None
        for extn in ['.npy', '.mat']:
            try:
                dataname = misc.findfileindirs(fileformat % fsn + extn, dirs)
            except IOError:
                continue
        if not dataname:
            raise IOError('Cannot find two-dimensional file!')
        headername = misc.findfileindirs(logfileformat % fsn, dirs)
        self.header = SASHeader.new_from_B1_log(headername)
        self.Intensity, self.Error = twodim.readint2dnorm(dataname)
        self.header.add_history('Intensity and Error matrices loaded from ' + dataname)
        return self

    def read_from_hdf5(self, hdf_group,load_mask=True):
        for k in hdf_group.keys():
            self.__setattr__(k,hdf_group[k].value)
        self.header=SASHeader()
        self.header.read_from_hdf5(hdf_group)
        if self.header['maskid'] is not None:
            self.mask=SASMask.new_from_hdf5(hdf_group.parent,self.header['maskid'])

    def read_from_PAXE(self, fsn, fileformat='XE%04d.DAT', dirs=['.']):
        paxe=twodim.readPAXE(misc.findfileindirs(fileformat%fsn,dirs=dirs))
        self.header=SASHeader()
        self.header.read_from_PAXE(paxe[0])
        self.Image=paxe[1]
        return self
    
    def read_from_BDF(self, fsn, fileformat, dirs=['.']):
        raise NotImplementedError
### ------------------- Interface routines ------------------------------------
    def set_mask(self, mask):
        self.mask = SASMask(mask)
        self.header['maskid']=self.mask.maskid
        self.header.add_history('Mask %s associated to exposure.'%self.mask.maskid)

    def get_matrix(self,name='Intensity',othernames=None):
        name=self.get_matrix_name(name,othernames)
        return getattr(self,name)

    def get_matrix_name(self,name='Intensity',othernames=None):
        if name in self.matrices.values():
            name=[k for k in self.matrices if self.matrices[k]==name][0]
        if hasattr(self,name) and (getattr(self,name) is not None):
            return name
        if othernames is None:
            othernames=self.matrix_names
        
        for k in othernames:
            try:
                if getattr(self,k) is not None:
                    return k
            except AttributeError:
                pass
        raise AttributeError('No matrix in this instance of'+str(type(self)))

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
                                            self.header['Dist'],
                                            self.header['PixelSize'],
                                            self.header['PixelSize'],
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
    def common_qrange(cls,*exps):
        """Find a common q-range for several exposures
        
        Usage:
        
        >>> SASExposure.common_qrange(exp1, exp2, exp3, ...)
        
        where exp1, exp2, exp3... are instances of the SASExposure class.
        
        Returns:
            the estimated common q-range in a np.ndarray (ndim=1).
        """
        if not all([isinstance(e,cls) for e in exps]):
            raise ValueError('All arguments should be SASExposure instances.')
        qranges=[e.get_qrange() for e in exps]
        qmin=max([qr.min() for qr in qranges])
        qmax=min([qr.max() for qr in qranges])
        N=min([len(qr) for qr in qranges])
        return np.linspace(qmin,qmax,N)

    def radial_average(self, qrange = None, pixel=False, matrix='Intensity', errormatrix='Error'):
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
        mat=getattr(self,matrix)
        if errormatrix is not None:
            err=getattr(self,errormatrix)
        else:
            err=None
        if not pixel:
            res = utils2d.integrate.radint(mat,err,
                                               self.header['EnergyCalibrated'],
                                               self.header['Dist'],
                                               self.header['PixelSize'],
                                               self.header['BeamPosX'],
                                               self.header['BeamPosY'],
                                               1 - self.mask.mask, qrange,
                                               returnavgq = True,
                                               returnpixel = True)
            if err is not None:
                q,I,E,A,p=res
            else:
                q,I,A,p=res
                E=np.zeros_like(q)
            ds = dataset.SASCurve(q, I, E)
            ds.addfield('Pixels', p)
        else:
            res = utils2d.integrate.radintpix(mat,err,
                                                     self.header['BeamPosX'],
                                                     self.header['BeamPosY'],
                                                     1-self.mask.mask, qrange,
                                                     returnavgpix=True)
            if err is not None:
                p,I,E,A=res
            else:
                p,I,A=res
                E=np.zeros_like(p)
            ds = dataset.DataSet(p, I, E)
        ds.addfield('Area', A)
        ds.header=SASHeader(self.header)
        return ds
    
    def sector_average(self, phi0, dphi, qrange=None, pixel=False, matrix='Intensity', errormatrix='Error', symmetric_sector=False):
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
        mat=getattr(self,matrix)
        if errormatrix is not None:
            err=getattr(self,errormatrix)
        else:
            err=None
        if not pixel:
            res = utils2d.integrate.radint(mat,err,
                                               self.header['EnergyCalibrated'],
                                               self.header['Dist'],
                                               self.header['PixelSize'],
                                               self.header['BeamPosX'],
                                               self.header['BeamPosY'],
                                               1 - self.mask.mask, qrange,
                                               returnavgq = True,
                                               returnpixel = True,
                                               phi0=phi0, dphi=dphi, symmetric_sector=symmetric_sector)
            if err is not None:
                q,I,E,A,p=res
            else:
                q,I,A,p=res
                E=np.zeros_like(q)
            ds = dataset.SASCurve(q, I, E)
            ds.addfield('Pixels', p)
        else:
            res = utils2d.integrate.radintpix(mat,err,
                                                     self.header['BeamPosX'],
                                                     self.header['BeamPosY'],
                                                     1-self.mask.mask, qrange,
                                                     returnavgpix=True, phi0=phi0,
                                                     dphi=dphi, symmetric_sector=symmetric_sector)
            if err is not None:
                p,I,E,A=res
            else:
                p,I,A=res
                E=np.zeros_like(p)
            ds = dataset.DataSet(p, I, E)
        ds.addfield('Area', A)
        ds.header=SASHeader(self.header)
        return ds

    def azimuthal_average(self, qmin, qmax, Ntheta=100, pixel=False, matrix='Intensity', errormatrix='Error'):
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
        mat=getattr(self,matrix)
        if errormatrix is not None:
            err=getattr(self,errormatrix)
        else:
            err=None
        if not pixel:
            res = utils2d.integrate.azimint(mat,err,
                                               self.header['EnergyCalibrated'],
                                               self.header['Dist'],
                                               self.header['PixelSize'],
                                               self.header['BeamPosX'],
                                               self.header['BeamPosY'],
                                               1 - self.mask.mask, Ntheta,
                                               qmin = qmin, qmax=qmax)
            if err is not None:
                theta,I,E,A=res
            else:
                theta,I,A=res
                E=np.zeros_like(theta)
            ds = dataset.DataSet(theta, I, E)
        else:
            res = utils2d.integrate.azimintpix(mat,err,
                                                     self.header['BeamPosX'],
                                                     self.header['BeamPosY'],
                                                     1-self.mask.mask, Ntheta,
                                                     pixmin=qmin, pixmax=qmax)
            if err is not None:
                theta,I,E,A=res
            else:
                theta,I,A=res
                E=np.zeros_like(theta)
            ds = dataset.DataSet(theta, I, E)
        ds.addfield('Area', A)
        ds.header=SASHeader(self.header)
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
        drawmask [True]: if the mask is to be plotted.
        qrange_on_axis [True]: if the q-range is to be set to the axis.
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
        kwargs_default={'zscale':'linear',
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
        my_kwargs=['zscale','crosshair','drawmask','qrange_on_axis','matrix',
                   'axis','invalid_color','mask_opacity','minvalue','maxvalue',
                   'return_matrix']
        kwargs_default.update(kwargs)
        return_matrix=kwargs_default['return_matrix'] # save this as this will be removed when kwars_default is fed into imshow()
        
        kwargs_for_imshow=dict([(k,kwargs_default[k]) for k in kwargs_default if k not in my_kwargs])
        if isinstance(kwargs_default['zscale'],basestring):
            if kwargs_default['zscale'].upper().startswith('LOG10'):
                kwargs_default['zscale']=np.log10
            elif kwargs_default['zscale'].upper().startswith('LN'):
                kwargs_default['zscale']=np.log
            elif kwargs_default['zscale'].upper().startswith('LIN'):
                kwargs_default['zscale']=lambda a:a.copy()
            elif kwargs_default['zscale'].upper().startswith('LOG'):
                kwargs_default['zscale']=np.log
            else:
                raise ValueError('Invalid value for zscale: %s'%kwargs_default['zscale'])
        mat=self.get_matrix(kwargs_default['matrix']).copy()
        mat[mat<kwargs_default['minvalue']]=kwargs_default['minvalue']
        mat[mat>kwargs_default['maxvalue']]=kwargs_default['maxvalue']
        mat=kwargs_default['zscale'](mat)

        if kwargs_default['drawmask']:
            self.check_for_mask() # die if no mask is present

        if kwargs_default['qrange_on_axis']:
            self.check_for_q()
            xmin=4*np.pi*np.sin(0.5*np.arctan((0-self.header['BeamPosY'])*self.header['PixelSize']/self.header['Dist']))*self.header['EnergyCalibrated']/12398.419
            xmax=4*np.pi*np.sin(0.5*np.arctan((mat.shape[1]-self.header['BeamPosY'])*self.header['PixelSize']/self.header['Dist']))*self.header['EnergyCalibrated']/12398.419
            ymin=4*np.pi*np.sin(0.5*np.arctan((0-self.header['BeamPosX'])*self.header['PixelSize']/self.header['Dist']))*self.header['EnergyCalibrated']/12398.419
            ymax=4*np.pi*np.sin(0.5*np.arctan((mat.shape[0]-self.header['BeamPosX'])*self.header['PixelSize']/self.header['Dist']))*self.header['EnergyCalibrated']/12398.419
            if kwargs_for_imshow['origin'].upper()=='UPPER': 
                kwargs_for_imshow['extent']=[xmin,xmax,ymax,ymin]
            else:
                kwargs_for_imshow['extent']=[xmin,xmax,ymin,ymax]
            bcx=0
            bcy=0
        else:
            bcx=self.header['BeamPosX']
            bcy=self.header['BeamPosY']
            xmin=0; xmax=mat.shape[1]; ymin=0; ymax=mat.shape[0]
        
        if kwargs_default['axis'] is None:
            kwargs_default['axis']=plt.gca()
        ret=kwargs_default['axis'].imshow(mat,**kwargs_for_imshow)
        if kwargs_default['drawmask']:
            #workaround: because of the colour-scaling we do here, full one and
            #   full zero masks look the SAME, i.e. all the image is shaded.
            #   Thus if we have a fully unmasked matrix, skip this section.
            #   This also conserves memory.
            if self.mask.mask.sum()!=self.mask.mask.size:
                #Mask matrix should be plotted with plt.imshow(maskmatrix, cmap=_colormap_for_mask)
                _colormap_for_mask=matplotlib.colors.ListedColormap(['white','white'],'_sastool_%s'%misc.random_str(10))
                _colormap_for_mask._init()
                _colormap_for_mask._lut[:,-1]=0
                _colormap_for_mask._lut[0,-1]=kwargs_default['mask_opacity']
                kwargs_for_imshow['cmap']=_colormap_for_mask
                kwargs_default['axis'].imshow(self.mask.mask,**kwargs_for_imshow)
        if kwargs_default['crosshair']:
            ax=kwargs_default['axis'].axis()
            kwargs_default['axis'].plot([xmin,xmax],[bcx]*2,'w-')
            kwargs_default['axis'].plot([bcy]*2,[ymin,ymax],'w-')
            kwargs_default['axis'].axis(ax)
        kwargs_default['axis'].set_axis_bgcolor(kwargs_default['invalid_color'])
        kwargs_default['axis'].figure.canvas.draw()
        if return_matrix:
            return ret,mat
        else:
            return ret

###  ------------------------ Beam center finding -----------------------------

    def update_beampos(self,bc,source=None):
        """Update the beam position in the header.
        
        Inputs:
            bc: beam position coordinates (row, col; starting from 0).
            source: name of the beam finding algorithm.
        """
        self.header['BeamPosX'],self.header['BeamPosY']=bc
        if not source:
            self.header.add_history('Beam position updated to:'+str(tuple(bc)))
        else:
            self.header.add_history('Beam found by *%s*: %s'%(source,str(tuple(bc))))
    def find_beam_semitransparent(self, bs_area, update=True):
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
        bs_area=[min(bs_area[2:]),max(bs_area[2:]),min(bs_area[:2]),max(bs_area[:2])]
        bc=utils2d.centering.findbeam_semitransparent(self.get_matrix(), bs_area)
        if update:
            self.update_beampos(bc, source='semitransparent')
        return bc
    def find_beam_slices(self, pixmin=0, pixmax=np.inf, sector_width=np.pi/9.,
                         update=True, callback=None):
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
        bc=utils2d.centering.findbeam_slices(self.get_matrix(),
                                             (self.header['BeamPosX'],
                                              self.header['BeamPosY']),
                                             self.mask.mask, dmin=pixmin, 
                                             dmax=pixmax, 
                                             sector_width=sector_width,
                                             callback=callback)
        if update:
            self.update_beampos(bc, source='slices')
        return bc
    def find_beam_gravity(self, update=True):
        """Find the beam position by finding the center of gravity in each row
        and column.
        
        Inputs:
            update: if the new value should be written in the header (default).
                If False, the newly found beam position is only returned.
        
        Outputs:
            the beam position (row,col).
        """
        self.check_for_mask()
        bc=utils2d.centering.findbeam_gravity(self.get_matrix(),self.mask.mask)
        if update:
            self.update_beampos(bc,source='gravity')
        return bc
    
    def find_beam_azimuthal_fold(self, Ntheta=50, dmin=0, dmax=np.inf,
                                 update=True, callback=None):
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
        bc=utils2d.centering.findbeam_azimuthal_fold(self.get_matrix(),
                                                     (self.header['BeamPosX'],
                                                      self.header['BeamPosY']),
                                                     self.mask.mask,
                                                     Ntheta=Ntheta, dmin=dmin,
                                                     dmax=dmax, callback=callback)
        if update:
            self.update_beampos(bc,source='azimuthal_fold')
        return bc
    
    def find_beam_radialpeak(self, pixmin, pixmax, drive_by='amplitude', extent=10, update=True, callback=None):
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
        bc=utils2d.centering.findbeam_radialpeak(self.get_matrix(),
                                                 (self.header['BeamPosX'],
                                                  self.header['BeamPosY']),
                                                 self.mask.mask, pixmin,
                                                 pixmax, drive_by=drive_by,
                                                 extent=extent, callback=callback)
        if update:
            self.update_beampos(bc,source='radialpeak')
        return bc
    
### ----------------------- Writing routines ----------------------------------

    def write_to_hdf5(self, hdf_or_filename, **kwargs):
        """Save exposure to a HDF5 file or group.
        
        Inputs:
            hdf_or_filename: a file name (string) or an instance of 
                h5py.highlevel.File (equivalent to a HDF5 root group) or
                h5py.highlevel.Group.
            other keyword arguments are passed on as keyword arguments to the
                h5py.highlevel.Group.create_dataset() method.
                
        A HDF5 group will be created with the name FSN<fsn> and the available
        matrices (Image, Intensity, Error) will be saved. Header data is saved
        as attributes to the HDF5 group.
        
        If a mask is associated to this exposure, it is saved as well as a
        sibling group of FSN<fsn> with the name <maskid>.
        """
        if 'compression' not in kwargs:
            kwargs['compression'] = 'gzip'
        with _HDF_parse_group(hdf_or_filename) as hpg:
            groupname='FSN%d'%self.header['FSN']
            if groupname in hpg.keys():
                del hpg[groupname]
            hpg.create_group(groupname)
            for k in ['Intensity','Error','Image']:
                hpg[groupname].create_dataset(k,data=self.__getattribute__(k),**kwargs)
            self.header.write_to_hdf5(hpg[groupname])
            if self.mask is not None:
                self.mask.write_to_hdf5(hpg)
            
class SASMask(object):
    """Class to represent mask matrices.
    
    Each mask matrix should have a mask ID (a string starting with 'mask'),
    which should be unique. If a single pixel changes, a new ID should be
    created.
    
    This class can be instantiated by several ways:
    
    >>> mask=SASMask(<2d numpy array>)
    
    >>> mask=SASMask(<filename>, [<dirs>])
    supported file formats: .mat, .npz, .npy, .edf
    
    >>> mask=SASMask(<other SASMask instance>)
    
    Under the hood:
        the mask matrix is kept with dtype==np.uint8. The elements should only
        be 0-s (masked) or 1-s (unmasked), otherwise unexpected events may
        occur. The constructor and the member functions take care to preserve
        this consistency.
    """
    maskid = None
    _mask=None
    def __init__(self, maskmatrix=None,dirs='.'):
        super(SASMask,self).__init__()
        if maskmatrix is not None:
            if isinstance(maskmatrix,basestring) and \
                maskmatrix.lower()[-4:] in ['.mat','.npz','.npy']:
                self.read_from_mat(misc.findfileindirs(maskmatrix,dirs))
            elif isinstance(maskmatrix,basestring) and \
                maskmatrix.lower()[-4:] in ['.edf']:
                self.read_from_edf(misc.findfileindirs(maskmatrix,dirs))
            elif isinstance(maskmatrix,np.ndarray):
                self.mask=maskmatrix.astype(np.uint8)
                self.maskid='mask'+misc.random_str(6)
            elif isinstance(maskmatrix,SASMask):
                maskmatrix.copy_into(self)
            else:
                raise NotImplementedError
        else:
            raise ValueError('Empty SASMasks cannot be instantiated.')
            
    def __unicode__(self):
        return u'SASMask('+self.maskid+')'
    __str__=__unicode__
    __repr__=__unicode__
    def _setmask(self, maskmatrix):
        self._mask=(maskmatrix!=0).astype(np.uint8)
    def _getmask(self):
        return self._mask
    mask=property(_getmask,_setmask,doc='Mask matrix')
    def _getshape(self):
        return self._mask.shape
    shape=property(_getshape,doc='Shortcut to the shape of the mask matrix')
    def copy_into(self,into):
        """Helper function for deep copy."""
        if not isinstance(into,type(self)):
            raise ValueError('Incompatible class!')
        if self.mask is not None:
            into.mask = self.mask.copy()
        else:
            into.mask = None
        into.maskid = self.maskid
    def read_from_edf(self,filename):
        """Read a mask from an EDF file."""
        edf=twodim.readedf(filename)
        self.maskid=os.path.splitext(os.path.split(edf['FileName'])[1])[0]
        self.mask=(np.absolute(edf['data']-edf['Dummy'])>edf['DDummy']).reshape(edf['data'].shape)
        return self
    def read_from_mat(self,filename,fieldname=None):
        """Try to load a maskfile (Matlab(R) matrix file or numpy npz/npy)
        
        Inputs:
            filename: the input file name
            fieldname: field in the mat/npz file. None to autodetect.
        """
        if filename.lower().endswith('.mat'):
            f=scipy.io.loadmat(filename)
        elif filename.lower().endswith('.npz'):
            f=np.load(filename)
        elif filename.lower().endswith('.npy'):
            f=dict([(os.path.splitext(os.path.split(filename)[1])[0],np.load(filename))])
        else:
            raise ValueError('Invalid file name format!')
        
        if f is None:
            raise IOError('Cannot find mask file %s!'%filename)
        if fieldname is None:
            validkeys=[k for k in f.keys() if not (k.startswith('_') and k.endswith('_'))];
            if len(validkeys)<1:
                raise ValueError('mask file contains no masks!')
            if len(validkeys)>1:
                raise ValueError('mask file contains multiple masks!')
            fieldname=validkeys[0]
        elif fieldname not in f:
            raise ValueError('Mask %s not in the file!'%fieldname)
        self.maskid=fieldname
        self.mask=f[fieldname].astype(np.uint8)
    def write_to_mat(self,filename):
        """Save this mask to a Matlab(R) .mat or a numpy .npy or .npz file.
        """
        if filename.lower().endswith('.mat'):
            scipy.io.savemat(filename,{self.maskid:self.mask})
        elif filename.lower().endswith('.npz'):
            np.savez(filename,**{self.maskid:self.mask})
        elif filename.lower().endswith('.npy'):
            np.save(filename,self.mask)
        else:
            raise ValueError('File name %s not understood (should end with .mat or .npz).'%filename)
    def write_to_hdf5(self,hdf_entity):
        """Write this mask as a HDF5 dataset.
        
        Input:
            hdf_entity: either a HDF5 filename or an open file (instance of
                h5py.highlevel.File) or a HDF5 group (instance of
                h5py.highlevel.Group). A new dataset will be created with the
                name equal to the maskid.
        """
        with _HDF_parse_group(hdf_entity) as hpg:
            if self.maskid in hpg.keys():
                del hpg[self.maskid]
            hpg.create_dataset(self.maskid,data=self.mask,compression='gzip')
    def read_from_hdf5(self,hdf_entity,maskid=None):
        """Read mask from a HDF5 entity.
        
        Inputs:
            hdf_entity: either a HDF5 filename or an open h5py.highlevel.File
                instance or a h5py.highlevel.Group instance.
            maskid: the name of the mask to be loaded from the HDF5 entity.
                If None and the entity contains only one dataset, it will be
                loaded. If None and the entity contains more datasets, a
                ValueError is raised.
        """
        with _HDF_parse_group(hdf_entity) as hpg:
            if len(hpg.keys())==0:
                raise ValueError('No datasets in the HDF5 group!')
            if maskid is None:
                if len(hpg.keys())==1:
                    self.maskid=hpg.keys()[0]
                    self.mask=hpg[self.maskid].value
                else:
                    raise ValueError('More than one datasets in the HDF5 group\
and maskid argument was omitted.')
            else:
                self.maskid=maskid
                self.mask=hpg[maskid].value
        return self
    @classmethod
    def new_from_hdf5(cls,hdf_entity,maskid=None):
        obj=cls()
        obj.read_from_hdf5(hdf_entity,maskid)
        return obj
    def rebin(self,xbin,ybin,enlarge=False):
        """Re-bin the mask."""
        obj=type(self)()
        obj.mask=twodim.rebinmask(self.mask.astype(np.uint8),int(xbin),int(ybin),enlarge)
        obj.maskid=self.maskid+'bin%dx%d_%s'%(xbin,ybin,['shrink','enlarge'][enlarge])
        return obj
    def invert(self):
        """Inverts the whole mask in-place"""
        self.mask=1-self.mask
        return self
    def edit_rectangle(self,x0,y0,x1,y1,whattodo='mask'):
        """Edit a rectangular part of the mask.
        
        Inputs:
            x0,y0,x1,y1: corners of the rectangle (x: row, y: column index).
            whattodo: 'mask', 'unmask' or 'invert' if the selected area should
                be masked, unmasked or inverted. 'mask' is the default.
        """
        col,row=np.meshgrid(np.arange(self.mask.shape[1]),
                            np.arange(self.mask.shape[0]))
        idx=(row>=min(x0,x1))&(row<=max(x0,x1))&(col<=max(y0,y1))&(col>=min(y0,y1))
        if whattodo.lower()=='mask':
            self.mask[idx]=0
        elif whattodo.lower()=='unmask':
            self.mask[idx]=1
        elif whattodo.lower()=='invert':
            self.mask[idx]=1-self.mask[idx]
        else:
            raise ValueError('Invalid value for argument \'whattodo\': '+whattodo)
        return self
    def edit_polygon(self,x,y,whattodo='mask'):
        """Edit points inside a polygon.
        
        Inputs:
            x,y: list of corners of the polygon (x: row, y: column index).
            whattodo: 'mask', 'unmask' or 'invert' if the selected area should
                be masked, unmasked or inverted. 'mask' is the default.
        """
        
        col,row=np.meshgrid(np.arange(self.mask.shape[1]),
                            np.arange(self.mask.shape[0]))
        points = np.vstack((col.flatten(), row.flatten())).T
        Nrows, Ncols = self.shape
        points_inside = matplotlib.nxutils.points_inside_poly(points, np.vstack((y,x)).T)
        idx=points_inside.astype('bool').reshape(self.shape)
        if whattodo.lower()=='mask':
            self.mask[idx]=0
        elif whattodo.lower()=='unmask':
            self.mask[idx]=1
        elif whattodo.lower()=='invert':
            self.mask[idx]=1-self.mask[idx]
        else:
            raise ValueError('Invalid value for argument \'whattodo\': '+whattodo)
        return self

    
    def edit_circle(self,x0,y0,r,whattodo='mask'):
        """Edit a circular part of the mask.
        
        Inputs:
            x0,y0: center of the circle (x0: row, y0: column coordinate)
            r: radius of the circle
            whattodo: 'mask', 'unmask' or 'invert' if the selected area should
                be masked, unmasked or inverted. 'mask' is the default.
        """
        col,row=np.meshgrid(np.arange(self.mask.shape[1]),
                            np.arange(self.mask.shape[0]))
        idx=(((row-x0)**2+(col-y0)**2)<=r**2)
        if whattodo.lower()=='mask':
            self.mask[idx]=0
        elif whattodo.lower()=='unmask':
            self.mask[idx]=1
        elif whattodo.lower()=='invert':
            self.mask[idx]=1-self.mask[idx]
        else:
            raise ValueError('Invalid value for argument \'whattodo\': '+whattodo)
        return self
    def edit_from_matrix(self, matrix, valmin=-np.inf, valmax=np.inf,
                         masknonfinite=True, whattodo='mask'):
        """Edit a part of the mask where the values of a given matrix of the
        same shape are between given thresholds
        
        Inputs:
            matrix: a matrix of the same shape as the mask.
            valmin, valmax: lower and upper threshold of the values in 'matrix'
            masknonfinite: if non-finite elements in the matrix should be masked
            whattodo: 'mask', 'unmask' or 'invert' if the selected area should
                be masked, unmasked or inverted. 'mask' is the default.
        """
        if matrix.shape!=self.mask.shape:
            raise ValueError('Incompatible shape for the matrix!')
        idx=(matrix>=valmin)&(matrix<=valmax)
        self.mask[-np.isfinite(matrix)]=0
        if whattodo.lower()=='mask':
            self.mask[idx]=0
        elif whattodo.lower()=='unmask':
            self.mask[idx]=1
        elif whattodo.lower()=='invert':
            self.mask[idx]=1-self.mask[idx]
        else:
            raise ValueError('Invalid value for argument \'whattodo\': '+whattodo)
        return self
    def edit_borders(self, left=0, right=0, top=0, bottom=0, whattodo='mask'): 
        """Edit borders of the mask.
        
        Inputs:
            left, right, top, bottom: width at the given direction to cut
                (directions correspond to those if the mask matrix is plotted
                by matplotlib.imshow(mask,origin='upper').
            whattodo: 'mask', 'unmask' or 'invert' if the selected area should
                be masked, unmasked or inverted. 'mask' is the default.
        """
        col,row=np.meshgrid(np.arange(self.mask.shape[1]),
                            np.arange(self.mask.shape[0]))
        idx=(col<left)|(col>self.shape[1]-1-right)|(row<top)|(row>self.shape[0]-1-bottom)
        if whattodo.lower()=='mask':
            self.mask[idx]=0
        elif whattodo.lower()=='unmask':
            self.mask[idx]=1
        elif whattodo.lower()=='invert':
            self.mask[idx]=1-self.mask[idx]
        else:
            raise ValueError('Invalid value for argument \'whattodo\': '+whattodo)
        return self
    def spy(self,*args,**kwargs):
        """Plot the mask matrix with matplotlib.pyplot.spy()
        """
        plt.spy(self.mask,*args,**kwargs)
        