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
import numbers
import h5py
import functools
import scipy.io
import os
import warnings

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
            

class SASHeader(collections.defaultdict):
    """A class for holding measurement meta-data."""
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
    _HDF5_read_postprocess_type = [(np.generic, lambda x:x.tolist()), ]
    _HDF5_read_postprocess_name = {'FSNs':lambda x:x.tolist(), 'History':_delinearize_history}
    _key_aliases = {}
    def __init__(self, *args, **kwargs):
        return super(SASHeader, self).__init__(self._default_factory, *args, **kwargs)
    def _default_factory(self, key):
        if key in ['FSNs']:
            return []
        elif key.endswith('Error'):
            return 0
        elif key in ['maskid']:
            return None
        elif key.startswith('FSN'):
            return 0
        else:
            raise KeyError(key)
    def __unicode__(self):
        return "FSN %s; %s; %s mm; %s eV" % (self['FSN'],self['Title'],self['Dist'],self['Energy'])
    __str__ = __unicode__
    def __getitem__(self, key):
        if key in self:
            return super(SASHeader,self).__getitem__(key)
        elif key in self._key_aliases:
            return self.__getitem__(self._key_aliases[key])
        else:
            raise KeyError(key)
    def __setitem__(self, key, value):
        if key in self._key_aliases:
            return self.__setitem__(self._key_aliases[key], value)
        else:
            return super(SASHeader,self).__setitem__(key, value)
    def __delitem__(self, key):
        if key in self:
            return super(SASHeader,self).__delitem__(key)
        elif key in self._key_aliases:
            return self.__delitem__(self._key_aliases[key])
        else:
            raise KeyError(key)
    def keys(self):
        return super(SASHeader,self).keys()+self._key_aliases.keys()
    def read_from_ESRF_ID02(self, filename_or_edf):
        if isinstance(filename_or_edf,basestring):
            filename_or_edf=twodim.readehf(filename_or_edf)
        self.update(filename_or_edf)
        self._key_aliases['FSN']='HMRunNumber'
        self._key_aliases['BeamPosX']='Center_1'
        self._key_aliases['BeamPosY']='Center_2'
        self._key_aliases['MeasTime']='ExposureTime'
        self._key_aliases['Monitor']='Intensity0'
        self._key_aliases['Detector']='DetectorInfo'
        self._key_aliases['Date']='HMStartTime'
        self._key_aliases['Wavelength']='WaveLength'
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
            fid = gzip.GzipFile(filename, 'rt')
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
        """Write the param structure into a logfile. See writelogfile() for an explanation.
        
        Inputs:
            filename: name of the file.
            param: param structure (dictionary).
            
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
                print "Tuple"
                if all([(fn not in allkeys) for fn in fieldnames]):
                    #if all the fields have been processed:
                    print "Bailout"
                    continue
                if isinstance(formatter, tuple) and len(formatter) == len(fieldnames):
                    print "Formatteristuple"
                    formatted = ' '.join([ft(self[fn]) for ft, fn in zip(formatter, fieldnames)])
                #...or a single callable...
                elif not isinstance(formatter, tuple):
                    print "Formatterisnottuple"
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
    def add_history(self, text, time=None):
        """Add a new entry to the history."""
        if time is None:
            time=datetime.datetime.now()
        if 'History' not in self:
            self['History'] = []
        self['History'].append(((time - datetime.datetime.fromtimestamp(0,time.tzinfo)).total_seconds(), text))
    @classmethod
    def new_from_ESRF_ID02(cls,filename):
        inst = cls()
        inst.read_from_ESRF_ID02(filename)
        return inst
    @classmethod
    def new_from_B1_org(cls, filename):
        """Load a header from an org_?????.header file, beamline B1, HASYLAB"""
        inst = cls() # create a new instance
        inst.read_from_B1_org(filename)
        return inst
    @classmethod
    def new_from_B1_log(cls, filename):
        """Load a logfile, beamline B1, HASYLAB"""
        inst = cls()
        inst.read_from_B1_log(filename)
        return inst
    def copy(self, *args, **kwargs):
        """Make a copy of this header structure"""
        d = super(SASHeader, self).copy(*args, **kwargs)
        return SASHeader(d)
    def __iadd__(self, other):
        """Add in-place. The actual work is done by the SASHeader.summarize() classmethod."""
        obj = SASHeader.summarize(self, other)
        for k in obj.keys():
            self[k] = obj[k]
        return self
    def __add__(self, other):
        """Add two headers. The actual work is done by the SASHeader.summarize() classmethod."""
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
    def write_to_hdf5(self, hdf_entity):
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

class SASExposure(object):
    """A class for holding SAS exposure data, i.e. intensity, error, metadata, mask"""
    matrices=collections.OrderedDict([('Image','Detector Image'),
                                      ('Intensity','Corrected intensity'),
                                      ('Error','Error of intensity')])
    _nr_instances=0
    def __init__(self):
        super(SASExposure, self).__init__()
        self.Intensity=None
        self.Error=None
        self.Image=None
        self.header=None
        self.mask=None
        SASExposure._nr_instances+=1
        print "SASExposure instances: ",SASExposure._nr_instances
    def check_for_mask(self):
        if self.mask is None:
            raise ValueError('mask not defined')
        
### -------------- Loading routines (new_from_xyz) ------------------------

    @classmethod
    def new_from_ESRF_ID02(cls,*args,**kwargs):
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
            masks={mn:SASMask.new_from_hdf5(hpg,mn) for mn in masknames}
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
        self.Intensity=edf['data']
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
            othernames=self.matrices
        
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
        if not all([isinstance(e,cls) for e in exps]):
            raise ValueError('All arguments should be SASExposure instances.')
        qranges=[e.get_qrange() for e in exps]
        qmin=max([qr.min() for qr in qranges])
        qmax=min([qr.max() for qr in qranges])
        N=min([len(qr) for qr in qranges])
        return np.linspace(qmin,qmax,N)

    def radial_average(self, qrange = None, pixel=False):
        """Do a radial averaging
        
        Inputs:
            qrange: the q-range. If None, auto-determine.
            pixel: do a pixel-integration (instead of q)
        
        Outputs:
            the one-dimensional curve as an instance of SASCurve (if pixel is
                True) or DataSet (if pixel is False)
        """
        self.check_for_mask()
        q, I, E, A, p = utils2d.integrate.radint(self.Intensity, self.Error,
                                           self.header['EnergyCalibrated'],
                                           self.header['Dist'],
                                           self.header['PixelSize'],
                                           self.header['BeamPosX'],
                                           self.header['BeamPosY'],
                                           1 - self.mask.mask, qrange,
                                           returnavgq = True,
                                           returnpixel = True)
        if pixel:
            ds = dataset.DataSet(p, I, E)
            ds.addfield('q',q)
        else:
            ds = dataset.SASCurve(q, I, E)
            ds.addfield('Pixels', p)
        ds.addfield('Area', A)
        ds.header=SASHeader(self.header)
        return ds

### ---------------------- Plotting -------------------------------------------

    def plot2d(self, zscale = 'log', crosshair = False, drawmask = True, qrange_on_axis = False):
        """Plot the matrix (imshow)
        
        """
        # @todo
        if zscale.upper().startswith('LOG'):
            self._plotmat = np.log10(self.Intensity.copy())
            goodidx = np.isfinite(self._plotmat)
            self._plotmat[-goodidx] = self._plotmat[goodidx].min()

    def imshow(self, *args, **kwargs):
        plt.imshow(np.log10(self.Intensity), *args, **kwargs)

###  ------------------------ Beam center finding -----------------------------

    def update_beampos(self,bc,source=None):
        self.header['BeamPosX'],self.header['BeamPosY']=bc
        if not source:
            self.header.add_history('Beam position updated to:'+str(tuple(bc)))
        else:
            self.header.add_history('Beam found by *%s*: %s'(source,str(tuple(bc))))
    def find_beam_semitransparent(self, bs_area, update=True):
        self.check_for_mask()
        bc=utils2d.centering.findbeam_semitransparent(self.get_matrix(), bs_area)
        if update:
            self.update_beampos(bc, source='semitransparent')
        return bc
    def find_beam_slices(self):
        pass
    def find_beam_radialpeak(self, pixmin, pixmax, drive_by='amplitude', extent=10, update=True, callback=None):
        self.check_for_mask()
        bc=utils2d.centering.findbeam_radialpeak(self.get_matrix(),
                                                 (self.header['BeamPosX'],
                                                  self.header['BeamPosY']),
                                                 self.mask.mask, pixmin,
                                                 pixmax, drive_by=drive_by,
                                                 extent=extent)
        if update:
            self.update_beampos(bc,source='radialpeak')
        return bc
    
### ----------------------- Writing routines ----------------------------------

    def write_to_hdf5(self, hdf_or_filename, **kwargs):
        if 'compression' not in kwargs:
            kwargs['compression'] = 'gzip'
        with _HDF_parse_group(hdf_or_filename) as hpg:
            groupname='FSN%d'%self.header['FSN']
            if groupname in hpg.keys():
                del hpg[groupname]
            hpg.create_group(groupname)
            for k in ['Intensity','Error']:
                hpg[groupname].create_dataset(k,data=self.__getattribute__(k),**kwargs)
            self.header.write_to_hdf5(hpg[groupname])
            if self.mask is not None:
                self.mask.write_to_hdf5(hpg)
    def __del__(self):
        print "Deleting instance of SASExposure"
        del self.Intensity
        del self.Error
        del self.Image
        del self.header
        del self.mask
        SASExposure._nr_instances-=1
        print "SASExposure instances remaining: ",SASExposure._nr_instances
            
class SASMask(object):
    maskid = None
    _mask=None
    def __init__(self, maskmatrix=None):
        super(SASMask,self).__init__()
        if maskmatrix is not None:
            if isinstance(maskmatrix,basestring) and \
                maskmatrix.lower()[-4:] in ['.mat','.npz','.npy']:
                self.read_from_mat(maskmatrix)
            elif isinstance(maskmatrix,basestring) and \
                maskmatrix.lower()[-4:] in ['.edf']:
                self.read_from_edf(maskmatrix)
            elif isinstance(maskmatrix,np.ndarray):
                self.mask=maskmatrix.astype(np.uint8)
                self.maskid='mask'+misc.random_str(6)
            elif isinstance(maskmatrix,SASMask):
                maskmatrix.copy_into(self)
            else:
                raise NotImplementedError
        else:
            self.maskid='mask'+misc.random_str(6)
    def __unicode__(self):
        return u'SASMask('+self.maskid+')'
    __str__=__unicode__
    __repr__=__unicode__
    def _setmask(self, maskmatrix):
        self._mask=maskmatrix.astype(np.uint8)
    def _getmask(self):
        return self._mask
    mask=property(_getmask,_setmask,doc='Mask matrix')
    def copy_into(self,into):
        if not isinstance(into,type(self)):
            raise ValueError('Incompatible class!')
        if self.mask is not None:
            into.mask = self.mask.copy()
        else:
            into.mask = None
        into.maskid = self.maskid
    def read_from_edf(self,filename):
        edf=twodim.readedf(filename)
        self.maskid=os.path.splitext(os.path.split(edf['FileName'])[1])[0]
        self.mask=(np.absolute(edf['data']-edf['Dummy'])>edf['DDummy']).reshape(edf['data'].shape)
        return self
    def read_from_mat(self,filename,fieldname=None):
        """Try to load a maskfile (Matlab(R) matrix file or numpy npz)
        
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
        if filename.lower().endswith('.mat'):
            scipy.io.savemat(filename,{self.maskid:self.mask})
        elif filename.lower().endswith('.npz'):
            np.savez(filename,**{self.maskid:self.mask})
        elif filename.lower().endswith('.npy'):
            np.save(filename,self.mask)
        else:
            raise ValueError('File name %s not understood (should end with .mat or .npz).'%filename)
    def write_to_hdf5(self,hdf_entity):
        with _HDF_parse_group(hdf_entity) as hpg:
            if self.maskid in hpg.keys():
                del hpg[self.maskid]
            hpg.create_dataset(self.maskid,data=self.mask,compression='gzip')
    def read_from_hdf5(self,hdf_entity,maskid=None):
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
        obj=type(self)()
        obj.mask=twodim.rebinmask(self.mask.astype(np.uint8),int(xbin),int(ybin),enlarge)
        obj.maskid=self.maskid+'bin%dx%d_%s'%(xbin,ybin,['shrink','enlarge'][enlarge])
        return obj