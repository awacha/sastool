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

from .. import dataset
from .. import utils2d
from .. import misc
import twodim

#information on how to store the param structure. Each sub-list corresponds to
# a line in the param structure and should be of the form
# [<linestart>,<field name(s)>,<formatter function>,<reader function>]
#
# Required are the first and second.
#
# linestart: the beginning of the line in the file, up to the colon.
#
# field name(s): field name can be a string or a tuple of strings.
#
# formatter function: can be (1) a function accepting a single argument (the 
#     value of the field) or (2) a tuple of functions or (3) None. In the latter
#     case and when omitted, unicode() will be used.
# 
# reader function: can be (1) a function accepting a string and returning as
#     much values as the number of field names is. Or if omitted, unicode() will
#     be used.
logfile_data = [('FSN', 'FSN', None, int),
                  ('FSNs', 'FSNs', lambda l:' '.join([unicode(x) for x in l]),
                   lambda s:[float(x) for x in s.replace(',', ' ').replace(';', ' ').split()]),
                  ('Sample name', 'Title'),
                  ('Sample title', 'Title'),
                  ('Sample-to-detector distance (mm)', 'Dist', None, float),
                  ('Sample thickness (cm)', 'Thickness', None, float),
                  ('Sample transmission', 'Transm', None, float),
                  ('Sample position (mm)', 'PosSample', None, float),
                  ('Temperature', 'Temperature', None, float),
                  ('Measurement time (sec)', 'MeasTime', None, float),
                  ('Scattering on 2D detector (photons/sec)', 'ScatteringFlux',
                   None, float),
                  ('Dark current subtracted (cps)', 'dclevel', None, float),
                  ('Dark current FSN', 'FSNdc', None, int),
                  ('Empty beam FSN', 'FSNempty', None, int),
                  ('Injection between Empty beam and sample measurements?',
                   'InjectionEB', lambda b:['n', 'y'][bool(b)],
                   lambda s:s.upper().startswith('Y')),
                  ('Glassy carbon FSN', 'FSNref1', None, int),
                  ('Glassy carbon thickness (cm)', 'Thicknessref1', None, float),
                  ('Injection between Glassy carbon and sample measurements?',
                   'InjectionGC', lambda b:['n', 'y'][bool(b)],
                   lambda s:s.upper().startswith('Y')),
                  ('Energy (eV)', 'Energy', None, float),
                  ('Calibrated energy (eV)', 'EnergyCalibrated', None, float),
                  ('Calibrated energy', 'EnergyCalibrated', None, float),
                  ('Beam x y for integration', ('BeamPosX', 'BeamPosY'), None,
                   lambda s: tuple([float(x) for x in s.replace(',',' ').replace(';',' ').split()])),
                  ('Normalisation factor (to absolute units)', 'NormFactor',
                   None, float),
                  ('Relative error of normalisation factor (percentage)',
                   'NormFactorRelativeError', None, float),
                  ('Beam size X Y (mm)', ('BeamsizeX', 'BeamsizeY'), None,
                   lambda s: tuple([float(x) for x in s.replace(',',' ').replace(';',' ').split()])),
                  ('Pixel size of 2D detector (mm)', 'PixelSize', None, float),
                  ('Primary intensity at monitor (counts/sec)', 'Monitor', None,
                   float),
                  ('Primary intensity calculated from GC (photons/sec/mm^2)',
                   'PrimaryIntensity', None, float),
                  ('Sample rotation around x axis', 'RotXsample', None, float),
                  ('Sample rotation around y axis', 'RotYsample', None, float),
                 ]

class SASHeader(collections.defaultdict):
    """A class for holding measurement meta-data."""
    #the following define fields treated specially when adding one or more headers together.
    # _fields_to_sum: these fields are to be added. The corresponding 'Error' fields are used for error propagation
    _fields_to_sum=['MeasTime','Anode','Monitor','MonitorDORIS','MonitorPIEZO']
    # _fields_to_average: these fields will be averaged. Error propagation is done.
    _fields_to_average=['Transm','Temperature','BeamPosX','BeamPosY']
    # _fields_to_collect: these will be collected.
    _fields_to_collect={'FSN':'FSNs'}
    # Testing equivalence
    _equiv_tests={'Dist':lambda a,b:(abs(a-b)<1),
                  'Energy':lambda a,b:(abs(a-b)<1),
                  'Temperature':lambda a,b:(abs(a-b)<0.5),
                  'Title':lambda a,b:(a==b),
                  }
    def __init__(self,*args,**kwargs):
        return super(SASHeader,self).__init__(self._default_factory,*args,**kwargs)
    def _default_factory(self,key):
        if key in ['FSNs']:
            return []
        elif key.endswith('Error'):
            return 0
        else:
            raise KeyError(key)
    def __unicode__(self):
        return "FSN {FSN}; {Title}; {Dist} mm; {Energy} eV".format(**self)
    __str__=__unicode__
    def read_from_B1_org(self,filename):
        #Planck's constant times speed of light: incorrect
        # constant in the old program on hasjusi1, which was
        # taken over by the measurement program, to keep
        # compatibility with that.
        jusifaHC=12396.4
        if filename.upper().endswith('.GZ'):
            fid=gzip.GzipFile(filename,'rt')
        else:
            fid=open(filename,'rt')
        lines=fid.readlines()
        fid.close()
        self['FSN']=int(lines[0].strip())
        self['Hour']=int(lines[17].strip())
        self['Minutes']=int(lines[18].strip())
        self['Month']=int(lines[19].strip())
        self['Day']=int(lines[20].strip())
        self['Year']=int(lines[21].strip())+2000
        self['FSNref1']=int(lines[23].strip())
        self['FSNdc']=int(lines[24].strip())
        self['FSNsensitivity']=int(lines[25].strip())
        self['FSNempty']=int(lines[26].strip())
        self['FSNref2']=int(lines[27].strip())
        self['Monitor']=float(lines[31].strip())
        self['Anode']=float(lines[32].strip())
        self['MeasTime']=float(lines[33].strip())
        self['Temperature']=float(lines[34].strip())
        self['BeamPosX']=float(lines[36].strip())
        self['BeamPosY']=float(lines[37].strip())
        self['Transm']=float(lines[41].strip())
        self['Energy']=jusifaHC/float(lines[43].strip())
        self['Dist']=float(lines[46].strip())
        self['XPixel']=1/float(lines[49].strip())
        self['YPixel']=1/float(lines[50].strip())
        self['Title']=lines[53].strip().replace(' ','_').replace('-','_')
        self['MonitorDORIS']=float(lines[56].strip())
        self['Owner']=lines[57].strip()
        self['RotXSample']=float(lines[59].strip())
        self['RotYSample']=float(lines[60].strip())
        self['PosSample']=float(lines[61].strip())
        self['DetPosX']=float(lines[62].strip())
        self['DetPosY']=float(lines[63].strip())
        self['MonitorPIEZO']=float(lines[64].strip())
        self['BeamsizeX']=float(lines[66].strip())
        self['BeamsizeY']=float(lines[67].strip())
        self['PosRef']=float(lines[70].strip())
        self['Monochromator1Rot']=float(lines[77].strip())
        self['Monochromator2Rot']=float(lines[78].strip())
        self['Heidenhain1']=float(lines[79].strip())
        self['Heidenhain2']=float(lines[80].strip())
        self['Current1']=float(lines[81].strip())
        self['Current2']=float(lines[82].strip())
        self['Detector']='Unknown'
        self['PixelSize']=(self['XPixel']+self['YPixel'])/2.0
        
        self['AnodeError']=math.sqrt(self['Anode'])
        self['TransmError']=0
        self['MonitorError']=math.sqrt(self['Monitor'])
        self['MonitorPIEZOError']=math.sqrt(self['MonitorPIEZO'])
        self['MonitorDORISError']=math.sqrt(self['MonitorDORIS'])
        self['Date']=datetime.datetime(self['Year'],self['Month'],self['Day'],self['Hour'],self['Minutes'])
        self.add_history('Loaded from '+filename)
        return self
    def read_from_B1_log(self,filename):
        """Read B1 logfile (*.log)
        
        Inputs:
            filename: the file name
                
        Output: the fields of this header are updated, old fields are kept
            unchanged. The header instance is returned as well.
        """
        fid=open(filename,'r'); #try to open. If this fails, an exception is raised
        for l in fid:
            try:
                ld=[ld for ld in logfile_data if l.split(':')[0].strip()==ld[0]][0]
            except IndexError:
                #line is not recognized.
                continue
            if len(ld)<4:
                reader=unicode
            else:
                reader=ld[3]
            vals=reader(l.split(':')[1].strip())
            if isinstance(ld[1],tuple):
                #more than one field names. The reader function should return a 
                # tuple here, a value for each field.
                if len(vals)!=len(ld[1]):
                    raise ValueError('Cannot read %d values from line %s in file!'%(len(ld[1]),l))
                self.update(dict(zip(ld[1],vals)))
            else:
                self[ld[1]]=vals
        fid.close()
        return self;
    def write_B1_log(self,filename):
        """Write the param structure into a logfile. See writelogfile() for an explanation.
        
        Inputs:
            filename: name of the file.
            param: param structure (dictionary).
            
        Notes:
            exceptions pass through to the caller.
        """
        allkeys=self.keys()
        f=open(filename,'wt')
        for ld in logfile_data: #process each line
            linebegin=ld[0]
            fieldnames=ld[1]
            #set the default formatter if it is not given
            if len(ld)<3:
                formatter=unicode
            elif ld[2] is None:
                formatter=unicode
            else:
                formatter=ld[2]
            #this will contain the formatted values.
            formatted=''
            if isinstance(fieldnames,basestring):
                #scalar field name, just one field. Formatter should be a callable.
                if fieldnames not in allkeys:
                    #this field has already been processed
                    continue
                try:
                    formatted=formatter(self[fieldnames])
                except KeyError:
                    #field not found in param structure
                    continue
            elif isinstance(fieldnames,tuple):
                #more than one field names in a tuple. In this case, formatter can
                # be a tuple of callables...
                if all([(fn not in allkeys) for fn in fieldnames]):
                    #if all the fields have been processed:
                    continue
                if isinstance(formatter,tuple) and len(formatter)==len(fieldnames):
                    formatted=' '.join([ft(self[fn]) for ft,fn in zip(formatter,fieldnames)])
                #...or a single callable...
                elif not isinstance(formatter,tuple):
                    formatted=' '.join([formatter(self[fn]) for fn in fieldnames])
                #...otherwise raise an exception.
                else:
                    raise SyntaxError('Programming error: formatter should be a scalar or a tuple\
of the same length as the field names in logfile_data.')
            else: #fieldnames is neither a string, nor a tuple.
                raise SyntaxError('Invalid syntax (programming error) in logfile_data in writeparamfile().')
            #try to get the values
            f.write(linebegin+':\t'+formatted+'\n')
            if isinstance(fieldnames,tuple):
                for fn in fieldnames: #remove the params treated.
                    if fn in allkeys:
                        allkeys.remove(fn)
            else:
                if fieldnames in allkeys:
                    allkeys.remove(fieldnames)
        #write untreated params
        for k in allkeys:
            f.write(k+':\t'+unicode(self[k])+'\n')
        f.close()
    def add_history(self,text):
        """Add a new entry to the history."""
        if 'History' not in self:
            self['History']=[]
        self['History'].append(((datetime.datetime.now()-datetime.datetime.fromtimestamp(0)).total_seconds(),text))
    @classmethod
    def new_from_B1_org(cls,filename):
        """Load a header from an org_?????.header file, beamline B1, HASYLAB"""
        inst=cls() # create a new instance
        inst.read_from_B1_org(filename)
        return inst
    @classmethod
    def new_from_B1_log(cls,filename):
        """Load a logfile, beamline B1, HASYLAB"""
        inst=cls()
        inst.read_from_B1_log(filename)
        return inst
    def copy(self,*args,**kwargs):
        """Make a copy of this header structure"""
        d=super(SASHeader,self).copy(*args,**kwargs)
        return SASHeader(d)
    def __iadd__(self,other):
        """Add in-place. The actual work is done by the SASHeader.summarize() classmethod."""
        obj=SASHeader.summarize(self,other)
        for k in obj.keys():
            self[k]=obj[k]
        return self
    def __add__(self,other):
        """Add two headers. The actual work is done by the SASHeader.summarize() classmethod."""
        return SASHeader.summarize(self,other)
    @classmethod
    def summarize(cls,*args):
        """Summarize several headers. Calling convention:
        
        summed=SASHeader.summarize(header1,header2,header3,...)
        
        Several fields are treated specially. Values of fields in 
        SASHeader._fields_to_sum are summed (error is calculated by taking the
        corresponding 'Error' fields into account). Values of fields in
        SASHeader._fields_to_average get averaged. Values of fields in
        SASHeader._fields_to_collect get collected to a corresponding list field.
        """
        if any([not isinstance(a,SASHeader) for a in args]):
            raise NotImplementedError('Only SASHeaders can be averaged!')
        obj=cls()
        fields_treated=[]
        allfieldnames=set(reduce(lambda (a,b):a+b, [a.keys() for a in args],[]))
        for k in allfieldnames.intersection(cls._fields_to_sum):
            dk=k+'Error'
            obj[k]=sum([a[k] for a in args])
            obj[dk]=math.sqrt(sum([a[dk]**2 for a in args]))
            fields_treated.extend([k,dk])
        allfieldnames=allfieldnames.difference(fields_treated)
        for k in allfieldnames.intersection(cls._fields_to_average):
            dk=k+'Error'
            sumweight=sum([1/a[dk]**2 for a in args]);
            obj[k]=sum([a[k]/a[dk]**2 for a in args])/sumweight
            obj[dk]=1/math.sqrt(sumweight)
            fields_treated.extend([k,dk])
        allfieldnames=allfieldnames.difference(fields_treated)
        for k in allfieldnames.intersection(cls._fields_to_collect.keys()):
            k_c=cls._fields_to_collect[k]
            obj[k_c]=sum([a[k_c] for a in args])
            obj[k_c].extend([a[k] for a in args])
            obj[k_c]=list(set(obj[k_c]))
            obj[k]=obj[k_c][0]  # take the first one.
            fields_treated.extend([k,k_c])
        allfieldnames=allfieldnames.difference(fields_treated)
        for k in allfieldnames:
            # find the first occurrence of the field
            obj[k]=[a for a in args if k in a.keys()][0][k]
        obj.add_history('Summed from: '+' and '.join([unicode(a) for a in args]))
        return obj
    def isequiv(self,other):
        """Test if the two headers are equivalent. The information found in
        SASHeader._equiv_tests is used to decide equivalence.
        """
        return all([self._equiv_tests[k](self[k],other[k]) for k in self._equiv_tests]+
                   [other._equiv_tests[k](self[k],other[k]) for k in other._equiv_tests])

class SASExposure(object):
    """A class for holding SAS exposure data, i.e. intensity, error, metadata, mask"""
    Intensity=None
    Error=None
    header=None
    mask=None
    def __init__(self):
        super(SASExposure,self).__init__()
    def read_from_B1_int2dnorm(self,fsn,fileformat='int2dnorm%d',logfileformat='intnorm%d.log',dirs=['.']):
        dataname=None
        for extn in ['.npy','.mat']:
            try:
                dataname=misc.findfileindirs(fileformat%fsn+extn,dirs)
            except IOError:
                continue
        if not dataname:
            raise IOError('Cannot find two-dimensional file!')
        headername=misc.findfileindirs(logfileformat%fsn,dirs)
        self.Intensity,self.Error=twodim.readint2dnorm(dataname)
        self.header=SASHeader.new_from_B1_log(headername)
        return self
    def set_mask(self,mask):
        self.mask=mask.astype(np.uint8)
    def get_qrange(self,N=None,spacing='linear'):
        qrange=utils2d.integrate.autoqscale(self.header['EnergyCalibrated'],
                                            self.header['Dist'],
                                            self.header['PixelSize'],
                                            self.header['PixelSize'],
                                            self.header['BeamPosX'],
                                            self.header['BeamPosY'],1-self.mask)
        if N is None:
            return qrange
        elif isinstance(N,numbers.Integral):
            if spacing.upper().startswith('LIN'):
                return np.linspace(qrange.min(),qrange.max(),N)
            elif spacing.upper().startswith('LOG'):
                return np.logspace(np.log10(qrange.min()),np.log10(qrange.max()),N)
        elif isinstance(N,numbers.Real):
            return np.arange(qrange.min(),qrange.max(),N)
        else:
            raise NotImplementedError
    def radial_average(self,qrange=None):
        q,I,E,A,p=utils2d.integrate.radint(self.Intensity,self.Error,
                                           self.header['EnergyCalibrated'],
                                           self.header['Dist'],
                                           self.header['PixelSize'],
                                           self.header['BeamPosX'],
                                           self.header['BeamPosY'],
                                           1-self.mask, qrange,
                                           returnavgq=True,
                                           returnpixel=True)
        ds=dataset.SASCurve(q,I,E)
        ds.addfield('Area',A)
        ds.addfield('Pixels',p)
        return ds
    def imshow(self,*args,**kwargs):
        plt.imshow(np.log10(self.Intensity),*args,**kwargs)
    @classmethod
    def new_from_B1_int2dnorm(cls,*args,**kwargs):
        obj=cls()
        obj.read_from_B1_int2dnorm(*args,**kwargs)
        return obj
