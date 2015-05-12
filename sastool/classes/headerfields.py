import datetime
from .history import SASHistory
import collections

class SASHeaderFieldLink(object):
    def __init__(self, tofield, strong=False):
        self.tofield = tofield
        self.strong = strong

class SASHeaderFieldCollectMode(object):
    FIRST = 0
    SUM = 1
    AVERAGE = 2
    LIST = 3

class SASHeaderField(object):
    _allknownfields = []
    type_ = None
    fieldname = None
    mnemonic = None
    minimum = None
    maximum = None
    default = None
    can_calibrated = False
    can_error = False
    unit = None
    custom_validator = None
    custom_fromstring = None
    custom_tostring = None
    collect_mode = SASHeaderFieldCollectMode.FIRST
    def __init__(self, **kwargs):
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        if [x for x in SASHeaderField._allknownfields if x.fieldname == self.fieldname]:
            raise ValueError('Cannot create new field: another one with this name already exists.')
        SASHeaderField._allknownfields.append(self)
        if self.can_calibrated:
            SASHeaderField(type_=self.type_, fieldname=self.fieldname + 'Calibrated', mnemonic='Calibrated ' + self.mnemonic,
                           minimum=self.minimum, maximum=self.maximum, default=self.default, can_calibrated=False,
                           can_error=self.can_error, unit=self.unit, custom_validator=self.custom_validator,
                           custom_fromstring=self.custom_fromstring, custom_tostring=self.custom_tostring,
                           collect_mode=self.collect_mode)
        if self.can_error:
            SASHeaderField(type_=self.type_, fieldname=self.fieldname + 'Error', mnemonic='Absolute error of ' + self.mnemonic,
                           minimum=0, maximum=None, default=0, can_error=False,
                           can_calibrated=False, unit=self.unit, custom_validator=self.custom_validator,
                           custom_fromstring=self.custom_fromstring, custom_tostring=self.custom_tostring,
                           collect_mode=self.collect_mode)
                           
            
    def validate(self, value):
        if not isinstance(value, self.type_):
            raise ValueError('Invalid type for field %s: %s' % (self.fieldname, str(type(value))))
        if (minimum is not None) and (value < minimum):
            raise ValueError('Value is lower than the minimum for field %s.' % self.fieldname)
        if (maximum is not None) and (value > maximum):
            raise ValueError('Value is higher than the maximum for field %s.' % self.fieldname)
        if isinstance(self.custom_validator, collections.Callable) and not self.custom_validator(value):
            raise ValueError('Custom validator for field %s not matched.' % self.fieldname)
        return True

    def fromstring(self, string):
        if isinstance(self.custom_fromstring, collections.Callable):
            return self.custom_fromstring(string)
        else:
            return self.type_(string)
    
    def tostring(self, value):
        if isinstance(self.custom_tostring, collections.Callable):
            return self.custom_tostring(value)
        else:
            return str(value)

    def get(self, header):
        try:
            return header[self.fieldname]
        except KeyError:
            if isinstance(self.default, SASHeaderFieldLink):
                fieldname = [x for x in SASHeaderField._allknownfields if x.fieldname == self.default.tofield]
                if not fieldname:
                    raise ValueError('Default value of field %s points to unknown field %s.' % (self.fieldname, self.default.tofield))
                else:
                    return fieldname[0].get(header)
            elif isinstance(self.default, self.type_):
                return self.default
            else:
                raise ValueError('No default for field %s.' % self.fieldname)
    
    def set(self, header, value, validate=True):
        if isinstance(self.default, SASHeaderFieldLink) and self.default.strong:
            fieldname = [x for x in  SASHeaderField._allknownfields if x.fieldname == self.default.tofield]
            if not fieldname:
                raise ValueError('Default value of field %s points to unknown field %s.' % (self.fieldname, self.default.tofield))
            else:
                return fieldname[0].set(header, value, validate)
        else:
            if validate:
                self.validate(value)
            header[self.fieldname] = value
    
    @classmethod
    def __iter__(cls):
        return iter(cls._allknownfields)
    
    @classmethod
    def get_instance(cls, name):
        try:
            return [x for x in cls._allknownfields if x.fieldname == name][0]
        except IndexError:
            raise ValueError('Unknown SAS field: %s' % str(name))
    
SASHeaderField(type_=str,
               fieldname='__Origin__',
               mnemonic='String to uniquely identify the experiment type',
               default='<Unknown>'
               )

SASHeaderField(type_=str,
               fieldname='__particle__',
               mnemonic='The probe particle',
               default='<Unknown>',
               custom_validator=lambda x:x in ['photon', 'neutron', '<Unknown>'],
               )

SASHeaderField(type_=int,
               fieldname='FSN',
               mnemonic='File sequence number',
               minimum=0,
               default=0,
               collect_mode=SASHeaderFieldCollectMode.LIST)

SASHeaderField(type_=float,
               fieldname='BeamPosX',
               mnemonic='Row coordinate of the beam position',
               default=0,
               unit='pixel',
               can_error=True,
               collect_mode=SASHeaderFieldCollectMode.AVERAGE)

SASHeaderField(type_=float,
               fieldname='BeamPosY',
               mnemonic='Column coordinate of the beam position',
               default=0,
               unit='pixel',
               can_error=True,
               collect_mode=SASHeaderFieldCollectMode.AVERAGE)

SASHeaderField(type_=datetime.datetime,
               fieldname='Date',
               mnemonic='Date of the measurement',
               default=datetime.datetime.now(),
               collect_mode=SASHeaderFieldCollectMode.LIST
               )

SASHeaderField(type_=datetime.datetime,
               fieldname='StartDate',
               mnemonic='Start date of the measurement',
               default=datetime.datetime.now(),
               collect_mode=SASHeaderFieldCollectMode.LIST
               )

SASHeaderField(type_=datetime.datetime,
               fieldname='EndDate',
               mnemonic='End date of the measurement',
               default=datetime.datetime.now(),
               collect_mode=SASHeaderFieldCollectMode.LIST
               )

SASHeaderField(type_=float,
               fieldname='Dist',
               mnemonic='Sample-to-detector distance',
               default=0,
               minimum=0,
               unit='mm',
               can_error=True,
               can_calibrated=True,
               collect_mode=SASHeaderFieldCollectMode.LIST,
               custom_tostring=lambda x:'%.2f' % x)

SASHeaderField(type_=float,
               fieldname='Energy',
               mnemonic='Beam energy',
               default=0,
               minimum=0,
               unit='eV',
               can_error=True,
               can_calibrated=True,
               collect_mode=SASHeaderFieldCollectMode.LIST,
               custom_tostring=lambda x:'%.2f' % x)

SASHeaderField(type_=float,
               fieldname='MeasTime',
               mnemonic='Exposure time',
               default=0,
               minimum=0,
               unit='sec',
               can_error=True,
               can_calibrated=False,
               collect_mode=SASHeaderFieldCollectMode.SUM)

SASHeaderField(type_=float,
               fieldname='Monitor',
               mnemonic='Monitor counts',
               default=0,
               minimum=0,
               unit='counts',
               can_error=True,
               can_calibrated=True,
               collect_mode=SASHeaderFieldCollectMode.SUM)

SASHeaderField(type_=float,
               fieldname='PixelSize',
               mnemonic='Pixel size for square pixeled detectors',
               default=0,
               minimum=0,
               unit='mm',
               can_error=True,
               can_calibrated=False)

SASHeaderField(type_=float,
               fieldname='Thickness',
               mnemonic='Thickness of the sample',
               default=1,
               minimum=0,
               unit='cm',
               can_error=True,
               can_calibrated=False,
               collect_mode=SASHeaderFieldCollectMode.AVERAGE)

SASHeaderField(type_=str,
               fieldname='Title',
               mnemonic='Sample name',
               default='<Unknown sample>',
               collect_mode=SASHeaderFieldCollectMode.LIST)

SASHeaderField(type_=float,
               fieldname='Transm',
               mnemonic='Transmission',
               default=1,
               minimum=0,
               maximum=1,
               can_error=True,
               can_calibrated=False,
               collect_mode=SASHeaderFieldCollectMode.AVERAGE)

SASHeaderField(type_=float,
               fieldname='Wavelength',
               mnemonic='Wavelength',
               default=1,
               minimum=0,
               unit='nm',
               can_error=True,
               can_calibrated=True,
               collect_mode=SASHeaderFieldCollectMode.LIST)

SASHeaderField(type_=float,
               fieldname='XPixel',
               mnemonic='Pixel size in the row direction',
               default=SASHeaderFieldLink('PixelSize'),
               minimum=0,
               unit='mm',
               can_error=True,
               can_calibrated=False)

SASHeaderField(type_=float,
               fieldname='YPixel',
               mnemonic='Pixel size in the column direction',
               default=SASHeaderFieldLink('PixelSize'),
               minimum=0,
               unit='mm',
               can_error=True,
               can_calibrated=False)

SASHeaderField(type_=float,
               fieldname='Anode',
               mnemonic='Counts on the anode of the gas detector',
               default=0,
               minimum=0,
               unit='counts',
               can_error=True,
               can_calibrated=False,
               collect_mode=SASHeaderFieldCollectMode.SUM)

SASHeaderField(type_=float,
               fieldname='BeamsizeX',
               mnemonic='Beam size in the row direction',
               default=0,
               minimum=0,
               unit='mm',
               collect_mode=SASHeaderFieldCollectMode.AVERAGE)

SASHeaderField(type_=float,
               fieldname='BeamsizeY',
               mnemonic='Beam size in the column direction',
               default=0,
               minimum=0,
               unit='mm',
               collect_mode=SASHeaderFieldCollectMode.AVERAGE)

SASHeaderField(type_=str,
               fieldname='Owner',
               mnemonic='Owner of the measurement',
               default='<Nobody>',
               collect_mode=SASHeaderFieldCollectMode.LIST)

SASHeaderField(type_=float,
               fieldname='Temperature',
               mnemonic='Sample temperature',
               default=0,
               unit='°C',
               collect_mode=SASHeaderFieldCollectMode.LIST)

SASHeaderField(type_=float,
               fieldname='NormFactor',
               mnemonic='Intensity normalization factor',
               default=1,
               unit='',
               can_error=True,
               collect_mode=SASHeaderFieldCollectMode.AVERAGE)

SASHeaderField(type_=SASHistory,
               fieldname='History',
               mnemonic='History of the exposure',
               custom_tostring=lambda x:x.linearize())

SASHeaderField(type_=list,
               fieldname='FSNs',
               mnemonic='List of FSNs',
               default=[],
               custom_validator=lambda lis:all(isinstance(x, int) for x in lis),
               custom_fromstring=lambda stri:[int(x) for x in stri.replace(',', '').replace(';', '').split()],
               custom_tostring=lambda lis:' '.join([str(x) for x in lis]),
               collect_mode=SASHeaderFieldCollectMode.SUM)

SASHeaderField(type_=str,
               fieldname='maskid',
               mnemonic='Mask ID',
               default='mask')

