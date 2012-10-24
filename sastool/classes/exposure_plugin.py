import re
import os
import h5py
import numpy as np
import warnings

from exposure import SASExposure
from ..io import twodim
from header import SASHeader
from mask import SASMask
from .. import misc
from common import _HDF_parse_group

def register_plugin(pluginclass, idx=None):
    SASExposure.register_IOplugin(pluginclass(), idx)
    return pluginclass

@register_plugin
class SASExposurePlugin(object):
    _isread = False
    _iswrite = False
    _name = '__plugin'
    _default_read_kwargs = {'dirs':[]}
    def read(self, filename, **kwargs):
        raise NotImplementedError
    def write(self, filename, exp, **kwargs):
        raise NotImplementedError
    def check_if_applies(self, filename):
        return False
    def is_read_supported(self):
        return self._isread
    def is_write_supported(self):
        return self._iswrite
    def _before_read(self, kwargs):
        for k in self._default_read_kwargs:
            if k not in kwargs:
                kwargs[k] = self._default_read_kwargs[k]
        for k in SASExposurePlugin._default_read_kwargs:
            if k not in kwargs:
                kwargs[k] = SASExposurePlugin._default_read_kwargs[k]
    def __repr__(self):
        return "<%s SASExposure I/O Plugin>" % self._name
    @property
    def name(self):
        """The name of this plug-in"""
        return self._name
    def __getitem__(self, key):
        if key in self._default_read_kwargs:
            return self._default_read_kwargs[key]
        elif key in SASExposurePlugin._default_read_kwargs:
            return SASExposurePlugin._default_read_kwargs[key]
        else:
            raise KeyError(key)
    def __setitem__(self, key, value):
        self._default_read_kwargs[key] = value
        
@register_plugin
class SEPlugin_ESRF(SASExposurePlugin):
    _isread = True
    _name = 'EDF'
    _default_read_kwargs = {'load_mask':True,
                          'estimate_errors':True}
    def check_if_applies(self, filename):
        if not isinstance(filename, basestring): return False
        filename = os.path.split(filename)[-1]
        return filename.upper().endswith('CCD') or filename.upper().endswith('.EDF')
    def read(self, filename, **kwargs):
        """Read an EDF file (ESRF beamline ID02 SAXS pattern)

        Inputs:
            filename: the name of the file to be loaded
            estimate_errors: error matrices are usually not saved, but they can
                be estimated from the intensity, if they are not present (True
                by default).
            load_mask: try to load the corresponding mask (True by default).
            dirs: folders to look file for.
        """
        self._before_read(kwargs)
        filename = misc.findfileindirs(filename, kwargs['dirs'])
        edf = twodim.readedf(filename)
        header = SASHeader(edf)
        Intensity = edf['data'].astype(np.double)
        if kwargs['load_mask']:
            mask = SASMask(misc.findfileindirs(header['MaskFileName'], kwargs['dirs']))
            if Intensity.shape != mask.shape:
                if all([Intensity.shape[i] > mask.shape[i] for i in [0, 1]]):
                    xbin, ybin = [Intensity.shape[i] / mask.shape[i] for i in [0, 1]]
                    extend = True
                elif all([Intensity.shape[i] < mask.shape[i] for i in [0, 1]]):
                    xbin, ybin = [mask.shape[i] / Intensity.shape[i] for i in [0, 1]]
                    extend = False
                else:
                    raise ValueError('Cannot do simultaneous forward and backward mask binning.')
                warnings.warn('Rebinning mask: %s x %s, direction: %s' % (xbin, ybin, ['shrink', 'enlarge'][extend]))
                mask = mask.rebin(xbin, ybin, extend)
            dummypixels = np.absolute(Intensity - header['Dummy']) <= header['DDummy']
            #self.Intensity[dummypixels]=0
            mask.mask &= (-dummypixels).reshape(Intensity.shape)
        else:
            mask = None
        if kwargs['estimate_errors']:
            sd = edf['SampleDistance']
            ps2 = edf['PSize_1'] * edf['PSize_2']
            I1 = edf['Intensity1']
            Error = (0.5 * sd * sd / ps2 / I1 + Intensity) * float(sd * sd) / (ps2 * I1)
        else:
            Error = None
        return {'header':header, 'Intensity':Intensity, 'mask':mask, 'Error':Error}
    
@register_plugin
class SEPlugin_B1_org(SASExposurePlugin):
    _isread = True
    _name = 'B1 org'
    _default_read_kwargs = {'header_extns':['.header', '.DAT', '.dat', '.DAT.gz', '.dat.gz'],
                          'data_extns':['.cbf', '.tif', '.tiff', '.DAT', '.DAT.gz', '.dat', '.dat.gz'],
                          }
    def check_if_applies(self, filename):
        if not isinstance(filename, basestring): return False
        filename = os.path.split(filename)[-1]
        return filename.upper().startswith('ORG')
    def read(self, filename, **kwargs):
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
        self._before_read(kwargs)
        #try to load header file
        header_extn = [x for x in kwargs['header_extns'] if filename.upper().endswith(x.upper())]
        data_extn = [x for x in kwargs['data_extns'] if filename.upper().endswith(x.upper())]

        # if an extension is found, remove it to get the basename.
        if header_extn + data_extn: # is not empty
            basename = os.path.splitext(filename)[0]
        else:
            basename = filename

        #prepend the already found extension (if any) to the list of possible
        # file extensions, both for header and data.
        header_extn.extend(kwargs['header_extns'])
        data_extn.extend(kwargs['data_extns'])

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
        header = SASHeader(headername)
        if dataname.lower().endswith('.cbf'):
            Intensity = twodim.readcbf(dataname)
        elif dataname.upper().endswith('.DAT') or dataname.upper().endswith('.DAT.GZ'):
            Intensity = twodim.readjusifaorg(dataname).reshape(256, 256)
        elif dataname.upper().endswith('.TIF') or dataname.upper().endswith('.TIFF'):
            Intensity = twodim.readtif(dataname)
        else:
            raise NotImplementedError(dataname)
        if kwargs['estimate_errors']:
            Error = np.sqrt(Intensity)
        else:
            Error = None
        return {'header':header, 'Error':Error, 'Intensity':Intensity, 'mask':None}

@register_plugin    
class SEPlugin_B1_int2dnorm(SASExposurePlugin):
    _isread = True
    _name = 'B1 int2dnorm'
    _default_read_kwargs = {'fileformat':'int2dnorm%d',
                          'logfileformat':'intnorm%d',
                          'logfileextn':'.log',
                          'data_extns':['.npy', '.mat', '.npz'], }
    def check_if_applies(self, filename):
        if not isinstance(filename, basestring): return False
        filename = os.path.split(filename)[-1]
        return filename.upper().startswith('INT2DNORM')
    def read(self, filename, **kwargs):
        self._before_read(kwargs)

        data_extn = [x for x in kwargs['data_extns'] if filename.upper().endswith(x.upper())]

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
        data_extn.extend(kwargs['data_extns'])

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
        header = SASHeader(headername)
        Intensity, Error = twodim.readint2dnorm(dataname)
        header.add_history('Intensity and Error matrices loaded from ' + dataname)
        return {'header':header, 'Intensity':Intensity, 'Error':Error, 'mask':None}

@register_plugin
class SEPlugin_PAXE(SASExposurePlugin):
    _isread = True
    _name = 'PAXE'
    _default_read_kwargs = {'estimate_errors':True}
    def check_if_applies(self, filename):
        if not isinstance(filename, basestring): return False
        filename = os.path.split(filename)[-1]
        return filename.upper().startswith('XE') and any([filename.upper().endswith(x) for x in ['.32', '.DAT']])
    def read(self, filename, **kwargs):
        self._before_read(kwargs)
        paxe = twodim.readPAXE(misc.findfileindirs(filename, dirs=kwargs['dirs']))
        header = SASHeader(paxe[0])
        Intensity = paxe[1]
        if kwargs['estimate_errors']:
            Error = np.sqrt(Intensity)
        else:
            Error = None
        return {'header':header, 'Intensity':Intensity, 'Error':Error, 'mask':None}

@register_plugin
class SEPlugin_BDF(SASExposurePlugin):
    _isread = True
    _name = 'BDF'
    _default_read_kwargs = {'bdfext':'.bdf',
                          'bhfext':'.bhf',
                          'read_corrected_if_present':True,
                          'load_mask':True,
                          }
    def check_if_applies(self, filename):
        if not isinstance(filename, basestring): return False
        filename = os.path.split(filename)[-1]
        return any([filename.upper().endswith(x) for x in ['.BHF', '.BDF']])
    def read(self, filename, **kwargs):
        self._before_read(kwargs)
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
        Intensity = data[Intname]
        Error = data[Errname]
        for dn in datanames:
            del data[dn]
        header = SASHeader(data)
        if kwargs['load_mask'] and 'CORR.Maskfile' in header:
            path, filename = os.path.split(header['CORR.Maskfile'])
            if path:
                dirs = [path] + kwargs['dirs']
            else:
                dirs = kwargs['dirs']
            mask = SASMask(filename, dirs=dirs)
        else:
            mask = None
        return {'Intensity':Intensity, 'Error':Error, 'header':header, 'mask':mask}

@register_plugin
class SEPlugin_MAR(SASExposurePlugin):
    _isread = True
    _name = 'MarCCD'
    _default_read_kwargs = {'estimate_errors':True}
    def check_if_applies(self, filename):
        if not isinstance(filename, basestring): return False
        filename = os.path.split(filename)[-1]
        return filename.upper().endswith('.IMAGE')
    def read(self, filename, **kwargs):
        self._before_read(kwargs)
        Intensity, header = twodim.readmar(misc.findfileindirs(filename, dirs=kwargs['dirs']))
        header = SASHeader(header)
        if kwargs['estimate_errors']:
            Error = np.sqrt(Intensity)
        else:
            Error = None
        return {'Intensity':Intensity, 'Error':Error, 'header':header, 'mask':None}

@register_plugin
class SEPlugin_BerSANS(SASExposurePlugin):
    _isread = True
    _name = 'BerSANS'
    def check_if_applies(self, filename):
        if not isinstance(filename, basestring): return False
        filename = os.path.split(filename)[-1]
        return bool(re.match('.*?D(\d+).(\d+)$', filename))
    def read(self, filename, **kwargs):
        self._before_read(kwargs)
        Intensity, Error, hed = twodim.readBerSANSdata(misc.findfileindirs(filename, dirs=kwargs['dirs']))
        header = SASHeader(hed)
        if Error is None and kwargs['estimate_errors']:
            Error = np.sqrt(Intensity)
        if ('MaskFile' in header) and kwargs['load_mask']:
            dir_ = os.path.split(filename)[0]
            if dir_:
                dirs = [dir_] + kwargs['dirs']
            else:
                dirs = kwargs['dirs']
            mask = SASMask(header['MaskFile'], dirs=dirs)
        else:
            mask = None
        return {'Intensity':Intensity, 'Error':Error, 'header':header, 'mask':mask}

@register_plugin
class SEPlugin_HDF5(SASExposurePlugin):
    _isread = True
    _default_read_kwargs = {'HDF5_Groupnameformat':'FSN%d',
                           'HDF5_Intensityname':'Intensity',
                           'HDF5_Errorname':'Error',
                           'estimate_error':True,
                           'load_mask':True}
    _name = 'HDF5'
    def check_if_applies(self, filename):
        if isinstance(filename, basestring):
            filename = os.path.split(filename)[-1]
            return any([filename.upper().endswith(x) for x in ['H5', 'HDF5']])
        elif isinstance(filename, h5py.highlevel.Group):
            return True
        else:
            return False
        
    def read(self, filename, **kwargs):
        self._before_read(kwargs)
        with _HDF_parse_group(filename) as hdf_group:
            if kwargs['HDF5_Intensityname'] in hdf_group.keys():
                Intensity = hdf_group[kwargs['HDF5_Intensityname']].value
            else:
                raise IOError('No Intensity dataset in HDF5 group ' + hdf_group.filename + ':' + hdf_group.name)
            if kwargs['HDF5_Errorname'] in hdf_group.keys():
                Error = hdf_group[kwargs['HDF5_Errorname']].value
            elif kwargs['estimate_error']:
                Error = np.sqrt(Intensity)
            header = SASHeader(hdf_group)
            if header['maskid'] is not None and kwargs['load_mask']:
                mask = SASMask(hdf_group.parent, maskid=header['maskid'])
            else:
                mask = None
        return {'Intensity':Intensity, 'Error':Error, 'header':header, 'mask':mask}
@register_plugin
class SEPlugin_BareImage(SASExposurePlugin):
    _isread = True
    _name = 'Bare image'
    _default_read_kwargs = {'estimate_errors':True}
    def check_if_applies(self, filename):
        if not isinstance(filename, basestring): return False
        filename = os.path.split(filename)[-1].upper()
        return any([filename.endswith(x) for x in ['.TIFF', '.TIF', '.CBF']])
    def read(self, filename, **kwargs):
        """Read a "bare" image file
        """
        self._before_read(kwargs)
        filename = misc.findfileindirs(filename, kwargs['dirs'])
        if filename.upper().endswith('.CBF'):
            Intensity = twodim.readcbf(filename)
        elif filename.upper().endswith('.TIF') or filename.upper().endswith('.TIFF'):
            Intensity = twodim.readtif(filename)
        if kwargs['estimate_errors']:
            Error = np.sqrt(Intensity)
        else:
            Error = None
        header = SASHeader({'__Origin__':'bare_image'})
        return {'Intensity':Intensity, 'Error':Error, 'header':header, 'mask':None}
    
