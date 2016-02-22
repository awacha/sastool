""" exposure_plugin.py

Created 25.10.2012. Andras Wacha

This module defines the I/O plugin ecosystem for the SASExposure class, which enables
the user to extend the read/write functionality of SASExposure.

Each plugin should be a descendant of the general SASExposurePlugin class.
Using the decorator ``register_plugin`` on the sub-class registers the plugin with
SASExposure.

For the actual working method of a plugin and for a short guide on what 
methods/attributes to override in order to implement a new plugin, see the 
docstring of SASExposurePlugin.

"""


import os
import h5py
import numpy as np
import warnings
import re
import scipy.io
import datetime
import sys


__all__ = []

from .exposure import SASExposure
from ..io import twodim
from .header import SASHeader
from .mask import SASMask
from .. import misc
from .common import _HDF_parse_group


def register_plugin(pluginclass, idx=None):
    SASExposure.register_IOplugin(pluginclass(), idx)
    return pluginclass


class SASExposurePlugin(object):
    """Base class for SASExposure I/O plugins. For an actually working plugin, the
    following attributes and methods can/need to be overridden:

    Attributes
    ----------

    `_isread` bool, if reading is supported
    `_iswrite` bool, if writing is supported
    `_name` string, the name of the plugin. This will be used as an `experiment_type`
        in SASExposure
    `_default_read_kwargs` dict, the default values of keyword arguments you are
        relying on if loading a file by the `read()` or `read_multi()` methods. Inside
        those methods you would like to first call _before_read(kwargs), which updates
        the keyword argument dict received originally by the reader methods (note the
        absence of ** in _before_read(kwargs) !!!). Note that from this attribute both
        the version optionall defined by the subclass and the one defined by this class
        is used, in that order.
    `_filename_regex` a compiled regular expression (by `re.compile()`) which should
        match the filename. This is used by the default version of `check_if_applicable`
        to decide if a filename is eligible for this plugin to work with. You should
        end your regex with a '$' sign as the regex will be `search()`-ed, and you should
        supply the `re.IGNORECASE` flag in `re.compile()`

    Methods
    -------

    `read(filename, **kwargs)`: this method does the actual reading, so you should
        override it if you are implementing a new plug-in. Its input
        is usually a filename, which should be further searched for by
        `sastool.misc.findfileindirs`, giving kwargs['dirs'] as an argument.
        If you need other keyword arguments, please define default values for them
        in _default_read_kwargs. You should call _before_read(kwargs) first thing.
        This function should return a dict with the fields 'Intensity', 'Error',
        'header', 'mask', from which the latter three can be None. A generator
        returning such dicts can also be returned. The header should be loaded
        by calling SASHeader(...) with the appropriate arguments. It is a good
        practice to forward all kwargs to the call to SASHeader()

    `read_multi(filenameformat, fsns, **kwargs)`: this should be a wrapper around
        `read()`, which should not normally be overridden, except if you want to
        use fancy indexing (e.g. the first argument can be a non-string, e.g. a
        HDF5 File or Group). This function should consist of a loop over fsns (which
        is guaranteed to be an iterable), and `yield`-ing results in the same format
        as `read()` does. Thus this always returns a generator!

    `write(filename,exp,**kwargs)`: write the exposure to a file (or other object).
        If you define this, you also have to set `_iswrite` to True.

    `check_if_applies(filename)`: this function checks if the given filename (or other
        object) can be treated by this plugin (either read or written). The default
        version simply checks the given filename with `_filename_regex`. If however
        you would like to load from non-files, you have to override this function.
        It should return True or False.

    `is_read_supported()` and `is_write_supported()`: self-explanatory functions

    Item getting and setting with dict-like indexing is also supported: this operates
    on the default kwargs in _default_read_kwargs.

    If you make a new plug-in, do not forget to apply the decorator `register_plugin`
    in order to make it known to SASExposure.
    """
    _isread = False
    _iswrite = False
    _name = '__plugin'
    _default_read_kwargs = {
        'dirs': '.', 'error_on_not_found': True, 'noheader': False}
    _filename_regex = None

    def read(self, filename, **kwargs):
        raise NotImplementedError

    def read_multi(self, filenameformat, fsns, **kwargs):
        self._before_read(kwargs)
        for f in fsns:
            try:
                yield self.read(filenameformat % f, **kwargs)
            except IOError:
                if kwargs['error_on_not_found']:
                    raise
                else:
                    continue
        return

    def write(self, filename, exp, **kwargs):
        raise NotImplementedError

    def check_if_applies(self, filename):
        if not isinstance(filename, str):
            return False
        if self._filename_regex is not None:
            return (self._filename_regex.search(filename) is not None)
        else:
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
        if 'dirs' in kwargs:
            if isinstance(kwargs['dirs'], str):
                kwargs['dirs'] = [kwargs['dirs']]
            else:
                kwargs['dirs'] = list(kwargs['dirs'])

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
    """SASExposure I/O plugin for EDF files (ID01 and ID02 at ESRF)"""
    _isread = True
    _name = 'EDF'
    _default_read_kwargs = {'load_mask': True,
                            'estimate_errors': True}
    _filename_regex = re.compile('(ccd|.edf)$', re.IGNORECASE)

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
        if kwargs['noheader']:
            header = SASHeader()
        else:
            header = SASHeader(edf, **kwargs)
        Intensity = edf['data'].astype(np.double)
        if kwargs['load_mask']:
            mask = SASMask(
                misc.findfileindirs(header['MaskFileName'], kwargs['dirs']))
            if Intensity.shape != mask.shape:
                if all([Intensity.shape[i] > mask.shape[i] for i in [0, 1]]):
                    xbin, ybin = [Intensity.shape[i] / mask.shape[i]
                                  for i in [0, 1]]
                    extend = True
                elif all([Intensity.shape[i] < mask.shape[i] for i in [0, 1]]):
                    xbin, ybin = [mask.shape[i] / Intensity.shape[i]
                                  for i in [0, 1]]
                    extend = False
                else:
                    raise ValueError(
                        'Cannot do simultaneous forward and backward mask binning.')
                warnings.warn('Rebinning mask: %s x %s, direction: %s' % (
                    xbin, ybin, ['shrink', 'enlarge'][extend]))
                mask = mask.rebin(xbin, ybin, extend)
            dummypixels = np.absolute(
                Intensity - header['Dummy']) <= header['DDummy']
            # self.Intensity[dummypixels]=0
            mask.mask &= (-dummypixels).reshape(Intensity.shape)
        else:
            mask = None
        if kwargs['estimate_errors']:
            sd = edf['SampleDistance']
            ps2 = edf['PSize_1'] * edf['PSize_2']
            I1 = edf['Intensity1']
            Error = (0.5 * sd * sd / ps2 / I1 + Intensity) * \
                float(sd * sd) / (ps2 * I1)
        else:
            Error = None
        header['FileName'] = filename
        return {'header': header, 'Intensity': Intensity, 'mask': mask, 'Error': Error}


@register_plugin
class SEPlugin_B1_org(SASExposurePlugin):
    """SASExposure I/O plugin for B1 (HASYLAB, DORIS III) original measurement data."""
    _isread = True
    _name = 'B1 org'
    _default_read_kwargs = {'header_extns': ['.header', '.DAT.gz', '.dat.gz', '.DAT', '.dat'],
                            'data_extns': ['.cbf', '.tif', '.tiff', '.DAT.gz', '.DAT', '.dat.gz', '.dat'],
                            'estimate_errors': True}
    _filename_regex = re.compile(r'org_?[^/\\]*\.[^/\\]*$', re.IGNORECASE)

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
        # try to load header file
        header_extn = [x for x in kwargs['header_extns']
                       if filename.upper().endswith(x.upper())]
        data_extn = [x for x in kwargs['data_extns']
                     if filename.upper().endswith(x.upper())]

        # if an extension is found, remove it to get the basename.
        basename = filename
        for x in header_extn + data_extn:
            if filename.upper().endswith(x.upper()):
                basename = filename[:-len(x)]
                break
        basename = os.path.basename(basename)
        # prepend the already found extension (if any) to the list of possible
        # file extensions, both for header and data.
        header_extn.extend(kwargs['header_extns'])
        data_extn.extend(kwargs['data_extns'])

        headername = ''

        for extn in header_extn:
            try:
                headername = misc.findfileindirs(
                    basename + extn, kwargs['dirs'])
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
            raise IOError('Could not find 2d org file')  # skip this file
        header = SASHeader(headername, **kwargs)
        header_loaded = {}
        if dataname.lower().endswith('.cbf'):
            Intensity, header_loaded = twodim.readcbf(
                dataname, load_header=True)
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
        header_loaded = SASHeader(header_loaded)
        header_loaded.update(header)
        header_loaded['FileName'] = dataname
        mask = None
        if 'maskid' in header_loaded and header_loaded['maskid'] is not None:
            maskbasename = os.path.basename(header_loaded['maskid'])
            maskdir = os.path.dirname(header_loaded['maskid'])
            for maskext in [''] + SASMask.supported_read_extensions:
                try:
                    maskname = misc.findfileindirs(
                        maskbasename + maskext, [maskdir] + kwargs['dirs'])
                except IOError:
                    continue
                mask = SASMask(maskname)
                break
            if mask is None:
                warnings.warn(
                    'Could not load mask file specified in the header (%s).' % header_loaded['maskid'])
        return {'header': header_loaded, 'Error': Error, 'Intensity': Intensity, 'mask': mask}


@register_plugin
class SEPlugin_CREDO(SASExposurePlugin):
    """SASExposure I/O plugin for the CREDO instrument."""
    _isread = True
    _name = 'CREDO Raw'
    _default_read_kwargs = {'header_extns': ['.pickle','.param'],
                            'data_extns': ['.cbf', '.tif'],
                            'estimate_errors': True}
    _filename_regex = re.compile(r'_((\d+)|%(\d+)d).(cbf|tif)$', re.IGNORECASE)

    def read(self, filename, **kwargs):
        """Read an original exposition (CREDO)

        Inputs:
            filename: the name of the file to be loaded
            dirs: folders to look for files in
        """
        self._before_read(kwargs)
        # try to load header file
        header_extn = [x for x in kwargs['header_extns']
                       if filename.upper().endswith(x.upper())]
        data_extn = [x for x in kwargs['data_extns']
                     if filename.upper().endswith(x.upper())]

        # if an extension is found, remove it to get the basename.
        basename = filename
        for x in header_extn + data_extn:
            if filename.upper().endswith(x.upper()):
                basename = filename[:-len(x)]
                break
        basename = os.path.basename(basename)
        # prepend the already found extension (if any) to the list of possible
        # file extensions, both for header and data.
        header_extn.extend(kwargs['header_extns'])
        data_extn.extend(kwargs['data_extns'])

        if not kwargs['noheader']:
            headername = ''

            for extn in header_extn:
                try:
                    headername = misc.findfileindirs(
                        basename + extn, kwargs['dirs'])
                except IOError:
                    continue
            if not headername:
                raise IOError(
                    'Could not find param file. Was looking for: %s+extn where extn is in %s.' % (basename, str(header_extn)))
        dataname = ''
        for extn in data_extn:
            try:
                dataname = misc.findfileindirs(basename + extn, kwargs['dirs'])
            except IOError:
                continue
        if not dataname:
            raise IOError('Could not find 2d crd file')  # skip this file
        if not kwargs['noheader']:
            header = SASHeader(headername, **kwargs)
        else:
            header = SASHeader()
        if dataname.lower().endswith('.cbf'):
            Intensity, header_loaded = twodim.readcbf(
                dataname, load_header=True)
        elif dataname.upper().endswith('.TIF') or dataname.upper().endswith('.TIFF'):
            Intensity = twodim.readtif(dataname)
            header_loaded = {}
        else:
            raise NotImplementedError(dataname)
        if kwargs['estimate_errors']:
            Error = np.sqrt(Intensity)
        else:
            Error = None
        header_loaded = SASHeader(header_loaded)
        header_loaded.update(header)
        header_loaded['FileName'] = dataname
        mask = None
        if 'maskid' in header_loaded and header_loaded['maskid'] is not None:
            maskbasename = os.path.basename(header_loaded['maskid'])
            maskdir = os.path.dirname(header_loaded['maskid'])
            for maskext in SASMask.supported_read_extensions + ['']:
                try:
                    maskname = misc.findfileindirs(
                        maskbasename + maskext, [maskdir] + kwargs['dirs'])
                except IOError:
                    continue
                mask = SASMask(maskname)
                break
            if mask is None:
                warnings.warn(
                    'Could not load mask file specified in the header (%s).' % header_loaded['maskid'])
        return {'header': header_loaded, 'Error': Error, 'Intensity': Intensity, 'mask': mask}


@register_plugin
class SEPlugin_B1_int2dnorm(SASExposurePlugin):
    """SASExposure I/O plugin for B1 (HASYLAB, DORISIII) reduced data."""
    _isread = True
    _iswrite = True
    _name = 'B1 int2dnorm'
    _default_read_kwargs = {'fileformat': 'int2dnorm%d',
                            'logfileformat': 'intnorm%d',
                            'logfileextn': '.log',
                            'data_extns': ['.npy', '.mat', '.npz'], }
    _filename_regex = re.compile(
        r'int2dnorm[^/\\]*\.(mat|npy|npz)$', re.IGNORECASE)

    def read(self, filename, **kwargs):
        self._before_read(kwargs)

        data_extn = [x for x in kwargs['data_extns']
                     if filename.upper().endswith(x.upper())]

        if os.path.isabs(filename):
            if kwargs['dirs'] is None:
                kwargs['dirs'] = []
            elif isinstance(kwargs['dirs'], str):
                kwargs['dirs'] = [kwargs['dirs']]
            kwargs['dirs'] = [os.path.split(filename)[0]] + kwargs['dirs']

        if data_extn:  # is not empty
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
        m = re.match(kwargs['fileformat'].replace(
            '%d', r'(\d+)'), os.path.split(basename)[1])
        if m is None:
            raise ValueError('Filename %s does not have the format %s, \
therefore the FSN cannot be determined.' % (dataname, kwargs['fileformat']))
        else:
            fsn = int(m.group(1))

        headername = misc.findfileindirs(
            kwargs['logfileformat'] % fsn + kwargs['logfileextn'], kwargs['dirs'])
        header = SASHeader(headername, **kwargs)
        Intensity, Error = twodim.readint2dnorm(dataname)
        header.add_history(
            'Intensity and Error matrices loaded from ' + dataname)
        header['FileName'] = dataname
        mask = None
        if 'maskid' in header and header['maskid'] is not None:
            maskbasename = os.path.basename(header['maskid'])
            maskdir = os.path.dirname(header['maskid'])
            for maskext in [''] + SASMask.supported_read_extensions:
                try:
                    maskname = misc.findfileindirs(
                        maskbasename + maskext, [maskdir] + kwargs['dirs'])
                except IOError:
                    continue
                mask = SASMask(maskname)
                break
            if mask is None:
                warnings.warn(
                    'Could not load mask file specified in the header (%s).' % header['maskid'])
        return {'header': header, 'Intensity': Intensity, 'Error': Error, 'mask': mask}

    def write(self, filename, ex, **kwargs):
        self._before_read(kwargs)
        folder = os.path.split(filename)[0]
        scipy.io.savemat(
            filename, {'Intensity': ex.Intensity, 'Error': ex.Error})
        ex.header.write(os.path.join(
            folder, kwargs['logfileformat'] % ex.header['FSN'] + kwargs['logfileextn']))


@register_plugin
class SEPlugin_CREDO_Reduced(SASExposurePlugin):
    """SASExposure I/O plugin for CREDO reduced data."""
    _isread = True
    _iswrite = True
    _name = 'CREDO Reduced'
    _default_read_kwargs = {'fileformat': 'crd_%d',
                            'logfileformat': 'crd_%d',
                            'logfileextn': '.log',
                            'data_extns': ['.npz'], }
    _filename_regex = re.compile(r'.*crd[^/\\]*\.(npz)$', re.IGNORECASE)

    def read(self, filename, **kwargs):
        self._before_read(kwargs)

        data_extn = [x for x in kwargs['data_extns']
                     if filename.upper().endswith(x.upper())]

        if os.path.isabs(filename):
            if kwargs['dirs'] is None:
                kwargs['dirs'] = []
            elif isinstance(kwargs['dirs'], str):
                kwargs['dirs'] = [kwargs['dirs']]
            kwargs['dirs'] = [os.path.split(filename)[0]] + kwargs['dirs']

        if data_extn:  # is not empty
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
        m = re.match('.*crd_(\\d+)', os.path.split(basename)[1])
        if m is None:
            raise ValueError('Filename %s does not have the format %s, \
therefore the FSN cannot be determined.' % (os.path.split(basename)[1], kwargs['fileformat']))
        else:
            fsn = int(m.group(1))
        try:
            headername = misc.findfileindirs(
                kwargs['logfileformat'] % fsn + kwargs['logfileextn'], kwargs['dirs'])
        except OSError:
            headername=misc.findfileindirs(
                dataname.split('/')[-1].rsplit('.',1)[0]+'.param', kwargs['dirs'])
        header = SASHeader(headername, **kwargs)
        Intensity, Error = twodim.readint2dnorm(dataname)
        header.add_history(
            'Intensity and Error matrices loaded from ' + dataname)
        header['FileName'] = dataname
        mask = None
        if 'maskid' in header and header['maskid'] is not None:
            maskbasename = os.path.basename(header['maskid'])
            maskdir = os.path.dirname(header['maskid'])
            for maskext in [''] + SASMask.supported_read_extensions:
                try:
                    maskname = misc.findfileindirs(
                        maskbasename + maskext, [maskdir] + kwargs['dirs'])
                except IOError:
                    continue
                mask = SASMask(maskname)
                break
            if mask is None:
                warnings.warn(
                    'Could not load mask file specified in the header (%s).' % header['maskid'])
        return {'header': header, 'Intensity': Intensity, 'Error': Error, 'mask': mask}

    def write(self, filename, ex, **kwargs):
        self._before_read(kwargs)
        folder = os.path.split(filename)[0]
        np.savez_compressed(filename, Intensity=ex.Intensity, Error=ex.Error)
        ex.header.write(os.path.join(
            folder, kwargs['logfileformat'] % ex.header['FSN'] + kwargs['logfileextn']))


@register_plugin
class SEPlugin_PAXE(SASExposurePlugin):
    """SASExposure I/O plugin for LLB PAXE (or BNC Yellow Submarine) SANS data."""
    _isread = True
    _name = 'PAXE'
    _default_read_kwargs = {'estimate_errors': True}
    _filename_regex = re.compile(r'xe[^/\\]*\.(dat|32)$', re.IGNORECASE)

    def read(self, filename, **kwargs):
        self._before_read(kwargs)
        filename = misc.findfileindirs(filename, dirs=kwargs['dirs'])
        paxe = twodim.readPAXE(filename)
        header = SASHeader(paxe[0], **kwargs)
        Intensity = paxe[1]
        if kwargs['estimate_errors']:
            Error = np.sqrt(Intensity)
        else:
            Error = None
        header['FileName'] = filename
        return {'header': header, 'Intensity': Intensity, 'Error': Error, 'mask': None}


@register_plugin
class SEPlugin_BDF(SASExposurePlugin):
    """SASExposure I/O plugin for Bessy Data Format v1 and v2"""
    _isread = True
    _name = 'BDF'
    _default_read_kwargs = {'bdfext': '.bdf',
                            'bhfext': '.bhf',
                            'read_corrected_if_present': True,
                            'load_mask': True,
                            }
    _filename_regex = re.compile(r'\.(bdf|bhf)$', re.IGNORECASE)

    def read(self, filename, **kwargs):
        self._before_read(kwargs)
        filename = misc.findfileindirs(filename, kwargs['dirs'])
        data = misc.flatten_hierarchical_dict(twodim.readbdf(filename))
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
        header = SASHeader(data, **kwargs)
        if kwargs['load_mask'] and 'CORR.Maskfile' in header:
            path, filename = os.path.split(header['CORR.Maskfile'])
            if path:
                dirs = [path] + kwargs['dirs']
            else:
                dirs = kwargs['dirs']
            mask = SASMask(filename, dirs=dirs)
        else:
            mask = None
        header['FileName'] = filename
        return {'Intensity': Intensity, 'Error': Error, 'header': header, 'mask': mask}


@register_plugin
class SEPlugin_MAR(SASExposurePlugin):
    """SASExposure I/O plugin for MarResearch .image files."""
    _isread = True
    _name = 'MarCCD'
    _default_read_kwargs = {'estimate_errors': True}
    _filename_regex = re.compile(r'\.image$', re.IGNORECASE)

    def read(self, filename, **kwargs):
        self._before_read(kwargs)
        filename = misc.findfileindirs(filename, kwargs['dirs'])
        Intensity, header = twodim.readmar(filename)
        header = SASHeader(header, **kwargs)
        if kwargs['estimate_errors']:
            Error = np.sqrt(Intensity)
        else:
            Error = None
        header['FileName'] = filename
        return {'Intensity': Intensity, 'Error': Error, 'header': header, 'mask': None}


@register_plugin
class SEPlugin_BerSANS(SASExposurePlugin):
    """SASExposure I/O plugin for BerSANS two-dimensional data"""
    _isread = True
    _name = 'BerSANS'
    _filename_regex = re.compile(r'D[^/\\]*\.(\d+)$', re.IGNORECASE)

    def read(self, filename, **kwargs):
        self._before_read(kwargs)
        filename = misc.findfileindirs(filename, dirs=kwargs['dirs'])
        Intensity, Error, hed = twodim.readBerSANSdata(filename)
        header = SASHeader(hed, **kwargs)
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
        header['FileName'] = filename
        return {'Intensity': Intensity, 'Error': Error, 'header': header, 'mask': mask}


@register_plugin
class SEPlugin_HDF5(SASExposurePlugin):
    """SASExposure I/O plugin for HDF5 files."""
    _isread = True
    _iswrite = True
    _default_read_kwargs = {'HDF5_Groupnameformat': 'FSN%d',
                            'HDF5_Intensityname': 'Intensity',
                            'HDF5_Errorname': 'Error',
                            'estimate_error': True,
                            'load_mask': True}
    _name = 'HDF5'
    _filename_regex = re.compile(r'\.(h5|hdf5)$', re.IGNORECASE)

    def check_if_applies(self, filename):
        return (SASExposurePlugin.check_if_applies(self, filename) or
                isinstance(filename, h5py.highlevel.Group))

    def read(self, filename, **kwargs):
        self._before_read(kwargs)
        with _HDF_parse_group(filename) as hdf_group:
            if kwargs['HDF5_Intensityname'] in list(hdf_group.keys()):
                Intensity = hdf_group[kwargs['HDF5_Intensityname']].value
            else:
                # if no intensity matrix is in the group, try if this group contains
                # more sub-groups, which correspond to measurements.
                try:
                    # try to get all possible fsns.
                    fsns = [int(re.match(misc.re_from_Cformatstring_numbers(kwargs['HDF5_Groupnameformat']), k).group(1)) for k in list(
                        hdf_group.keys()) if re.match(misc.re_from_Cformatstring_numbers(kwargs['HDF5_Groupnameformat']), k)]
                    if not fsns:
                        # if we did not find any fsns, we raise an IOError
                        raise RuntimeError
                    # if we did find fsns, read them one-by-one.
                    return self.read_multi(filename, fsns, **kwargs)
                except RuntimeError:
                    raise IOError(
                        'No Intensity dataset in HDF5 group ' + hdf_group.filename + ':' + hdf_group.name)
            if kwargs['HDF5_Errorname'] in list(hdf_group.keys()):
                Error = hdf_group[kwargs['HDF5_Errorname']].value
            elif kwargs['estimate_error']:
                Error = np.sqrt(Intensity)
            header = SASHeader(hdf_group, **kwargs)
            if header['maskid'] is not None and kwargs['load_mask']:
                mask = SASMask(hdf_group.parent, maskid=header['maskid'])
            else:
                mask = None
        return {'Intensity': Intensity, 'Error': Error, 'header': header, 'mask': mask}

    def read_multi(self, filenameformat, fsns, **kwargs):
        self._before_read(kwargs)
        we_have_loaded_a_hdf5_file = False
        try:
            if isinstance(filenameformat, str):
                try:
                    filenameformat % 1
                except TypeError:
                    # filenameformat is not a format string, it is to be treated as a HDF5
                    # file. Open it
                    filenameformat = h5py.highlevel.File(filenameformat)
                    we_have_loaded_a_hdf5_file = True
                    pass
                else:
                    for f in fsns:
                        yield self.read(filenameformat % f, **kwargs)
                    return
            if isinstance(filenameformat, h5py.highlevel.Group):
                for f in fsns:
                    groupname = kwargs['HDF5_Groupnameformat'] % f

                    yield self.read(filenameformat[groupname], **kwargs)
            else:
                raise ValueError(
                    'Invalid argument "filenameformat": ' + str(filenameformat))
        finally:
            if we_have_loaded_a_hdf5_file:
                filenameformat.close()
        return

    def write(self, hdf_or_filename, exposure, **kwargs):
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
        with _HDF_parse_group(hdf_or_filename, mode='a') as hpg:
            groupname = 'FSN%d' % exposure.header['FSN']
            if groupname in list(hpg.keys()):
                del hpg[groupname]
            hpg.create_group(groupname)
            for k in exposure.matrix_names:
                if k is None:
                    continue
                hpg[groupname].create_dataset(
                    k, data=getattr(exposure, k), **kwargs)
            exposure.header.write(hpg[groupname], **kwargs)
            if exposure.mask is not None:
                exposure.mask.write_to_hdf5(hpg)


@register_plugin
class SEPlugin_BareImage(SASExposurePlugin):
    """SASExposure I/O plugin for cbf and tif image files without metadata."""
    _isread = True
    _name = 'bare'
    _default_read_kwargs = {'estimate_errors': True}
    _filename_regex = re.compile('\.(n5|nx5)$', re.IGNORECASE)

    def read(self, filename, **kwargs):
        """Read a "bare" image file
        """
        self._before_read(kwargs)
        filename = misc.findfileindirs(filename, kwargs['dirs'])
        if filename.upper().endswith('.CBF'):
            Intensity, header = twodim.readcbf(filename, load_header=True)
        elif filename.upper().endswith('.TIF') or filename.upper().endswith('.TIFF'):
            Intensity = twodim.readtif(filename)
            header = SASHeader({'__Origin__': 'bare_image'}, **kwargs)
        if kwargs['estimate_errors']:
            Error = np.sqrt(Intensity)
        else:
            Error = None
        header['FileName'] = filename
        return {'Intensity': Intensity, 'Error': Error, 'header': header, 'mask': None}
