import re
import os
import numpy as np
import numbers
import collections
import datetime
import h5py
import pickle as pickle

from .header import SASHeader, SASHistory
from ..io import header
from .. import misc
from .common import _HDF_parse_group

__all__ = []


def register_plugin(pluginclass, idx=None):
    SASHeader.register_IOplugin(pluginclass(), idx)
    return pluginclass


class SASHeaderPlugin(object):
    _isread = False
    _iswrite = False
    _name = '__plugin'
    _default_read_kwargs = {'dirs': '.', 'error_on_not_found': True}
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

    def write(self, filename, hed, **kwargs):
        raise NotImplementedError

    def check_if_applies(self, filename_or_dict):
        if isinstance(filename_or_dict, dict):
            if '__Origin__' in filename_or_dict:
                return filename_or_dict['__Origin__'] == self._name
            else:
                return False
        elif isinstance(filename_or_dict, str):
            if self._filename_regex is not None:
                return (self._filename_regex.search(filename_or_dict) is not None)
            else:
                return False
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
        for k in SASHeaderPlugin._default_read_kwargs:
            if k not in kwargs:
                kwargs[k] = SASHeaderPlugin._default_read_kwargs[k]

    def __repr__(self):
        return "<%s SASHeader I/O Plugin>" % self._name

    @property
    def name(self):
        """The name of this plug-in"""
        return self._name

    def __getitem__(self, key):
        if key in self._default_read_kwargs:
            return self._default_read_kwargs[key]
        elif key in SASHeaderPlugin._default_read_kwargs:
            return SASHeaderPlugin._default_read_kwargs[key]
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        self._default_read_kwargs[key] = value

@register_plugin
class SHPlugin_CREDOpickle(SASHeaderPlugin):
    _isread = True
    _name = 'CREDO pickle'
    _filename_regex = re.compile('(ccd|.edf)$', re.IGNORECASE)

    def read(self, filename_or_dict, **kwargs):
        """Read header data from an CREDO pickle file

        Inputs:
            filename_or_dict: the full name of the file or a dict

        Outputs: the updated header structure. Fields not present in the file
            are kept unchanged.
        """
        self._before_read(kwargs)
        if isinstance(filename_or_dict, str):
            with open(find_in_subfolders(filename_or_dict, kwargs['dirs']), 'rb') as f:
                filename_or_dict = pickle.load(f)

        h = flatten_dict(filename_or_dict)

        ka = {}

        ka['FSN'] = 'fsn'
        ka['BeamPosX'] = 'geometry.beamposx'
        ka['BeamPosY'] = 'geometry.beamposy'
        ka['MeasTime'] = 'devices.pilatus.exptime'
        ka['Monitor'] = 'devices.pilatus.exptime'
        ka['Detector'] = 'devices.pilatus.cameraSN'
        ka['Date'] = 'devices.pilatus.starttime'
        ka['Wavelength'] = 'geometry.wavelength'
        ka['WavelengthError']='geometry.wavelength.err'
        ka['Transm']='sample.transmission.val'
        ka['TransmError']='sample.transmissionl.err'
        ka['Dist']='geometry.dist_sample_det'
        ka['DistError']='geometry.dist_sample_det'
        ka['DistCalibrated']='geometry.truedistance'
        ka['DistCalibratedError']='geometry.truedistance.err'
        ka['XPixel']='geometry.pixelsize'
        ka['YPixel']='geometry.pixelsize'
        ka['Title']='sample.title'
        ka['maskid']='geometry.mask'
        ka['Thickness']='sample.thickness.val'
        ka['ThicknessError']='sample.thickness.err'
        h['__particle__'] = 'photon'
        return (h, ka)


@register_plugin
class SHPlugin_ESRF(SASHeaderPlugin):
    _isread = True
    _name = 'EDF ID02'
    _filename_regex = re.compile('(ccd|.edf)$', re.IGNORECASE)

    def read(self, filename_or_edf, **kwargs):
        """Read header data from an ESRF ID02 EDF file.

        Inputs:
            filename_or_edf: the full name of the file or an edf structure read
                by readehf()

        Outputs: the updated header structure. Fields not present in the file
            are kept unchanged.
        """
        self._before_read(kwargs)
        if isinstance(filename_or_edf, str):
            filename_or_edf = header.readehf(
                misc.findfileindirs(filename_or_edf, kwargs['dirs']))
        h = filename_or_edf
        ka = {}

        ka['FSN'] = 'HMRunNumber'
        ka['BeamPosX'] = 'Center_2'
        ka['BeamPosY'] = 'Center_1'
        ka['MeasTime'] = 'ExposureTime'
        ka['Monitor'] = 'Intensity0'
        ka['Detector'] = 'DetectorInfo'
        ka['Date'] = 'HMStartTime'
        ka['Wavelength'] = 'WaveLength'
        h['Transm'] = h['Intensity1'] / h['Intensity0']
        h['Dist'] = h['SampleDistance'] * 1000
        h['XPixel'] = (h['PSize_1'] * 1000)
        h['YPixel'] = (h['PSize_2'] * 1000)
        h['Title'] = h['TitleBody']
        h['maskid'] = os.path.splitext(h['MaskFileName'])[0]
        his = SASHistory()
        for k in sorted([k for k in h if k.startswith('History')]):
            his.add(h[k], h['HMStartTime'])
        his.add('Loaded EDF header from file ' + filename_or_edf['FileName'])
        h['History'] = his
        h['__particle__'] = 'photon'
        return (h, ka)


@register_plugin
class SHPlugin_B1_org(SASHeaderPlugin):
    _isread = True
    _name = 'B1 original header'
    _filename_regex = re.compile(r'org_?[^/\\]*\.[^/\\]*$', re.IGNORECASE)

    def read(self, filename, **kwargs):
        self._before_read(kwargs)
        if isinstance(filename, str):
            hed = header.readB1header(
                misc.findfileindirs(filename, kwargs['dirs']))
        else:
            hed = filename
        if 'History' in hed:
            hed['History'] = SASHistory(hed['History'])
        else:
            hed['History'] = SASHistory()
        if isinstance(filename, str):
            hed['History'].add('Original header loaded: ' + filename)
        hed['__particle__'] = 'photon'
        return (hed, {})


@register_plugin
class SHPlugin_B1_int2dnorm(SASHeaderPlugin):
    _isread = True
    _name = 'B1 log'
    _iswrite = True
    _filename_regex = re.compile(
        r'(intnorm[^/\\]*\.log$|\.param$)', re.IGNORECASE)

    def read(self, filename, **kwargs):
        self._before_read(kwargs)
        if isinstance(filename, str):
            hed = header.readB1logfile(
                misc.findfileindirs(filename, kwargs['dirs']))
        else:
            hed = filename
        if 'History' in hed:
            hed['History'] = SASHistory(hed['History'])
        else:
            hed['History'] = SASHistory()
        if isinstance(filename, str):
            hed['History'].add('B1 logfile loaded: ' + filename)
        hed['__particle__'] = 'photon'
        return (hed, {})

    def write(self, filename, hed, **kwargs):
        header.writeB1logfile(filename, hed)


@register_plugin
class SHPlugin_CREDO_Reduced(SASHeaderPlugin):
    _isread = True
    _name = 'CREDO Reduced'
    _iswrite = True
    _filename_regex = re.compile(r'(crd[^/\\]*\.log$$)', re.IGNORECASE)

    def read(self, filename, **kwargs):
        self._before_read(kwargs)
        if isinstance(filename, str):
            hed = header.readB1logfile(
                misc.findfileindirs(filename, kwargs['dirs']))
        else:
            hed = filename
        if 'History' in hed:
            hed['History'] = SASHistory(hed['History'])
        else:
            hed['History'] = SASHistory()
        if isinstance(filename, str):
            hed['History'].add('CREDO Reduced logfile loaded: ' + filename)
        hed['__particle__'] = 'photon'
        return (hed, {})

    def write(self, filename, hed, **kwargs):
        header.writeB1logfile(filename, hed)


@register_plugin
class SHPlugin_HDF5(SASHeaderPlugin):
    _isread = True
    _iswrite = True
    _name = 'HDF5'
    _filename_regex = re.compile(r'\.(h5|hdf5)$', re.IGNORECASE)
    _default_read_kwargs = {'HDF5_Groupnameformat': 'FSN%d',
                            'HDF5_Intensityname': 'Intensity',
                            'HDF5_Errorname': 'Error'}
    # information on HDF5 reading: not all Python datatypes have their HDF5
    # equivalents. These help to convert them to/from HDF5.
    # depending on the type: list of (type, converter_function) tuples
    _HDF5_read_postprocess_type = [(np.generic, lambda x:x.tolist()), ]
    # depending on the key name: dictionary of 'key':converter_function pairs
    _HDF5_read_postprocess_name = {
        'FSNs': lambda x: x.tolist(), 'History': SASHistory}

    def check_if_applies(self, filename):
        return (SASHeaderPlugin.check_if_applies(self, filename) or
                isinstance(filename, h5py.highlevel.Group))

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

    def read(self, hdf_entity, **kwargs):
        """Read the parameter structure from the attributes of a HDF entity
        (group or dataset). hdf_entity should be an instance of
        h5py.highlevel.Dataset or h5py.highlevel.Group or h5py.highlevel.File.
        """
        self._before_read(kwargs)
        hed = {}
        with _HDF_parse_group(hdf_entity) as hdf_group:
            if kwargs['HDF5_Intensityname'] in list(hdf_group.keys()):
                for k in list(hdf_entity.attrs.keys()):
                    attr = hdf_entity.attrs[k]
                    if k in self._HDF5_read_postprocess_name:
                        hed[k] = self._HDF5_read_postprocess_name[k](attr)
                    else:
                        typematch = [
                            x for x in self._HDF5_read_postprocess_type if isinstance(attr, x[0])]
                        if typematch:
                            hed[k] = typematch[0][1](attr)
                        else:
                            hed[k] = attr
                if 'History' not in hed:
                    hed['History'] = SASHistory()
                hed['History'].add(
                    'Header read from HDF:' + hdf_entity.file.filename + hdf_entity.name)
            else:
                # if no intensity matrix is in the group, try if this group contains
                # more sub-groups, which correspond to measurements.
                try:
                    # try to get all possible fsns.
                    fsns = [int(re.match(misc.re_from_Cformatstring_numbers(kwargs['HDF5_Groupnameformat']), k).group(1))
                            for k in list(hdf_group.keys()) if re.match(misc.re_from_Cformatstring_numbers(kwargs['HDF5_Groupnameformat']), k)]
                    if not fsns:
                        # if we did not find any fsns, we raise an IOError
                        raise RuntimeError
                    # if we did find fsns, read them one-by-one.
                    return self.read_multi(hdf_entity, fsns, **kwargs)
                except RuntimeError:
                    raise IOError(
                        'No Intensity dataset in HDF5 group ' + hdf_group.filename + ':' + hdf_group.name)
        return (hed, {})

    def write(self, hdf_entity, hed, **kwargs):
        """Write the parameter structure to a HDF entity (group or dataset) as
        attributes. hdf_entity should be an instance of h5py.highlevel.Dataset
        or h5py.highlevel.Group or h5py.highlevel.File."""
        with _HDF_parse_group(hdf_entity, mode='a') as hdf_entity:
            for k in hed:
                if k == 'History':
                    hdf_entity.attrs[k] = str(hed[k])
                elif isinstance(hed[k], bool):
                    hdf_entity.attrs[k] = int(hed[k])
                elif isinstance(hed[k], numbers.Number):
                    hdf_entity.attrs[k] = hed[k]
                elif isinstance(hed[k], str):
                    hdf_entity.attrs[k] = hed[k].encode('utf-8')
                elif isinstance(hed[k], collections.Sequence):
                    hdf_entity.attrs[k] = hed[k]
                elif isinstance(hed[k], datetime.datetime):
                    hdf_entity.attrs[k] = str(hed[k])
                else:
                    raise ValueError(
                        'Invalid field type: ' + str(k) + ', ', repr(type(hed[k])))


@register_plugin
class SHPlugin_BDF(SASHeaderPlugin):
    _isread = True
    _name = 'BDF'
    _filename_regex = re.compile(r'\.(bdf|bhf)$', re.IGNORECASE)

    def check_if_applies(self, filename):
        if isinstance(filename, dict):
            if '__Origin__' in filename:
                return filename['__Origin__'].startswith('BDF')
            else:
                return False
        else:
            return SASHeaderPlugin.check_if_applies(self, filename)

    def read(self, filename, **kwargs):
        self._before_read(kwargs)
        if isinstance(filename, str):
            hed = misc.flatten_hierarchical_dict(
                header.readbhf(misc.findfileindirs(filename, kwargs['dirs'])))
        else:
            hed = misc.flatten_hierarchical_dict(filename)
        ka = {}
        if 'History' in hed:
            hed['History'] = SASHistory(hed['History'])
        else:
            hed['History'] = SASHistory()
        if hed['C.bdfVersion'] < 2:
            hed['BeamPosX'] = hed['C.xcen'] - 1
            hed['BeamPosY'] = hed['C.ycen'] - 1
            for h in hed['his']:
                hed['History'].add('BDF: ' + h)
        elif hed['C.bdfVersion'] >= 2:
            if 'CORR.CenterX' in hed:
                hed['BeamPosX'] = hed['CORR.CenterX'] - 1
            if 'CORR.CenterY' in hed:
                hed['BeamPosY'] = hed['CORR.CenterY'] - 1
            if 'CORR.EnergyReal' in hed:
                ka['EnergyCalibrated'] = 'CORR.EnergyReal'
            if 'CORR.PixelSizeX' in hed:
                hed['XPixel'] = hed['CORR.PixelSizeX'] * 10
            if 'CORR.PixelSizeY' in hed:
                hed['YPixel'] = hed['CORR.PixelSizeY'] * 10
            for h in hed['HIS']:
                hed['History'].add('BDF: ' + h)
            if 'CORR.SampleThickness' in hed:
                ka['Thickness'] = 'CORR.SampleThickness'
            if 'CORR.SampleThicknessError' in hed:
                ka['ThicknessError'] = 'CORR.SampleThicknessError'
        # common to BDFv1 and v2
        hed['History'].add('History imported from BDF file')
        ka['Energy'] = 'M.Energy'
        ka['Dist'] = 'M.SD'
        ka['Title'] = 'C.Sample'
        ka['Temperature'] = 'C.isTemp'
        ka['MeasTime'] = 'CS.Seconds'
        ka['Monitor'] = 'CS.Monitor'
        ka['Anode'] = 'CS.Anode'
        ka['PosSample'] = 'M.VacSampleX'
        ka['PosRef'] = 'M.RefSampleX'
        ka['Transm'] = 'CT.trans'
        ka['TransmError'] = 'CT.transerr'
        try:
            hed['FSN'] = int(re.findall('\d+', hed['C.Frame'])[-1])
        except IndexError:
            hed['FSN'] = hed['C.Frame']
        hed['__particle__'] = 'photon'
        return (hed, ka)


@register_plugin
class SHPlugin_PAXE(SASHeaderPlugin):
    _isread = True
    _name = 'PAXE'
    _filename_regex = re.compile(r'xe[^/\\]*\.(dat|32)$', re.IGNORECASE)

    def read(self, filename_or_paxe, **kwargs):
        """Read header data from a PAXE (Saclay, France or Budapest, Hungary)
        measurement file.

        Inputs:
            filename_or_paxe: the file name (usually XE????.DAT) or a dict
                loaded by readPAXE().

        Outputs: the updated header structure. Fields not present in the file
            are kept unchanged.
        """
        self._before_read(kwargs)
        if isinstance(filename_or_paxe, str):
            paxe = header.readPAXE(
                misc.findfileindirs(filename_or_paxe, kwargs['dirs']))
        else:
            paxe = filename_or_paxe
        if isinstance(filename_or_paxe, str):
            if 'History' in paxe:
                paxe['History'] = SASHistory(paxe['History'])
            else:
                paxe['History'] = SASHistory()
            paxe['History'].add('Loaded from PAXE file ' + filename_or_paxe)
        paxe['__particle__'] = 'neutron'
        return (paxe, {})


@register_plugin
class SHPlugin_MAR(SASHeaderPlugin):
    _isread = True
    _name = 'MarResearch .image'
    _filename_regex = re.compile(r'\.image$', re.IGNORECASE)

    def read(self, filename_or_mar, **kwargs):
        """Read header data from a MarResearch .image file.

        Inputs:
            filename_or_mar: the full filename or a dict loaded by
                readmarheader()

        Outputs: the updated header structure. Fields not present in the file
            are kept unchanged.
        """
        self._before_read(kwargs)
        if not isinstance(filename_or_mar, dict):
            mar = header.readmarheader(
                misc.findfileindirs(filename_or_mar, kwargs['dirs']))
        else:
            mar = filename_or_mar
        if not isinstance(filename_or_mar, dict):
            if 'History' in mar:
                mar['History'] = SASHistory(mar['History'])
            else:
                mar['History'] = SASHistory()
            mar['History'].add(
                'Loaded from MAR .image file ' + filename_or_mar)
            mar['FileName'] = filename_or_mar
        mar['__particle__'] = 'photon'
        return (mar, {})


@register_plugin
class SHPlugin_BerSANS(SASHeaderPlugin):
    _isread = True
    _name = 'BerSANS'
    _filename_regex = re.compile(r'D[^/\\]*\.(\d+)$', re.IGNORECASE)

    def read(self, filename, **kwargs):
        self._before_read(kwargs)
        if isinstance(filename, str):
            hed = header.readBerSANS(
                misc.findfileindirs(filename, kwargs['dirs']))
        else:
            hed = filename
        if 'History' in hed:
            hed['History'] = SASHistory(hed['History'])
        else:
            hed['History'] = SASHistory()
        hed['History'].add('Imported from a BerSANS file')
        ka = {'maskid': 'MaskFile'}
        hed['__particle__'] = 'neutron'
        return (hed, ka)


@register_plugin
class SHPlugin_ASA_SAS(SASHeaderPlugin):
    _isread = True
    _name = 'SAXSEVAL SAS file'
    _filename_regex = re.compile(r'\.sas$', re.IGNORECASE)

    def read(self, filename, **kwargs):
        self._before_read(kwargs)
        if isinstance(filename, str):
            with open(misc.findfileindirs(filename, kwargs['dirs']), 'r') as f:
                hed1 = pickle.load(f)
        else:
            hed1 = filename
        hed = hed1['params']
        if 'datasettype' in hed1:
            hed['datasettype'] = hed1['datasettype']
        hed['__Origin__'] = 'SAXSEVAL SAS file'
        hed['__particle__'] = 'photon'
        ka = {'Date': 'params.Datetime', 'MeasTime':
              'params.LiveTime', 'FSN': 'basename'}
        return (hed, ka)


@register_plugin
class SHPlugin_bare_image(SASHeaderPlugin):
    _isread = True
    _name = 'bare_image'
    _filename_regex = re.compile('(.tif|.cbf|.jpg|.jpeg)$', re.IGNORECASE)

    def read(self, filename_or_dict, **kwargs):
        """Reader for an empty header.
        """
        self._before_read(kwargs)
        if isinstance(filename_or_dict, str):
            filename_or_dict = {'__Origin__': 'bare_image'}
        return (filename_or_dict, {})


# DO NOT PUT NEW PLUGINS BELOW THIS LINE! The copydict plugin should be
# the last.

@register_plugin
class SHPlugin_copydict(SASHeaderPlugin):
    _isread = True
    _name = 'dummy'
    _filename_regex = None

    def check_if_applies(self, arg):
        return True

    def read(self, dict_to_copy, **kwargs):
        self._before_read(kwargs)
        return (dict_to_copy, {})
