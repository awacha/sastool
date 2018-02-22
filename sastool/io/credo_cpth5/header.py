import datetime
import gzip
from typing import Optional, Union, Dict
import h5py

import dateutil.parser
import scipy.constants

from ... import classes2
from ...misc.errorvalue import ErrorValue

class Header(classes2.Header):
    _data:Dict=None
    @classmethod
    def new_from_file(cls, filename:str, samplename:str, dist:float):
        with h5py.File(filename) as f:
            dist = sorted([d for d in f['Samples'][samplename].keys()], key=lambda d:abs(float(d)-dist))[0]
            return cls.new_from_group(f['Samples'][samplename][dist])

    @classmethod
    def new_from_group(cls, grp:h5py.Group):
        self = cls()
        self._data = {}
        for a in grp.attrs:
            self._data[a]=grp.attrs[a]
        return self

    @property
    def title(self) -> str:
        return self._data['title']

    @title.setter
    def title(self, value: str):
        self._data['title'] = value

    @property
    def fsn(self) -> int:
        return self._data['fsn']

    @fsn.setter
    def fsn(self, value: int):
        self._data['fsn'] = value

    @property
    def energy(self) -> ErrorValue:
        """X-ray energy"""
        return (ErrorValue(*(scipy.constants.physical_constants['speed of light in vacuum'][0::2])) *
                ErrorValue(*(scipy.constants.physical_constants['Planck constant in eV s'][0::2])) /
                scipy.constants.nano /
                self.wavelength)

    @energy.setter
    def energy(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self.wavelength = (ErrorValue(*(scipy.constants.physical_constants['speed of light in vacuum'][0::2])) *
                           ErrorValue(*(scipy.constants.physical_constants['Planck constant in eV s'][0::2])) /
                           scipy.constants.nano /
                           value)

    @property
    def wavelength(self) -> ErrorValue:
        """X-ray wavelength"""
        return ErrorValue(self._data["wavelength"], self._data["wavelength.err"])

    @wavelength.setter
    def wavelength(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['wavelength'] = value.val
        self._data['wavelength.err'] = value.err

    @property
    def distance(self) -> ErrorValue:
        """Sample-to-detector distance"""
        return ErrorValue(self._data['distance'], self._data['distance.err'])

    @distance.setter
    def distance(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['distance'] = value.val
        self._data['distance.err'] = value.err

    @property
    def temperature(self) -> Optional[ErrorValue]:
        """Sample temperature"""
        try:
            return ErrorValue(self._data['temperature'], self._data['temperature.err'])
        except KeyError:
            return None

    @temperature.setter
    def temperature(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['temperature'] = value.val
        self._data['temperature.err'] = value.err

    @property
    def beamcenterx(self) -> ErrorValue:
        """X (column) coordinate of the beam center, pixel units, 0-based."""
        return ErrorValue(self._data['beamcenterx'], self._data['beamcenterx.err'])

    @beamcenterx.setter
    def beamcenterx(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['beamcenterx'] = value.val
        self._data['beamcenterx.err'] = value.err

    @property
    def beamcentery(self) -> ErrorValue:
        """Y (row) coordinate of the beam center, pixel units, 0-based."""
        return ErrorValue(self._data['beamcentery'], self._data['beamcentery.err'])

    @beamcentery.setter
    def beamcentery(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['beamcentery'] = value.val
        self._data['beamcentery.err'] = value.err

    @property
    def pixelsizex(self) -> ErrorValue:
        """X (column) size of a pixel, in mm units"""
        return ErrorValue(self._data['pixelsizex'], self._data['pixelsizex.err'])

    @pixelsizex.setter
    def pixelsizex(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['pixelsizex'] = value.val
        self._data['pixelsizex.err'] = value.err

    @property
    def pixelsizey(self) -> ErrorValue:
        """Y (row) size of a pixel, in mm units"""
        return ErrorValue(self._data['pixelsizey'], self._data['pixelsizey.err'])

    @pixelsizey.setter
    def pixelsizey(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['pixelsizey'] = value.val
        self._data['pixelsizey.err'] = value.err

    @property
    def exposuretime(self) -> ErrorValue:
        """Exposure time in seconds"""
        return ErrorValue(self._data['exposuretime'], self._data['exposuretime.err'])

    @exposuretime.setter
    def exposuretime(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['exposuretime'] = value.val
        self._data['exposuretime.err'] = value.val

    @property
    def date(self) -> datetime.datetime:
        """Date of the experiment (start of exposure)"""
        return dateutil.parser.parse(self._data['date'])

    @date.setter
    def date(self, value: datetime.datetime):
        self._data = str(value)

    @property
    def maskname(self) -> Optional[str]:
        """Name of the mask matrix file."""
        try:
            maskid = self._data['maskname']
            if not maskid.endswith('.mat'):
                maskid = maskid + '.mat'
            return maskid
        except KeyError:
            return None

    @maskname.setter
    def maskname(self, value: str):
        self._data['maskname'] = value

    @property
    def transmission(self) -> ErrorValue:
        """Sample transmission."""
        return ErrorValue(self._data['transmission'], self._data['transmission.err'])

    @transmission.setter
    def transmission(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['transmission'] = value.val
        self._data['transmission.err'] = value.err

    @property
    def vacuum(self) -> ErrorValue:
        """Vacuum pressure around the sample"""
        return ErrorValue(self._data['vacuum'], self._data['vacuum.err'])

    @vacuum.setter
    def vacuum(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['vacuum'] = value.val
        self._data['vacuum.err'] = value.err

    @property
    def flux(self) -> ErrorValue:
        """X-ray flux in photons/sec."""
        try:
            return ErrorValue(self._data['flux'], self._data['flux.err'])
        except KeyError:
            return 1 / self.pixelsizex / self.pixelsizey / ErrorValue(self._data['absintfactor'],
                                                                      self._data['absintfactor.err'])

    @flux.setter
    def flux(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['flux'] = value.val
        self._data['flux.err'] = value.err

    @property
    def thickness(self) -> ErrorValue:
        """Sample thickness in cm"""
        return ErrorValue(self._data['thickness'], self._data['thickness.err'])

    @thickness.setter
    def thickness(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['thickness'] = value.val
        self._data['thickness.err'] = value.err

    @property
    def distancedecrease(self) -> ErrorValue:
        """Distance by which the sample is nearer to the detector than the
        distance calibration sample"""
        return ErrorValue(self._data['distancedecrease'], self._data['distancedecrease.err'])

    @distancedecrease.setter
    def distancedecrease(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['distancedecrease'] = value.val
        self._data['distancedecrease.err'] = value.err

    @property
    def samplex(self) -> ErrorValue:
        """Horizontal sample position"""
        return ErrorValue(self._data['samplex'], self._data['samplex.err'])

    @samplex.setter
    def samplex(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['samplex'] = value.val
        self._data['samplex.err'] = value.err

    @property
    def sampley(self) -> ErrorValue:
        """Vertical sample position"""
        return ErrorValue(self._data['sampley'], self._data['sampley.err'])

    @sampley.setter
    def sampley(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['sampley'] = value.val
        self._data['sampley.err'] = value.err

    def motorposition(self, motorname: str) -> float:
        """Position of the motor `motorname`."""
        return self._data[motorname]

    @property
    def username(self) -> str:
        """Name of the instrument operator"""
        return self._data['username']

    @username.setter
    def username(self, value: str):
        self._data['username'] = value

    @property
    def project(self) -> str:
        """Project name"""
        return self._data['project']

    @project.setter
    def project(self, value: str):
        self._data['project'] = value

    @property
    def fsn_emptybeam(self) -> int:
        """File sequence number of the empty beam measurement"""
        return self._data['fsn_emptybeam']

    @fsn_emptybeam.setter
    def fsn_emptybeam(self, value: int):
        self._data['fsn_emptybeam'] = value

    @property
    def fsn_absintref(self) -> int:
        """File sequence number of the absolute intensity reference measurement
        """
        return self._data['fsn_absintref']

    @fsn_absintref.setter
    def fsn_absintref(self, value: int):
        self._data['fsn_absintref'] = value

    @property
    def absintfactor(self) -> ErrorValue:
        """Absolute intensity calibration factor"""
        return ErrorValue(self._data['absintfactor'], self._data['absintfactor.err'])

    @absintfactor.setter
    def absintfactor(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['absintfactor'] = value.val
        self._data['absintfactor.err'] = value.err
