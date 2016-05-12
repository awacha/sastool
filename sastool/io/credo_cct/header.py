import datetime
import os
import pickle
from typing import Optional, Union

import dateutil
import scipy.constants

from ... import classes2
from ...misc.errorvalue import ErrorValue


class Header(classes2.Header):
    """Header file written by SAXSCtrl"""

    def __init__(self):
        super().__init__()
        self._data = {}

    @classmethod
    def new_from_file(cls, filename):
        self = cls()
        with open(filename, 'rb') as f:
            self._data = pickle.load(f)
        return self

    @property
    def title(self) -> str:
        return self._data['sample']['title']

    @title.setter
    def title(self, value: str):
        self._data['sample']['title'] = value

    @property
    def fsn(self) -> int:
        return self._data['exposure']['fsn']

    @fsn.setter
    def fsn(self, value: int):
        self._data['exposure']['fsn'] = value
        self._data['fsn'] = value

    @property
    def energy(self) -> ErrorValue:
        """X-ray energy"""
        return (ErrorValue(*(scipy.constants.physical_constants['speed of light in vacuum'][0::2])) *
                ErrorValue(*(scipy.constants.physical_constants['Planck constant in eV s'][0::2])) /
                scipy.constants.nano /
                self.wavelength)

    @energy.setter
    def energy(self, value: Union[float, ErrorValue]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self.wavelength = (ErrorValue(*(scipy.constants.physical_constants['speed of light in vacuum'][0::2])) *
                           ErrorValue(*(scipy.constants.physical_constants['Planck constant in eV s'][0::2])) /
                           scipy.constants.nano /
                           value)

    @property
    def wavelength(self) -> ErrorValue:
        """X-ray wavelength"""
        return ErrorValue(self._data["geometry"]['wavelength'], self._data['geometry']['wavelength.err'])

    @wavelength.setter
    def wavelength(self, value: Union[float, ErrorValue]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['geometry']['wavelength'] = value.val
        self._data['geometry']['wavelength.err'] = value.err

    @property
    def distance(self) -> ErrorValue:
        """Sample-to-detector distance"""
        return ErrorValue(self._data['geometry']['truedistance'],
                          self._data['geometry']['truedistance.err'])

    @distance.setter
    def distance(self, value: Union[float, ErrorValue]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['geometry']['truedistance'] = value.val
        self._data['geometry']['truedistance.err'] = value.err

    @property
    def temperature(self) -> Optional[ErrorValue]:
        """Sample temperature"""
        try:
            return self._data['environment']['temperature']
        except KeyError:
            return None

    @temperature.setter
    def temperature(self, value: float):
        self._data['environment']['temperature'] = value

    @property
    def beamcenterx(self) -> ErrorValue:
        """X (column) coordinate of the beam center, pixel units, 0-based."""
        return ErrorValue(self._data['geometry']['beamposy'], 0)

    @beamcenterx.setter
    def beamcenterx(self, value: Union[float, ErrorValue]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['geometry']['beamposy'] = value.val
        self._data['geometry']['beamposy.err'] = value.err

    @property
    def beamcentery(self) -> ErrorValue:
        """Y (row) coordinate of the beam center, pixel units, 0-based."""
        return ErrorValue(self._data['geometry']['beamposx'], 0)

    @beamcentery.setter
    def beamcentery(self, value: Union[float, ErrorValue]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['geometry']['beamposx'] = value.val
        self._data['geometry']['beamposx.err'] = value.err

    @property
    def pixelsizex(self) -> ErrorValue:
        """X (column) size of a pixel, in mm units"""
        return ErrorValue(self._data['geometry']['pixelsize'], 0)

    @pixelsizex.setter
    def pixelsizex(self, value: Union[float, ErrorValue]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['geometry']['pixelsize'] = value.val

    @property
    def pixelsizey(self) -> ErrorValue:
        """Y (row) size of a pixel, in mm units"""
        return ErrorValue(self._data['geometry']['pixelsize'], 0)

    @pixelsizey.setter
    def pixelsizey(self, value: Union[float, ErrorValue]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['geometry']['pixelsize'] = value.val

    @property
    def exposuretime(self) -> ErrorValue:
        """Exposure time in seconds"""
        return ErrorValue(self._data['exposure']['exptime'], 0)

    @exposuretime.setter
    def exposuretime(self, value: float):
        self._data['exposure']['exptime'] = value

    @property
    def date(self) -> datetime.datetime:
        """Date of the experiment (start of exposure)"""
        return dateutil.parser.parse(self._data['exposure']['startdate'])

    @date.setter
    def date(self, value: datetime.datetime):
        self._data['exposure']['startdate'] = str(value)

    @property
    def maskname(self) -> Optional[str]:
        """Name of the mask matrix file."""
        mask = self._data['geometry']['mask']
        if os.path.abspath(mask):
            mask = os.path.split(mask)[-1]
        return mask

    @maskname.setter
    def maskname(self, value: str):
        self._data['geometry']['mask'] = value

    @property
    def transmission(self) -> ErrorValue:
        """Sample transmission."""
        return ErrorValue(self._data['sample']['transmission.val'], self._data['sample']['transmission.err'])

    @transmission.setter
    def transmission(self, value: Union[float, ErrorValue]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['sample']['transmission.val'] = value.val
        self._data['sample']['transmission.err'] = value.err

    @property
    def vacuum(self) -> ErrorValue:
        """Vacuum pressure around the sample"""
        return ErrorValue(self._data['environment']['vacuum_pressure'], 0)

    @vacuum.setter
    def vacuum(self, value: float):
        self._data['environment']['vacuum_pressure'] = value

    @property
    def flux(self) -> ErrorValue:
        """X-ray flux in photons/sec."""
        return ErrorValue(self._data['datareduction']['flux'],
                          self._data['datareduction']['flux.err'])

    @flux.setter
    def flux(self, value: Union[float, ErrorValue]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['datareduction']['flux'] = value.val
        self._data['datareduction']['flux.err'] = value.err

    @property
    def thickness(self) -> ErrorValue:
        """Sample thickness in cm"""
        return ErrorValue(self._data['sample']['thickness.val'],
                          self._data['sample']['thickness.err'])

    @thickness.setter
    def thickness(self, value: Union[float, ErrorValue]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['sample']['thickness.val'] = value.val
        self._data['sample']['thickness.err'] = value.err
