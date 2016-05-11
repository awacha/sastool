import datetime
import os
import pickle
from typing import Optional

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

    @property
    def fsn(self) -> int:
        return self._data['exposure']['fsn']

    @property
    def energy(self) -> ErrorValue:
        """X-ray energy"""
        return (ErrorValue(*(scipy.constants.physical_constants['speed of light in vacuum'][0::2])) *
                ErrorValue(*(scipy.constants.physical_constants['Planck constant in eV s'][0::2])) /
                scipy.constants.nano /
                self.wavelength)

    @property
    def wavelength(self) -> ErrorValue:
        """X-ray wavelength"""
        return ErrorValue(self._data["geometry"]['wavelength'], self._data['geometry']['wavelength.err'])

    @property
    def distance(self) -> ErrorValue:
        """Sample-to-detector distance"""
        return ErrorValue(self._data['geometry']['truedistance'],
                          self._data['geometry']['truedistance.err'])

    @property
    def temperature(self) -> Optional[ErrorValue]:
        """Sample temperature"""
        try:
            return self._data['environment']['temperature']
        except KeyError:
            return None

    @property
    def beamcenterx(self) -> ErrorValue:
        """X (column) coordinate of the beam center, pixel units, 0-based."""
        return ErrorValue(self._data['geometry']['beamposy'], 0)

    @property
    def beamcentery(self) -> ErrorValue:
        """Y (row) coordinate of the beam center, pixel units, 0-based."""
        return ErrorValue(self._data['geometry']['beamposx'], 0)

    @property
    def pixelsizex(self) -> ErrorValue:
        """X (column) size of a pixel, in mm units"""
        return ErrorValue(self._data['geometry']['pixelsize'], 0)

    @property
    def pixelsizey(self) -> ErrorValue:
        """Y (row) size of a pixel, in mm units"""
        return ErrorValue(self._data['geometry']['pixelsize'], 0)

    @property
    def exposuretime(self) -> ErrorValue:
        """Exposure time in seconds"""
        return ErrorValue(self._data['exposure']['exptime'], 0)

    @property
    def date(self) -> datetime.datetime:
        """Date of the experiment (start of exposure)"""
        return self._data['exposure']['startdate']

    @property
    def maskname(self) -> Optional[str]:
        """Name of the mask matrix file."""
        mask = self._data['geometry']['mask']
        if os.path.abspath(mask):
            mask = os.path.split(mask)[-1]
        return mask

    @property
    def transmission(self) -> ErrorValue:
        """Sample transmission."""
        return ErrorValue(self._data['sample']['transmission.val'], self._data['sample']['transmission.err'])

    @property
    def vacuum(self) -> ErrorValue:
        """Vacuum pressure around the sample"""
        return ErrorValue(self._data['environment']['vacuum_pressure'], 0)

    @property
    def flux(self) -> ErrorValue:
        """X-ray flux in photons/sec."""
        return ErrorValue(self._data['datareduction']['flux'],
                          self._data['datareduction']['flux.err'])
