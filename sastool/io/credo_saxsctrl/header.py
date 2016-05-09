import datetime
from typing import Optional

import dateutil.parser
import scipy.constants

from ...classes2 import Header
from ...misc.errorvalue import ErrorValue


class SAXSCtrl_Header(Header):
    """Header file written by SAXSCtrl"""

    def __init__(self):
        super().__init__()
        self._data = {}

    @classmethod
    def new_from_file(cls, filename):
        self = cls()
        with open(filename, 'rt', encoding='utf-8') as f:
            for l in f:
                if ':' not in l:
                    continue
                left, right = [x.strip() for x in l.split(':', 1)]
                if left.startswith('Sample name'):
                    self._data['Title'] = right
                elif left.startswith('Sample-to-detector distance'):
                    self._data['Dist'] = float(right)
                elif left.startswith('Sample thickness'):
                    self._data['Thickness'] = float(right)
                elif left.startswith('Sample transmission'):
                    self._data['Transm'] = float(right)
                elif left.startswith('Beam x y for integration'):
                    self._data['BeamPosX'] = float(right.split()[0]) - 1
                    self._data['BeamPosY'] = float(right.split()[1]) - 1
                elif left.startswith('Pixel size of 2D detector'):
                    self._data['PixelSize'] = float(right)
                elif left.startswith('Measurement time'):
                    self._data['ExpTime'] = float(right)
                else:
                    for t in [int, float, dateutil.parser.parse, str]:
                        try:
                            self._data[left] = t(right)
                        except:
                            continue
                    if left not in self._data:
                        raise ValueError("Cannot interpret line: %s" % l)

    @property
    def title(self) -> str:
        return self._data['Title']

    @property
    def fsn(self) -> int:
        return self._data['FSN']

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
        return ErrorValue(self._data["Wavelength"], 0)

    @property
    def distance(self) -> ErrorValue:
        """Sample-to-detector distance"""
        if 'DistCalibrated' in self._data:
            dist = self._data['DistCalibrated']
        else:
            dist = self._data["Dist"]
        if 'DistCalibratedError' in self._data:
            disterr = self._data['DistCalibratedError']
        elif 'DistError' in self._data:
            disterr = self._data['DistError']
        else:
            disterr = 0
        return ErrorValue(dist, disterr)

    @property
    def temperature(self) -> Optional[ErrorValue]:
        """Sample temperature"""
        try:
            return self._data['Temperature']
        except KeyError:
            return None

    @property
    def beamcenterx(self) -> ErrorValue:
        """X (column) coordinate of the beam center, pixel units, 0-based."""
        return ErrorValue(self._data['BeamPosY'], 0)

    @property
    def beamcentery(self) -> ErrorValue:
        """Y (row) coordinate of the beam center, pixel units, 0-based."""
        return ErrorValue(self._data['BeamPosX'], 0)

    @property
    def pixelsizex(self) -> ErrorValue:
        """X (column) size of a pixel, in mm units"""
        return ErrorValue(self._data['PixelSize'], 0)

    @property
    def pixelsizey(self) -> ErrorValue:
        """Y (row) size of a pixel, in mm units"""
        return ErrorValue(self._data['PixelSize'], 0)

    @property
    def exposuretime(self) -> ErrorValue:
        """Exposure time in seconds"""
        return ErrorValue(self._data['ExpTime'], 0)

    @property
    def date(self) -> datetime.datetime:
        """Date of the experiment (start of exposure)"""
        return self._data['Date']

    @property
    def maskname(self) -> Optional[str]:
        """Name of the mask matrix file."""
        try:
            return self._data['maskid'] + '.mat'
        except KeyError:
            return None

    @property
    def transmission(self) -> ErrorValue:
        """Sample transmission."""
        return ErrorValue(self._data['Transm'], self._data['TransmError'])
