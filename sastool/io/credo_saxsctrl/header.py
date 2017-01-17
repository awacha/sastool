import datetime
import gzip
from typing import Optional, Union

import dateutil.parser
import scipy.constants

from ... import classes2
from ...misc.errorvalue import ErrorValue


class Header(classes2.Header):
    """Header file written by SAXSCtrl"""

    @classmethod
    def new_from_file(cls, filename):
        self = cls()
        try:
            if filename.endswith('.gz'):
                f = gzip.open(filename, 'rt', encoding='utf-8')
            else:
                f = open(filename, 'rt', encoding='utf-8')
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
                    self._data['BeamPosX'] = float(right.split()[1]) - 1
                    self._data['BeamPosY'] = float(right.split()[0]) - 1
                elif left.startswith('Pixel size of 2D detector'):
                    self._data['PixelSize'] = float(right) * 1000  # there is a bug in the header files.
                elif left.startswith('Measurement time'):
                    self._data['ExpTime'] = float(right)
                elif left.startswith('Normalisation factor'):
                    self._data['NormFactor'] = float(right)
                else:
                    for t in [int, float, dateutil.parser.parse, str]:
                        try:
                            self._data[left] = t(right)
                            break
                        except ValueError:
                            continue
                    if left not in self._data:
                        raise ValueError("Cannot interpret line: %s" % l)
        finally:
            try:
                f.close()
            except UnboundLocalError:
                pass
        return self

    @property
    def title(self) -> str:
        return self._data['Title']

    @title.setter
    def title(self, value: str):
        self._data['Title'] = value

    @property
    def fsn(self) -> int:
        return self._data['FSN']

    @fsn.setter
    def fsn(self, value: int):
        self._data['FSN'] = value

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
        return ErrorValue(self._data["Wavelength"], 0)

    @wavelength.setter
    def wavelength(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['Wavelength'] = value.val
        self._data['WavelengthError'] = value.err

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

    @distance.setter
    def distance(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['DistCalibrated'] = value.val
        self._data['DistCalibratedError'] = value.err

    @property
    def temperature(self) -> Optional[ErrorValue]:
        """Sample temperature"""
        try:
            return self._data['Temperature']
        except KeyError:
            return None

    @temperature.setter
    def temperature(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['Temperature'] = value.val
        self._data['TemperatureError'] = value.err

    @property
    def beamcenterx(self) -> ErrorValue:
        """X (column) coordinate of the beam center, pixel units, 0-based."""
        return ErrorValue(self._data['BeamPosX'], 0)

    @beamcenterx.setter
    def beamcenterx(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['BeamPosX'] = value.val
        self._data['BeamPosXError'] = value.err

    @property
    def beamcentery(self) -> ErrorValue:
        """Y (row) coordinate of the beam center, pixel units, 0-based."""
        return ErrorValue(self._data['BeamPosY'], 0)

    @beamcentery.setter
    def beamcentery(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['BeamPosY'] = value.val
        self._data['BeamPosYError'] = value.err

    @property
    def pixelsizex(self) -> ErrorValue:
        """X (column) size of a pixel, in mm units"""
        return ErrorValue(self._data['XPixel'], 0)

    @pixelsizex.setter
    def pixelsizex(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['XPixel'] = value.val
        self._data['XPixelError'] = value.err

    @property
    def pixelsizey(self) -> ErrorValue:
        """Y (row) size of a pixel, in mm units"""
        return ErrorValue(self._data['YPixel'], 0)

    @pixelsizey.setter
    def pixelsizey(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['YPixel'] = value.val
        self._data['YPixelError'] = value.err

    @property
    def exposuretime(self) -> ErrorValue:
        """Exposure time in seconds"""
        return ErrorValue(self._data['ExpTime'], 0)

    @exposuretime.setter
    def exposuretime(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['ExpTime'] = value.val
        self._data['ExpTimeError'] = value.val

    @property
    def date(self) -> datetime.datetime:
        """Date of the experiment (start of exposure)"""
        return self._data['Date']

    @date.setter
    def date(self, value: datetime.datetime):
        self._data['Date'] = value

    @property
    def maskname(self) -> Optional[str]:
        """Name of the mask matrix file."""
        try:
            maskid = self._data['maskid']
            if not maskid.endswith('.mat'):
                maskid = maskid + '.mat'
            return maskid
        except KeyError:
            return None

    @maskname.setter
    def maskname(self, value: str):
        self._data['maskid'] = value

    @property
    def transmission(self) -> ErrorValue:
        """Sample transmission."""
        return ErrorValue(self._data['Transm'], self._data['TransmError'])

    @transmission.setter
    def transmission(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['Transm'] = value.val
        self._data['TransmError'] = value.err

    @property
    def vacuum(self) -> ErrorValue:
        """Vacuum pressure around the sample"""
        return ErrorValue(self._data['Vacuum'], 0)

    @vacuum.setter
    def vacuum(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['Vacuum'] = value.val
        self._data['VacuumError'] = value.err

    @property
    def flux(self) -> ErrorValue:
        """X-ray flux in photons/sec."""
        try:
            return ErrorValue(self._data['Flux'], self._data['FluxError'])
        except KeyError:
            return 1 / self.pixelsizex / self.pixelsizey / ErrorValue(self._data['NormFactor'],
                                                                      self._data['NormFactorError'])

    @flux.setter
    def flux(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['Flux'] = value.val
        self._data['FluxError'] = value.err

    @property
    def thickness(self) -> ErrorValue:
        """Sample thickness in cm"""
        return ErrorValue(self._data['Thickness'], self._data['ThicknessError'])

    @thickness.setter
    def thickness(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['Thickness'] = value.val
        self._data['ThicknessError'] = value.err

    @property
    def distancedecrease(self) -> ErrorValue:
        """Distance by which the sample is nearer to the detector than the
        distance calibration sample"""
        return ErrorValue(self._data['DistMinus'], 0.0)

    @distancedecrease.setter
    def distancedecrease(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['DistMinus'] = value.val
        self._data['DistMinusError'] = value.err

    @property
    def samplex(self) -> ErrorValue:
        """Horizontal sample position"""
        return ErrorValue(self._data['PosSampleX'], self._data['PosSampleXError'])

    @samplex.setter
    def samplex(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['PosSampleX'] = value.val
        self._data['PosSampleXError'] = value.err

    @property
    def sampley(self) -> ErrorValue:
        """Vertical sample position"""
        return ErrorValue(self._data['PosSample'], self._data['PosSampleError'])

    @sampley.setter
    def sampley(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['PosSample'] = value.val
        self._data['PosSampleError'] = value.err

    def motorposition(self, motorname: str) -> float:
        """Position of the motor `motorname`."""
        return self._data[motorname]

    @property
    def username(self) -> str:
        """Name of the instrument operator"""
        return self._data['Owner']

    @username.setter
    def username(self, value: str):
        self._data['Owner'] = value

    @property
    def project(self) -> str:
        """Project name"""
        return self._data['Project']

    @project.setter
    def project(self, value: str):
        self._data['Project'] = value

    @property
    def fsn_emptybeam(self) -> int:
        """File sequence number of the empty beam measurement"""
        return self._data['Empty beam FSN']

    @fsn_emptybeam.setter
    def fsn_emptybeam(self, value: int):
        self._data['Empty beam FSN'] = value

    @property
    def fsn_absintref(self) -> int:
        """File sequence number of the absolute intensity reference measurement
        """
        return self._data['Glassy carbon FSN']

    @fsn_absintref.setter
    def fsn_absintref(self, value: int):
        self._data['Glassy carbon FSN'] = value

    @property
    def absintfactor(self) -> ErrorValue:
        """Absolute intensity calibration factor"""
        return ErrorValue(self._data['NormFactor'], self._data['NormFactorError'])

    @absintfactor.setter
    def absintfactor(self, value: Union[ErrorValue, float]):
        if not isinstance(value, ErrorValue):
            value = ErrorValue(value, 0)
        self._data['NormFactor'] = value.val
        self._data['NormFactorError'] = value.err
