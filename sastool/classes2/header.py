"""This module defines the header class."""
import abc
import datetime
from typing import Optional, Dict

from ..misc.errorvalue import ErrorValue


class Header(object, metaclass=abc.ABCMeta):
    """A generic header class for SAXS experiments, with the bare minimum attributes to facilitate data processing and
    reduction.

    """

    def __init__(self, headerdict: Optional[Dict] = None):
        if headerdict is None:
            self._data = {}
        else:
            self._data = headerdict

    @property
    @abc.abstractmethod
    def title(self) -> str:
        """Sample name"""

    @property
    @abc.abstractmethod
    def fsn(self) -> int:
        """File sequence number """

    @property
    @abc.abstractmethod
    def energy(self) -> ErrorValue:
        """X-ray energy"""

    @property
    @abc.abstractmethod
    def wavelength(self) -> ErrorValue:
        """X-ray wavelength"""

    @property
    @abc.abstractmethod
    def distance(self) -> ErrorValue:
        """Sample-to-detector distance"""

    @property
    @abc.abstractmethod
    def temperature(self) -> Optional[ErrorValue]:
        """Sample temperature"""

    @property
    @abc.abstractmethod
    def beamcenterx(self) -> ErrorValue:
        """X (column) coordinate of the beam center, pixel units, 0-based."""

    @property
    @abc.abstractmethod
    def beamcentery(self) -> ErrorValue:
        """Y (row) coordinate of the beam center, pixel units, 0-based."""

    @property
    @abc.abstractmethod
    def pixelsizex(self) -> ErrorValue:
        """X (column) size of a pixel, in mm units"""

    @property
    @abc.abstractmethod
    def pixelsizey(self) -> ErrorValue:
        """Y (row) size of a pixel, in mm units"""

    @property
    @abc.abstractmethod
    def exposuretime(self) -> ErrorValue:
        """Exposure time in seconds"""

    @property
    @abc.abstractmethod
    def date(self) -> datetime.datetime:
        """Date of the experiment (start of exposure)"""

    @property
    @abc.abstractmethod
    def maskname(self) -> Optional[str]:
        """Name of the mask matrix file."""

    @abc.abstractclassmethod
    def new_from_file(self, filename):
        """Load a header from a file."""

    @property
    @abc.abstractmethod
    def transmission(self) -> ErrorValue:
        """Sample transmission."""

    @property
    @abc.abstractmethod
    def vacuum(self) -> ErrorValue:
        """Vacuum pressure around the sample"""

    @property
    @abc.abstractmethod
    def flux(self) -> ErrorValue:
        """X-ray flux in photons/sec."""

    @property
    @abc.abstractmethod
    def thickness(self) -> ErrorValue:
        """Sample thickness in cm"""

    @property
    @abc.abstractmethod
    def distancedecrease(self) -> ErrorValue:
        """Distance by which the sample is nearer to the detector than the
        distance calibration sample"""

    @property
    @abc.abstractmethod
    def samplex(self) -> ErrorValue:
        """Horizontal sample position"""

    @property
    @abc.abstractmethod
    def sampley(self) -> ErrorValue:
        """Vertical sample position"""

    @abc.abstractmethod
    def motorposition(self, motorname: str) -> float:
        """Position of the motor `motorname`."""

    @property
    @abc.abstractmethod
    def username(self) -> str:
        """Name of the instrument operator"""

    @property
    @abc.abstractmethod
    def project(self) -> str:
        """Project name"""

    @property
    @abc.abstractmethod
    def fsn_emptybeam(self) -> int:
        """File sequence number of the empty beam measurement"""

    @property
    @abc.abstractmethod
    def fsn_absintref(self) -> int:
        """File sequence number of the absolute intensity reference measurement
        """

    @property
    @abc.abstractmethod
    def absintfactor(self) -> ErrorValue:
        """Absolute intensity calibration factor"""
