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

    @abc.abstractproperty
    def title(self) -> str:
        """Sample name"""

    @abc.abstractproperty
    def fsn(self) -> int:
        """File sequence number """

    @abc.abstractproperty
    def energy(self) -> ErrorValue:
        """X-ray energy"""

    @abc.abstractproperty
    def wavelength(self) -> ErrorValue:
        """X-ray wavelength"""

    @abc.abstractproperty
    def distance(self) -> ErrorValue:
        """Sample-to-detector distance"""

    @abc.abstractproperty
    def temperature(self) -> Optional[ErrorValue]:
        """Sample temperature"""

    @abc.abstractproperty
    def beamcenterx(self) -> ErrorValue:
        """X (column) coordinate of the beam center, pixel units, 0-based."""

    @abc.abstractproperty
    def beamcentery(self) -> ErrorValue:
        """Y (row) coordinate of the beam center, pixel units, 0-based."""

    @abc.abstractproperty
    def pixelsizex(self) -> ErrorValue:
        """X (column) size of a pixel, in mm units"""

    @abc.abstractproperty
    def pixelsizey(self) -> ErrorValue:
        """Y (row) size of a pixel, in mm units"""

    @abc.abstractproperty
    def exposuretime(self) -> ErrorValue:
        """Exposure time in seconds"""

    @abc.abstractproperty
    def date(self) -> datetime.datetime:
        """Date of the experiment (start of exposure)"""

    @abc.abstractproperty
    def maskname(self) -> Optional[str]:
        """Name of the mask matrix file."""

    @abc.abstractclassmethod
    def new_from_file(self, filename):
        """Load a header from a file."""

    @abc.abstractproperty
    def transmission(self) -> ErrorValue:
        """Sample transmission."""

    @abc.abstractproperty
    def vacuum(self) -> ErrorValue:
        """Vacuum pressure around the sample"""

    @abc.abstractproperty
    def flux(self) -> ErrorValue:
        """X-ray flux in photons/sec."""

    @abc.abstractproperty
    def thickness(self) -> ErrorValue:
        """Sample thickness in cm"""
