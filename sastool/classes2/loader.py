import abc
import os
from typing import Union, List

from .exposure import Exposure
from .header import Header


class Loader(object, metaclass=abc.ABCMeta):
    """A loader class for SAS experiment sessions"""

    def __init__(self, basedir: Union[str, List[str]], recursive: bool = True):
        """Initialize the loader"""
        if isinstance(basedir, str):
            basedir = os.path.expanduser(basedir)
            self._path = []
            if recursive:
                for d, sds, fs in os.walk(basedir, followlinks=True):
                    self._path.append(d)
            else:
                self._path = [basedir]

    @abc.abstractmethod
    def loadheader(self, fsn: int) -> Header:
        """Load the header for the given file sequence number."""

    @abc.abstractmethod
    def loadexposure(self, fsn: int) -> Exposure:
        """Load the exposure for the given file sequence number."""

    def find_file(self, filename: str) -> str:
        """Find file in the path"""
        for d in self._path:
            if os.path.exists(os.path.join(d, filename)):
                return os.path.join(d, filename)
        raise FileNotFoundError(filename)
