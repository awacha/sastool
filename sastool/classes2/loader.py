import abc
import os
from typing import Union, List

import numpy as np

from .curve import Curve
from .exposure import Exposure
from .header import Header


class Loader(object, metaclass=abc.ABCMeta):
    """A loader class for SAS experiment sessions. This is an abstract base class, you may need to
    subclass it for serious work."""

    def __init__(self, basedir: Union[str, List[str]], recursive: bool = True, processed: bool = True, maskpath=None,
                 headerpath=None):
        """Initialize the loader

        Inputs:
            basedir:
                str: the root directory of the measurement data.
                list of str-s: several directories where the data files are searched for.
            recursive: if True (default), all subdirectories of basedir (or the subdirectories
                of the elements of basedir if it was a list) are enumerated and added to the
                data loading path upon calling the constructor.
            processed: if processed data are to be loaded (default). If this is False, raw, i.e.
                unprocessed data is expected.
            maskpath: a list of directories to search masks for. If None (default), use the base
                path for looking for mask matrices.
            headerpath: a list of directories to search header files for. If None (default), use
                the base path for looking for header files.

        Remarks:
            Constructs with the "~" character are resolved using os.path.expanduser()
        """
        if isinstance(basedir, str):
            basedir = [basedir]
        elif isinstance(basedir, list):
            if not all([isinstance(d, str) for d in basedir]):
                raise TypeError('All elements of the basedir list must be strings.')
        else:
            raise TypeError('Unknown type %s for basedir argument.' % type(basedir))
        basedir = [os.path.expanduser(b) for b in basedir]
        self._path = []
        for bd in basedir:
            if recursive:
                for d, sds, fs in os.walk(bd, followlinks=True):
                    self._path.append(d)
            else:
                self._path.append(bd)
        self.processed = processed
        self.basedir = basedir
        if headerpath is None:
            headerpath = self._path
        self._headerpath = headerpath
        if maskpath is None:
            maskpath = self._path
        self._maskpath = maskpath

    @abc.abstractmethod
    def loadheader(self, fsn: int) -> Header:
        """Load the header for the given file sequence number."""

    @abc.abstractmethod
    def loadexposure(self, fsn: int) -> Exposure:
        """Load the exposure for the given file sequence number."""

    def find_file(self, filename: str, strip_path: bool = True, what='exposure') -> str:
        """Find file in the path"""
        if what == 'exposure':
            path = self._path
        elif what == 'header':
            path = self._headerpath
        elif what == 'mask':
            path = self._maskpath
        else:
            path = self._path
        tried = []
        if strip_path:
            filename = os.path.split(filename)[-1]
        for d in path:
            if os.path.exists(os.path.join(d, filename)):
                tried.append(os.path.join(d, filename))
                return os.path.join(d, filename)
        raise FileNotFoundError('Not found: {}. Tried: {}'.format(filename, ', '.join(tried)))

    @abc.abstractmethod
    def loadmask(self, name: str) -> np.ndarray:
        """Load a mask file."""

    @abc.abstractmethod
    def loadcurve(self, fsn: int) -> Curve:
        """Load a radial scattering curve"""

    def get_subpath(self, subpath: str):
        """Search a file or directory relative to the base path"""
        for d in self._path:
            if os.path.exists(os.path.join(d, subpath)):
                return os.path.join(d, subpath)
        raise FileNotFoundError
