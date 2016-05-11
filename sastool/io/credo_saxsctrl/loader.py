import os

import numpy as np
import scipy.io

from .exposure import Exposure
from .header import Header
from ... import classes2


class Loader(classes2.Loader):
    def __init__(self, basedir: str, recursive: bool = True, processed: bool = True, exposureclass: str = 'crd'):
        if processed:
            super().__init__([os.path.join(basedir, subdir) for subdir in ['eval2d', 'mask', 'eval1d']],
                             recursive, processed)
        else:
            super().__init__(
                    [os.path.join(basedir, subdir) for subdir in ['param_override', 'param', 'images', 'mask']],
                    recursive, processed)
        self._exposureclass = exposureclass

    def loadheader(self, fsn: int) -> Header:
        return Header.new_from_file(self.find_file(self._exposureclass + '_%05d.param' % fsn))

    def loadexposure(self, fsn: int) -> Exposure:
        header = self.loadheader(fsn)
        mask = self.loadmask(header.maskname)
        if self.processed:
            return Exposure.new_from_file(self.find_file(self._exposureclass + '_%05d.npz' % fsn),
                                          header_data=header, mask_data=mask)
        else:
            return Exposure.new_from_file(self.find_file(self._exposureclass + '_%05d.cbf' % fsn),
                                          header_data=header, mask_data=mask)

    def loadmask(self, filename: str) -> np.ndarray:
        """Load a mask file."""
        mask = scipy.io.loadmat(self.find_file(filename))
        maskkey = [k for k in mask.keys() if not (k.startswith('_') or k.endswith('_'))][0]
        return mask[maskkey].astype(np.bool)

    def loadcurve(self, fsn: int) -> classes2.Curve:
        """Load a radial scattering curve"""
        return classes2.Curve.new_from_file(self.find_file(self._exposureclass + '_%05d.txt' % fsn))
