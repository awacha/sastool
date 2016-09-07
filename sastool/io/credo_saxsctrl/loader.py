import os

import numpy as np
import scipy.io

from .exposure import Exposure
from .header import Header
from ... import classes2


class Loader(classes2.Loader):
    def __init__(self, basedir: str, recursive: bool = True, processed: bool = True, exposureclass: str = 'crd'):
        if processed:
            datasubdirs = ['eval2d', 'eval1d']
        else:
            datasubdirs = ['param_override', 'param', 'images']
        basepath = [basedir]
        for d in datasubdirs:
            d = os.path.join(basedir, d)
            basepath.append(d)
            if recursive:
                lis = os.listdir(d)
                for l in [l_ for l_ in lis if '.' not in l_]:
                    if os.path.isdir(os.path.join(d, l)):
                        basepath.append(os.path.join(d, l))
        basepath.append(os.path.join(basedir, 'mask'))
        if recursive:
            for d, sds, fs in os.walk(os.path.join(basedir, 'mask'), followlinks=True):
                if d not in basepath:
                    basepath.append(d)
        super().__init__(basepath, False, processed)
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
