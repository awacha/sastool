import os
import weakref

import numpy as np
import scipy.io

from .exposure import Exposure
from .header import Header
from ... import classes2


class Loader(classes2.Loader):
    def __init__(self, basedir: str, recursive: bool = False, processed: bool = True, exposureclass: str = 'crd'):
        if processed:
            datasubdirs = ['eval2d', os.path.join('eval2d', exposureclass), 'eval1d']
            headersubdirs = datasubdirs
        else:
            datasubdirs = [os.path.join('images', exposureclass), 'images']
            headersubdirs = ['param_override', os.path.join('param_override', exposureclass),
                             'param', os.path.join('param', exposureclass)]
        basedir = os.path.expanduser(basedir)
        basepath = [os.path.join(basedir, sd) for sd in datasubdirs]
        headerpath = [os.path.join(basedir, sd) for sd in headersubdirs]
        maskpath = []
        for d, sds, fs in os.walk(os.path.join(basedir, 'mask'), followlinks=True):
            maskpath.append(d)
        super().__init__(basepath, recursive, processed, headerpath=headerpath, maskpath=maskpath)
        self._exposureclass = exposureclass

    def loadheader(self, fsn: int) -> Header:
        return Header.new_from_file(self.find_file(self._exposureclass + '_%05d.param' % fsn, what='header'))

    def loadexposure(self, fsn: int) -> Exposure:
        header = self.loadheader(fsn)
        mask = self.loadmask(header.maskname)
        if self.processed:
            ex = Exposure.new_from_file(self.find_file(self._exposureclass + '_%05d.npz' % fsn, what='exposure'),
                                        header_data=header, mask_data=mask)
        else:
            ex = Exposure.new_from_file(self.find_file(self._exposureclass + '_%05d.cbf' % fsn, what='exposure'),
                                        header_data=header, mask_data=mask)
        ex.loader = weakref.proxy(self)
        return ex

    def loadmask(self, filename: str) -> np.ndarray:
        """Load a mask file."""
        mask = scipy.io.loadmat(self.find_file(filename, what='mask'))
        maskkey = [k for k in mask.keys() if not (k.startswith('_') or k.endswith('_'))][0]
        return mask[maskkey].astype(np.bool)

    def loadcurve(self, fsn: int) -> classes2.Curve:
        """Load a radial scattering curve"""
        return classes2.Curve.new_from_file(self.find_file(self._exposureclass + '_%05d.txt' % fsn))
