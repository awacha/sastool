from typing import Union, List

import scipy.io

from .exposure import Exposure
from .header import Header
from ... import classes2


class Loader(classes2.Loader):
    def __init__(self, basedir: Union[str, List[str]], recursive: bool = True, exposureclass: str = 'crd'):
        super().__init__(basedir, recursive)
        self._exposureclass = exposureclass

    def loadheader(self, fsn: int) -> Header:
        return Header.new_from_file(self.find_file(self._exposureclass + '_%05d.param' % fsn))

    def loadexposure(self, fsn: int) -> Exposure:
        header = self.loadheader(fsn)
        mask = scipy.io.loadmat(self.find_file(header.maskname))
        maskkey = [k for k in mask.keys() if not (k.startswith('_') or k.endswith('_'))][0]
        mask = mask[maskkey]
        return Exposure.new_from_file(self.find_file(self._exposureclass + '_%05d.npz' % fsn),
                                      header_data=header, mask_data=mask)
