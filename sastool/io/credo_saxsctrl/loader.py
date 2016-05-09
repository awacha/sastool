from typing import Union, List

from .header import SAXSCtrl_Header
from ...classes2 import Loader


class SAXSCtrl_Loader(Loader):
    def __init__(self, basedir: Union[str, List[str]], recursive: bool = True, exposureclass: str = 'crd'):
        super().__init__(basedir, recursive)
        self._exposureclass = exposureclass

    def loadheader(self, fsn: int):
        return SAXSCtrl_Header.new_from_file(self.find_file(self._exposureclass + '_%05d.param' % fsn))

    def loadexposure(self, fsn: int):
        return SAXSCtrl_Exposure.new_from_file(self.find_file(self._exposureclass + '_%05d.npz' % fsn))
