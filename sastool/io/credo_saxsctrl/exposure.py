from typing import Optional

import numpy as np

from .header import Header
from ... import classes2


class Exposure(classes2.Exposure):
    @classmethod
    def new_from_file(cls, filename: str, header_data: Optional[Header] = None, mask_data: Optional[np.ndarray] = None):
        data = np.load(filename)
        ex = cls()
        ex.intensity = data['Intensity']
        ex.error = data['Error']
        if header_data is None:
            raise ValueError('Header data needs to be supplied')
        ex.header = header_data
        if mask_data is None:
            mask_data = np.ones(ex.shape, dtype=np.bool)
        ex.mask = mask_data
        del data
        return ex
