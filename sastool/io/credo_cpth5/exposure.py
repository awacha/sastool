from typing import Optional

import numpy as np
import h5py

from .header import Header
from ... import classes2


class Exposure(classes2.Exposure):
    @classmethod
    def new_from_file(cls, filename: str, samplename:str, dist:float):
        with h5py.File(filename) as f:
            dist = sorted([d for d in f['Samples'][samplename].keys()], key=lambda d:abs(float(d)-dist))[0]
            return cls.new_from_group(f['Samples'][samplename][dist])

    @classmethod
    def new_from_group(cls, group:h5py.Group):
        ex = cls()
        ex.intensity = np.array(group['image'])
        ex.error = np.array(group['image_uncertainty'])
        ex.mask = np.array(group['mask'])
        ex.header = Header.new_from_group(group)
        return ex
