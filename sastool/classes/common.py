'''
Created on Jun 15, 2012

@author: andris
'''

import h5py

from sastool import misc

__all__ = ['SASExposureException', 'SASMaskException']

class SASExposureException(misc.SASException):
    "This exception is raised when averaging incompatible data"
    pass

class SASMaskException(misc.SASException):
    "This exception is raised if something is not OK with the mask"
    pass



class _HDF_parse_group(object):
    hdf_group = None
    we_opened = False
    def __init__(self, hdf_argument, dirs=None, mode='r'):
        self.hdf_argument = hdf_argument
        self.hdf_group = None
        self.hdf_mode = mode
        self.dirs = dirs
    def __enter__(self):
        if isinstance(self.hdf_argument, str):
            try:
                filename = misc.findfileindirs(self.hdf_argument, self.dirs)
            except IOError:
                if self.hdf_mode.lower() in ['a', 'w', 'w-']:
                    filename = self.hdf_argument
            self.hdf_group = h5py.highlevel.File(filename, self.hdf_mode)
            self.we_opened = True
        elif isinstance(self.hdf_argument, h5py.highlevel.Group):
            self.hdf_group = self.hdf_argument
        else:
            raise ValueError
        return self.hdf_group
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.we_opened:
            self.hdf_group.close()
