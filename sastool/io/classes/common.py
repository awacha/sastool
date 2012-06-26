'''
Created on Jun 15, 2012

@author: andris
'''

import h5py

from ... import misc


class SASExposureException(misc.SASException):
    "This exception is raised when averaging incompatible data"
    pass

class SASMaskException(misc.SASException):
    "This exception is raised if something is not OK with the mask"
    pass



class _HDF_parse_group(object):
    def __init__(self, hdf_argument, dirs = None):
        self.hdf_argument = hdf_argument
        self.hdf_file = None
        self.hdf_group = None
        self.dirs = None
    def __enter__(self):
        if isinstance(self.hdf_argument, basestring):
            self.hdf_file = h5py.highlevel.File(misc.findfileindirs(self.hdf_argument, self.dirs))
            self.hdf_group = self.hdf_file
        elif isinstance(self.hdf_argument, h5py.highlevel.File):
            self.hdf_file = self.hdf_argument
            self.hdf_group = self.hdf_file
        elif isinstance(self.hdf_argument, h5py.highlevel.Group):
            self.hdf_file = self.hdf_argument.file
            self.hdf_group = self.hdf_argument
        else:
            raise ValueError
        return self.hdf_group
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if isinstance(self.hdf_argument, basestring) and self.hdf_file is not None:
            self.hdf_file.close()
