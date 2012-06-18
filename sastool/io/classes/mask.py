'''
Created on Jun 15, 2012

@author: andris
'''

import numpy as np
import os
import scipy
import matplotlib.nxutils
import matplotlib.pyplot as plt

from common import _HDF_parse_group, SASMaskException
from ... import misc
from .. import twodim


class SASMask(object):
    """Class to represent mask matrices.
    
    Each mask matrix should have a mask ID (a string starting with 'mask'),
    which should be unique. If a single pixel changes, a new ID should be
    created.
    
    This class can be instantiated by several ways:
    
    >>> mask=SASMask(<2d numpy array>)
    
    >>> mask=SASMask(<filename>, [<dirs>])
    supported file formats: .mat, .npz, .npy, .edf
    
    >>> mask=SASMask(<other SASMask instance>)
    
    Under the hood:
        the mask matrix is kept with dtype==np.uint8. The elements should only
        be 0-s (masked) or 1-s (unmasked), otherwise unexpected events may
        occur. The constructor and the member functions take care to preserve
        this consistency.
    """
    maskid = None
    _mask = None
    def __init__(self, maskmatrix = None, dirs = None):
        super(SASMask, self).__init__()
        if maskmatrix is not None:
            if isinstance(maskmatrix, basestring) and \
                maskmatrix.lower()[-4:] in ['.mat', '.npz', '.npy']:
                self.read_from_mat(misc.findfileindirs(maskmatrix, dirs))
            elif isinstance(maskmatrix, basestring) and \
                maskmatrix.lower()[-4:] in ['.edf']:
                self.read_from_edf(misc.findfileindirs(maskmatrix, dirs))
            elif isinstance(maskmatrix, np.ndarray):
                self.mask = maskmatrix.astype(np.uint8)
                self.maskid = 'mask' + misc.random_str(6)
            elif isinstance(maskmatrix, SASMask):
                maskmatrix.copy_into(self)
            else:
                raise NotImplementedError
        else:
            raise ValueError('Empty SASMasks cannot be instantiated.')

    def __unicode__(self):
        return u'SASMask(' + self.maskid + ')'
    __str__ = __unicode__
    __repr__ = __unicode__
    def _setmask(self, maskmatrix):
        self._mask = (maskmatrix != 0).astype(np.uint8)
    def _getmask(self):
        return self._mask
    mask = property(_getmask, _setmask, doc = 'Mask matrix')
    def _getshape(self):
        return self._mask.shape
    shape = property(_getshape, doc = 'Shortcut to the shape of the mask matrix')
    def copy_into(self, into):
        """Helper function for deep copy."""
        if not isinstance(into, type(self)):
            raise ValueError('Incompatible class!')
        if self.mask is not None:
            into.mask = self.mask.copy()
        else:
            into.mask = None
        into.maskid = self.maskid
    def read_from_edf(self, filename):
        """Read a mask from an EDF file."""
        edf = twodim.readedf(filename)
        self.maskid = os.path.splitext(os.path.split(edf['FileName'])[1])[0]
        self.mask = (np.absolute(edf['data'] - edf['Dummy']) > edf['DDummy']).reshape(edf['data'].shape)
        return self
    def read_from_mat(self, filename, fieldname = None):
        """Try to load a maskfile (Matlab(R) matrix file or numpy npz/npy)
        
        Inputs:
            filename: the input file name
            fieldname: field in the mat/npz file. None to autodetect.
        """
        if filename.lower().endswith('.mat'):
            f = scipy.io.loadmat(filename)
        elif filename.lower().endswith('.npz'):
            f = np.load(filename)
        elif filename.lower().endswith('.npy'):
            f = dict([(os.path.splitext(os.path.split(filename)[1])[0], np.load(filename))])
        else:
            raise ValueError('Invalid file name format!')

        if f is None:
            raise IOError('Cannot find mask file %s!' % filename)
        if fieldname is None:
            validkeys = [k for k in f.keys() if not (k.startswith('_') and k.endswith('_'))]
            if len(validkeys) < 1:
                raise ValueError('mask file contains no masks!')
            if len(validkeys) > 1:
                raise ValueError('mask file contains multiple masks!')
            fieldname = validkeys[0]
        elif fieldname not in f:
            raise ValueError('Mask %s not in the file!' % fieldname)
        self.maskid = fieldname
        self.mask = f[fieldname].astype(np.uint8)
    def write_to_mat(self, filename):
        """Save this mask to a Matlab(R) .mat or a numpy .npy or .npz file.
        """
        if filename.lower().endswith('.mat'):
            scipy.io.savemat(filename, {self.maskid:self.mask})
        elif filename.lower().endswith('.npz'):
            np.savez(filename, **{self.maskid:self.mask}) #IGNORE:W0142
        elif filename.lower().endswith('.npy'):
            np.save(filename, self.mask)
        else:
            raise ValueError('File name %s not understood (should end with .mat or .npz).' % filename)
    def write_to_hdf5(self, hdf_entity):
        """Write this mask as a HDF5 dataset.
        
        Input:
            hdf_entity: either a HDF5 filename or an open file (instance of
                h5py.highlevel.File) or a HDF5 group (instance of
                h5py.highlevel.Group). A new dataset will be created with the
                name equal to the maskid.
        """
        with _HDF_parse_group(hdf_entity) as hpg:
            if self.maskid in hpg.keys():
                del hpg[self.maskid]
            hpg.create_dataset(self.maskid, data = self.mask, compression = 'gzip')
    def read_from_hdf5(self, hdf_entity, maskid = None):
        """Read mask from a HDF5 entity.
        
        Inputs:
            hdf_entity: either a HDF5 filename or an open h5py.highlevel.File
                instance or a h5py.highlevel.Group instance.
            maskid: the name of the mask to be loaded from the HDF5 entity.
                If None and the entity contains only one dataset, it will be
                loaded. If None and the entity contains more datasets, a
                ValueError is raised.
        """
        with _HDF_parse_group(hdf_entity) as hpg:
            if len(hpg.keys()) == 0:
                raise ValueError('No datasets in the HDF5 group!')
            if maskid is None:
                if len(hpg.keys()) == 1:
                    self.maskid = hpg.keys()[0]
                    self.mask = hpg[self.maskid].value
                else:
                    raise ValueError('More than one datasets in the HDF5 group\
and maskid argument was omitted.')
            else:
                self.maskid = maskid
                self.mask = hpg[maskid].value
        return self
    @classmethod
    def new_from_hdf5(cls, hdf_entity, maskid = None):
        obj = cls()
        obj.read_from_hdf5(hdf_entity, maskid)
        return obj
    def rebin(self, xbin, ybin, enlarge = False):
        """Re-bin the mask."""
        obj = type(self)()
        obj.mask = twodim.rebinmask(self.mask.astype(np.uint8), int(xbin), int(ybin), enlarge)
        obj.maskid = self.maskid + 'bin%dx%d_%s' % (xbin, ybin, ['shrink', 'enlarge'][enlarge])
        return obj
    def invert(self):
        """Inverts the whole mask in-place"""
        self.mask = 1 - self.mask
        return self
    def edit_rectangle(self, x0, y0, x1, y1, whattodo = 'mask'):
        """Edit a rectangular part of the mask.
        
        Inputs:
            x0,y0,x1,y1: corners of the rectangle (x: row, y: column index).
            whattodo: 'mask', 'unmask' or 'invert' if the selected area should
                be masked, unmasked or inverted. 'mask' is the default.
        """
        col, row = np.meshgrid(np.arange(self.mask.shape[1]),
                            np.arange(self.mask.shape[0]))
        idx = (row >= min(x0, x1)) & (row <= max(x0, x1)) & (col <= max(y0, y1)) & (col >= min(y0, y1))
        if whattodo.lower() == 'mask':
            self.mask[idx] = 0
        elif whattodo.lower() == 'unmask':
            self.mask[idx] = 1
        elif whattodo.lower() == 'invert':
            self.mask[idx] = 1 - self.mask[idx]
        else:
            raise ValueError('Invalid value for argument \'whattodo\': ' + whattodo)
        return self
    def edit_polygon(self, x, y, whattodo = 'mask'):
        """Edit points inside a polygon.
        
        Inputs:
            x,y: list of corners of the polygon (x: row, y: column index).
            whattodo: 'mask', 'unmask' or 'invert' if the selected area should
                be masked, unmasked or inverted. 'mask' is the default.
        """

        col, row = np.meshgrid(np.arange(self.mask.shape[1]),
                            np.arange(self.mask.shape[0]))
        points = np.vstack((col.flatten(), row.flatten())).T
        points_inside = matplotlib.nxutils.points_inside_poly(points, np.vstack((y, x)).T)
        idx = points_inside.astype('bool').reshape(self.shape)
        if whattodo.lower() == 'mask':
            self.mask[idx] = 0
        elif whattodo.lower() == 'unmask':
            self.mask[idx] = 1
        elif whattodo.lower() == 'invert':
            self.mask[idx] = 1 - self.mask[idx]
        else:
            raise ValueError('Invalid value for argument \'whattodo\': ' + whattodo)
        return self


    def edit_circle(self, x0, y0, r, whattodo = 'mask'):
        """Edit a circular part of the mask.
        
        Inputs:
            x0,y0: center of the circle (x0: row, y0: column coordinate)
            r: radius of the circle
            whattodo: 'mask', 'unmask' or 'invert' if the selected area should
                be masked, unmasked or inverted. 'mask' is the default.
        """
        col, row = np.meshgrid(np.arange(self.mask.shape[1]),
                            np.arange(self.mask.shape[0]))
        idx = (((row - x0) ** 2 + (col - y0) ** 2) <= r ** 2)
        if whattodo.lower() == 'mask':
            self.mask[idx] = 0
        elif whattodo.lower() == 'unmask':
            self.mask[idx] = 1
        elif whattodo.lower() == 'invert':
            self.mask[idx] = 1 - self.mask[idx]
        else:
            raise ValueError('Invalid value for argument \'whattodo\': ' + whattodo)
        return self

    def edit_from_matrix(self, matrix, valmin = -np.inf, valmax = np.inf,
                         masknonfinite = True, whattodo = 'mask'):
        """Edit a part of the mask where the values of a given matrix of the
        same shape are between given thresholds
        
        Inputs:
            matrix: a matrix of the same shape as the mask.
            valmin, valmax: lower and upper threshold of the values in 'matrix'
            masknonfinite: if non-finite elements in the matrix should be masked
            whattodo: 'mask', 'unmask' or 'invert' if the selected area should
                be masked, unmasked or inverted. 'mask' is the default.
        """
        if matrix.shape != self.mask.shape:
            raise ValueError('Incompatible shape for the matrix!')
        idx = (matrix >= valmin) & (matrix <= valmax)
        if masknonfinite:
            self.mask[-np.isfinite(matrix)] = 0
        if whattodo.lower() == 'mask':
            self.mask[idx] = 0
        elif whattodo.lower() == 'unmask':
            self.mask[idx] = 1
        elif whattodo.lower() == 'invert':
            self.mask[idx] = 1 - self.mask[idx]
        else:
            raise ValueError('Invalid value for argument \'whattodo\': ' + whattodo)
        return self
    def edit_borders(self, left = 0, right = 0, top = 0, bottom = 0, whattodo = 'mask'):
        """Edit borders of the mask.
        
        Inputs:
            left, right, top, bottom: width at the given direction to cut
                (directions correspond to those if the mask matrix is plotted
                by matplotlib.imshow(mask,origin='upper').
            whattodo: 'mask', 'unmask' or 'invert' if the selected area should
                be masked, unmasked or inverted. 'mask' is the default.
        """
        col, row = np.meshgrid(np.arange(self.mask.shape[1]),
                            np.arange(self.mask.shape[0]))
        idx = (col < left) | (col > self.shape[1] - 1 - right) | (row < top) | (row > self.shape[0] - 1 - bottom)
        if whattodo.lower() == 'mask':
            self.mask[idx] = 0
        elif whattodo.lower() == 'unmask':
            self.mask[idx] = 1
        elif whattodo.lower() == 'invert':
            self.mask[idx] = 1 - self.mask[idx]
        else:
            raise ValueError('Invalid value for argument \'whattodo\': ' + whattodo)
        return self
    def spy(self, *args, **kwargs):
        """Plot the mask matrix with matplotlib.pyplot.spy()
        """
        plt.spy(self.mask, *args, **kwargs)
