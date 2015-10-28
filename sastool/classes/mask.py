'''
Created on Jun 15, 2012

@author: andris
'''
#, unicode_literals
from __future__ import print_function
import numpy as np
import os
import scipy.io

try:
    import matplotlib.nxutils

    def pointinsidepolygon(points, vertices):
        return matplotlib.nxutils.points_inside_poly(points, vertices)
except ImportError:
    import matplotlib.path

    def pointinsidepolygon(points, vertices):
        return matplotlib.path.Path(vertices).contains_points(points)
import matplotlib.pyplot as plt
import h5py
import numbers
import collections
import logging
import sys
if sys.version_info[0] == 3:
    basestring = str


from .common import _HDF_parse_group, SASMaskException
from .. import misc
from ..io import twodim  # IGNORE:E0611

__all__ = ['SASMask']
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    supported_read_extensions = ['.mat', '.npz', '.npy', '.edf', '.sma']

    @staticmethod
    def _set_default_kwargs(kwargs):
        if 'dirs' not in kwargs:
            kwargs['dirs'] = None
        if 'maskid' not in kwargs:
            kwargs['maskid'] = 'mask' + misc.random_str(6)
        return kwargs

    def __init__(self, maskmatrix=None, **kwargs):
        if maskmatrix is None:
            raise ValueError('Empty SASMasks canot be instantiated!')
        logger.debug('SASMask.__init__. MaskMatrix: type: ' +
                     str(type(maskmatrix)) + ' value: ' + repr(maskmatrix))
        kwargs = self._set_default_kwargs(kwargs)
        super(SASMask, self).__init__()
        if isinstance(maskmatrix, basestring) and \
                maskmatrix.strip().lower()[-4:] in ['.mat', '.npz', '.npy']:
            # load from a '.mat', '.npz' or '.npy' file
            self.read_from_mat(maskmatrix, **kwargs)
        elif isinstance(maskmatrix, basestring) and \
                maskmatrix.lower()[-4:] in ['.edf']:
            # load from an EDF file
            self.read_from_edf(maskmatrix, **kwargs)
        elif isinstance(maskmatrix, basestring) and \
                maskmatrix.lower()[-4:] in ['.sma']:
            self.read_from_sma(maskmatrix)
        elif isinstance(maskmatrix, np.ndarray):
            # convert a numpy array to a SASMask
            self.mask = (maskmatrix != 0).astype(np.uint8)
            self.maskid = kwargs['maskid']
        elif isinstance(maskmatrix, SASMask):
            # make a copy of an existing SASMask instance
            self.mask = maskmatrix.mask.copy()
            self.maskid = maskmatrix.maskid
        elif isinstance(maskmatrix, h5py.highlevel.Group) or \
            (isinstance(maskmatrix, str) and
             (maskmatrix.lower().endswith('.h5') or maskmatrix.lower().endswith('.hdf5'))):
            self.read_from_hdf5(maskmatrix, **kwargs)
        elif isinstance(maskmatrix, numbers.Integral):
            self.mask = np.zeros((maskmatrix, maskmatrix), dtype=np.uint8)
            self.maskid = kwargs['maskid']
        elif isinstance(maskmatrix, collections.Sequence) and len(maskmatrix) >= 2:
            self.mask = np.zeros(maskmatrix[:2], dtype=np.uint8)
            self.maskid = kwargs['maskid']
        elif isinstance(maskmatrix, basestring):
            raise IOError('Could not open mask file: ' + maskmatrix)
        else:
            raise NotImplementedError

    def __unicode__(self):
        return 'SASMask(' + self.maskid + ')'
    __str__ = __unicode__
    __repr__ = __unicode__

    def _setmask(self, maskmatrix):
        self._mask = (maskmatrix != 0).astype(np.uint8)

    def _getmask(self):
        return self._mask
    mask = property(_getmask, _setmask, doc='Mask matrix')

    def read_from_edf(self, filename, **kwargs):
        """Read a mask from an EDF file."""
        kwargs = self._set_default_kwargs(kwargs)
        filename = misc.findfileindirs(filename, kwargs['dirs'])
        edf = twodim.readedf(filename)
        self.maskid = os.path.splitext(os.path.split(edf['FileName'])[1])[0]
        self.mask = (np.absolute(edf['data'] - edf['Dummy']) > edf['DDummy']).reshape(
            edf['data'].shape)
        self.filename = filename
        return self

    def read_from_mat(self, filename, fieldname=None, **kwargs):
        """Try to load a maskfile (Matlab(R) matrix file or numpy npz/npy)

        Inputs:
            filename: the input file name
            fieldname: field in the mat/npz file. None to autodetect.
        """
        kwargs = self._set_default_kwargs(kwargs)
        filename = misc.findfileindirs(filename, kwargs['dirs'])
        if filename.lower().endswith('.mat'):
            f = scipy.io.loadmat(filename)
        elif filename.lower().endswith('.npz'):
            f = np.load(filename)
        elif filename.lower().endswith('.npy'):
            f = dict(
                [(os.path.splitext(os.path.split(filename)[1])[0], np.load(filename))])
        else:
            raise ValueError('Invalid file name format!')

        if f is None:
            raise IOError('Cannot find mask file %s!' % filename)
        if fieldname is None:
            validkeys = [k for k in list(f.keys()) if not (
                k.startswith('_') and k.endswith('_'))]
            if len(validkeys) < 1:
                raise ValueError('mask file contains no masks!')
            if len(validkeys) > 1:
                raise ValueError('mask file contains multiple masks!')
            fieldname = validkeys[0]
        elif fieldname not in f:
            raise ValueError('Mask %s not in the file!' % fieldname)
        self.maskid = fieldname
        self.mask = f[fieldname].astype(np.uint8)
        self.filename = filename

    def read_from_sma(self, filename):
        """Load a mask from a *.sma file (produced by BerSANS)
        """
        self.mask = twodim.readBerSANSmask(filename).astype(np.uint8)
        self.maskid = filename.split('.')[0]
        self.filename = filename

    def write_to_sma(self, filename=None):
        """Write mask a *.sma file (produced by BerSANS)"""
        if filename is None:
            filename = self.maskid + '.sma'
        twodim.writeBerSANSmask(filename, self.mask)

    def write_to_mat(self, filename=None):
        """Save this mask to a Matlab(R) .mat or a numpy .npy or .npz file.
        """
        if filename is None:
            filename = self.maskid + '.mat'
        if filename.lower().endswith('.mat'):
            scipy.io.savemat(filename, {self.maskid: self.mask})
        elif filename.lower().endswith('.npz'):
            np.savez_compressed(
                filename, **{self.maskid: self.mask})  # IGNORE:W0142
        elif filename.lower().endswith('.npy'):
            np.save(filename, self.mask)
        else:
            raise ValueError(
                'File name %s not understood (should end with .mat or .npz).' % filename)

    def write_to_hdf5(self, hdf_entity, **kwargs):
        """Write this mask as a HDF5 dataset.

        Input:
            hdf_entity: either a HDF5 filename or an open file (instance of
                h5py.highlevel.File) or a HDF5 group (instance of
                h5py.highlevel.Group). A new dataset will be created with the
                name equal to the maskid.
        """
        kwargs = self._set_default_kwargs(kwargs)
        with _HDF_parse_group(hdf_entity, kwargs['dirs']) as hpg:
            if self.maskid in list(hpg.keys()):
                del hpg[self.maskid]
            hpg.create_dataset(self.maskid, data=self.mask, compression='gzip')

    def read_from_hdf5(self, hdf_entity, maskid=None, **kwargs):
        """Read mask from a HDF5 entity.

        Inputs:
            hdf_entity: either a HDF5 filename or an open h5py.highlevel.File
                instance or a h5py.highlevel.Group instance.
            maskid: the name of the mask to be loaded from the HDF5 entity.
                If None and the entity contains only one dataset, it will be
                loaded. If None and the entity contains more datasets, a
                ValueError is raised.
        """
        kwargs = self._set_default_kwargs(kwargs)
        with _HDF_parse_group(hdf_entity, kwargs['dirs']) as hpg:
            if len(list(hpg.keys())) == 0:
                raise ValueError('No datasets in the HDF5 group!')
            if maskid is None:
                if len(list(hpg.keys())) == 1:
                    self.maskid = list(hpg.keys())[0]
                    self.mask = hpg[self.maskid].value
                    self.filename = hpg.file.filename + \
                        ':' + hpg[self.maskid].name
                else:
                    raise ValueError('More than one datasets in the HDF5 group\
and maskid argument was omitted.')
            else:
                if maskid not in list(hpg.keys()):
                    maskid_new = os.path.splitext(os.path.basename(maskid))[0]
                    if maskid_new not in list(hpg.keys()):
                        raise ValueError(
                            'Cannot find mask with ID %s in the HDF5 group!' % maskid)
                    else:
                        maskid = maskid_new
                self.mask = hpg[maskid].value
                self.filename = hpg.file.filename + ':' + hpg[self.maskid].name
                self.maskid = maskid
        return self

    def rebin(self, xbin, ybin, enlarge=False):
        """Re-bin the mask."""
        mask = twodim.rebinmask(
            self.mask.astype(np.uint8), int(xbin), int(ybin), enlarge)
        maskid = self.maskid + \
            'bin%dx%d_%s' % (xbin, ybin, ['shrink', 'enlarge'][enlarge])
        return type(self)(mask, maskid=maskid)

    def invert(self):
        """Inverts the whole mask in-place"""
        self.mask = 1 - self.mask
        return self

    def edit_rectangle(self, x0, y0, x1, y1, whattodo='mask'):
        """Edit a rectangular part of the mask.

        Inputs:
            x0,y0,x1,y1: corners of the rectangle (x: row, y: column index).
            whattodo: 'mask', 'unmask' or 'invert' if the selected area should
                be masked, unmasked or inverted. 'mask' is the default.
        """
        col, row = np.meshgrid(np.arange(self.mask.shape[1]),
                               np.arange(self.mask.shape[0]))
        idx = (row >= min(x0, x1)) & (row <= max(x0, x1)) & (
            col <= max(y0, y1)) & (col >= min(y0, y1))
        return self.edit_general(idx, whattodo)

    def edit_polygon(self, x, y, whattodo='mask'):
        """Edit points inside a polygon.

        Inputs:
            x,y: list of corners of the polygon (x: row, y: column index).
            whattodo: 'mask', 'unmask' or 'invert' if the selected area should
                be masked, unmasked or inverted. 'mask' is the default.
        """

        col, row = np.meshgrid(np.arange(self.mask.shape[1]),
                               np.arange(self.mask.shape[0]))
        points = np.vstack((col.flatten(), row.flatten())).T
        points_inside = pointinsidepolygon(points, np.vstack((y, x)).T)
        idx = points_inside.astype('bool').reshape(self.shape)
        return self.edit_general(idx, whattodo)

    def edit_circle(self, x0, y0, r, whattodo='mask'):
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
        return self.edit_general(idx, whattodo)

    def edit_from_matrix(self, matrix, valmin=-np.inf, valmax=np.inf,
                         masknonfinite=True, whattodo='mask'):
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
            idx &= (-np.isfinite(matrix))
        return self.edit_general(idx, whattodo)

    def edit_borders(self, left=0, right=0, top=0, bottom=0, whattodo='mask'):
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
        idx = (col < left) | (col > self.shape[1] - 1 - right) | (
            row < top) | (row > self.shape[0] - 1 - bottom)
        return self.edit_general(idx, whattodo)

    def edit_general(self, idx, whattodo):
        if whattodo.lower() == 'mask':
            self.mask[idx] = 0
        elif whattodo.lower() == 'unmask':
            self.mask[idx] = 1
        elif whattodo.lower() == 'invert':
            self.mask[idx] = 1 - self.mask[idx]
        else:
            raise ValueError(
                'Invalid value for argument \'whattodo\': ' + whattodo)
        return self

    def spy(self, *args, **kwargs):
        """Plot the mask matrix with matplotlib.pyplot.spy()
        """
        plt.spy(self.mask, *args, **kwargs)

    def __array__(self, dt=np.uint8):
        return self.mask.astype(dt)

    @property
    def shape(self):
        return self.mask.shape

    def edit_gaps(self, module_rows=195, module_columns=487, gap_rows=17, gap_columns=7, first_row=0, first_column=0, whattodo='mask'):
        col, row = np.meshgrid(np.arange(self.mask.shape[1]),
                               np.arange(self.mask.shape[0]))
        idx = ((col - first_column) % (module_columns + gap_columns) >= module_columns) | \
            ((row - first_row) % (module_rows + gap_rows) >= module_rows)
        return self.edit_general(idx, whattodo)

    def edit_nonfinite(self, matrix, whattodo='mask'):
        return self.edit_function(matrix, lambda a: -np.isfinite(a), whattodo)

    def edit_nonpositive(self, matrix, whattodo='mask'):
        return self.edit_function(matrix, lambda a: a <= 0, whattodo)

    def edit_function(self, matrix, func, whattodo='mask'):
        return self.edit_general(func(matrix), whattodo)

    def edit_badpixels(self, matrix, factor=100, whattodo='mask'):
        """Find pixels whose intensity is much larger than its neighbours'.

        Inputs:
        -------
            matrix: np.ndarray
                the scattering matrix to be masked
            factor: positive number
                this value controls what is deemed as "much larger": those pixels,
                where:

                abs(pix-mean)<std*factor

                where pix is the pixel intensity and mean and std are the mean and std
                intensities of all its 8 neighbours.
        """
        assert(matrix.shape == self.shape)
        sides = np.array(
            (matrix[2:, 1:-1], matrix[:-2, 1:-1], matrix[1:-1, 2:],
             matrix[1:-1, :-2], matrix[2:, 2:], matrix[:-2, :-2],
             matrix[2:, :-2], matrix[:-2, 2:]))
        mean = sides.mean(axis=0)
        std = sides.std(axis=0)
        idx = np.zeros(self.shape, np.bool)
        idx[1:-1, 1:-1] = np.absolute(matrix[1:-1, 1:-1] - mean) < factor * std
        return self.edit_general(idx, whattodo)

    def __getitem__(self, key):
        m = self.mask[key]
        if isinstance(m, np.ndarray):
            return self.__class__(self.mask[key], maskid=self.maskid + '$trim')
        else:
            return m

    def __and__(self, other):
        obj = SASMask(self)
        obj &= other
        return obj

    def __iand__(self, other):
        if not isinstance(other, SASMask):
            return NotImplemented
        if not (self.shape == other.shape):
            raise ValueError('Shape mismatch')
        self.mask &= other.mask
        return self

    def __or__(self, other):
        obj = SASMask(self)
        obj |= other
        return obj

    def __ior__(self, other):
        if not isinstance(other, SASMask):
            return NotImplemented
        if not (self.shape == other.shape):
            raise ValueError('Shape mismatch')
        self.mask |= other.mask
        return self

    def __xor__(self, other):
        obj = SASMask(self)
        obj ^= other
        return obj

    def __ixor__(self, other):
        if not isinstance(other, SASMask):
            return NotImplemented
        if not (self.shape == other.shape):
            raise ValueError('Shape mismatch')
        self.mask ^= other.mask
        return self
