import abc
from typing import Optional

import numpy as np

from .curve import Curve
from .header import Header
from ..misc.arithmetic import ArithmeticBase
from ..misc.errorvalue import ErrorValue
from ..utils2d.integrate import radint_fullq_errorprop


class Exposure(ArithmeticBase, metaclass=abc.ABCMeta):
    """The basic exposure class. After successful loading, it must have at least the following attributes:

    intensity: two-dimensional np.ndarray (dtype=double)
    error: two-dimensional np.ndarray (dtype=double)
    mask: two-dimensional np.ndarray (dtype=bool): True if the pixel is valid, False if it is invalid.
    header: an instance of the appropriate subclass of Header
    """

    @abc.abstractclassmethod
    def new_from_file(self, filename: str, header_data: Optional[Header] = None,
                      mask_data: Optional[np.ndarray] = None):
        """Load an exposure from a file."""

    def sum(self, only_valid=True) -> ErrorValue:
        """Calculate the sum of pixels, not counting the masked ones if only_valid is True."""
        if not only_valid:
            mask = 1
        else:
            mask = self.mask
        return ErrorValue((self.intensity * mask).sum(),
                          ((self.error * mask) ** 2).sum() ** 0.5)

    def mean(self, only_valid=True) -> ErrorValue:
        """Calculate the mean of the pixels, not counting the masked ones if only_valid is True."""
        if not only_valid:
            intensity = self.intensity
            error = self.error
        else:
            intensity = self.intensity[self.mask]
            error = self.error[self.mask]
        return ErrorValue(intensity.mean(),
                          (error ** 2).mean() ** 0.5)

    @property
    def twotheta(self) -> np.ndarray:
        """Calculate the two-theta array"""
        row, column = np.ogrid[0:self.shape[0], 0:self.shape[0]]
        rho = (((row - self.header.beamcentery) * self.header.pixelsizey) ** 2 +
               ((column - self.header.beamcenterx) * self.header.pixelsizex) ** 2) ** 0.5
        return np.arctan(rho / self.header.distance.val)

    @property
    def shape(self) -> tuple:
        """The shape of the matrices"""
        return self.intensity.shape

    def plot2d(self, *args, **kwargs):
        raise NotImplementedError

    def __iadd__(self, other):
        if isinstance(other, Exposure):
            self.error = (self.error ** 2 + other.error ** 2) ** 0.5
            self.intensity = self.intensity + other.intensity
        elif isinstance(other, ErrorValue):
            self.error = (self.error ** 2 + other.err ** 2) ** 0.5
            self.intensity = (self.intensity + other.val)
        elif isinstance(other, int) or isinstance(other, float) or isinstance(other, np.ndarray) or isinstance(other,
                                                                                                               complex):
            self.intensity = self.intensity + other
            # self.error remains the same.
        else:
            return NotImplemented

    def __imul__(self, other):
        if isinstance(other, Exposure):
            self.error = (self.error ** 2 * other.intensity ** 2 + other.error ** 2 * self.intensity ** 2) ** 0.5
            self.intensity = self.intensity * other.intensity
        elif isinstance(other, ErrorValue):
            self.error = (self.error ** 2 * other.val ** 2 + other.err ** 2 * self.intensity ** 2) ** 0.5
            self.intensity = (self.intensity * other.val)
        elif isinstance(other, int) or isinstance(other, float) or isinstance(other, np.ndarray) or isinstance(other,
                                                                                                               complex):
            self.intensity = self.intensity * other
            # self.error remains the same.
        else:
            return NotImplemented

    def __neg__(self):
        obj = type(self)()
        obj.intensity = -self.intensity
        obj.error = self.error
        obj.mask = self.mask
        obj.header = self.header
        return obj

    def __reciprocal__(self):
        obj = type(self)()
        obj.error = self.error / (self.intensity ** 2)
        obj.intensity = 1 / self.intensity
        obj.mask = self.mask
        obj.header = self.header
        return obj

    def radial_average(self, qrange=None, pixel=False, returnmask=False,
                       errorpropagation=3, abscissa_errorpropagation=3):
        """Do a radial averaging

        Inputs:
            qrange: the q-range. If None, auto-determine. If 'linear', auto-determine
                with linear spacing (same as None). If 'log', auto-determine
                with log10 spacing.
            pixel: do a pixel-integration (instead of q)
            returnmask: if the effective mask matrix is to be returned.
            errorpropagation: the type of error propagation (3: highest of squared or
                std-dev, 2: squared, 1: linear, 0: independent measurements of
                the same quantity)
            abscissa_errorpropagation: the type of the error propagation in the
                abscissa (3: highest of squared or std-dev, 2: squared, 1: linear,
                0: independent measurements of the same quantity)

        Outputs:
            the one-dimensional curve as an instance of SASCurve (if pixel is
                False) or SASPixelCurve (if pixel is True)
            the mask matrix (if returnmask was True)
        """
        if isinstance(qrange, str):
            if qrange == 'linear':
                qrange = None
                autoqrange_linear = True
            elif qrange == 'log':
                qrange = None
                autoqrange_linear = False
            else:
                raise ValueError(
                        'Value given for qrange (''%s'') not understood.' % qrange)
        else:
            autoqrange_linear = True  # whatever
        if not pixel:
            res = radint_fullq_errorprop(self.intensity, self.error, self.header.wavelength.val,
                                         self.header.wavelength.err, self.header.distance.val,
                                         self.header.distance.err, self.header.pixelsizex.val,
                                         self.header.pixelsizey.val, self.header.beamcenterx.val,
                                         self.header.beamcenterx.err, self.header.beamcentery.val,
                                         self.header.beamcentery.err, (self.mask == 0).astype(np.uint8),
                                         qrange, returnmask=returnmask, errorpropagation=errorpropagation,
                                         autoqrange_linear=autoqrange_linear, abscissa_kind=0,
                                         abscissa_errorpropagation=abscissa_errorpropagation)
            q, dq, I, E = res[:4]
            if returnmask:
                retmask = res[5]
            c = Curve(q, I, E, dq)
        else:
            res = radint_fullq_errorprop(self.intensity, self.error, self.header.wavelength.val,
                                         self.header.wavelength.err,
                                         self.header.distance.val,
                                         self.header.distance.err,
                                         self.header.pixelsizex.val, self.header.pixelsizey.val,
                                         self.header.beamcenterx.val, self.header.beamcenterx.err,
                                         self.header.beamcentery.val, self.header.beamcentery.err,
                                         (self.mask == 0).astype(np.uint8), qrange,
                                         returnmask=returnmask, errorpropagation=errorpropagation,
                                         autoqrange_linear=autoqrange_linear, abscissa_kind=3,
                                         abscissa_errorpropagation=abscissa_errorpropagation)
            p, dp, I, E = res[:4]
            if returnmask:
                retmask = res[5]
            c = Curve(p, I, E, dp)
        if returnmask:
            return c, retmask
        else:
            return c
