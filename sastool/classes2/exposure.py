import abc
from typing import Optional

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

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

    def pixel_to_q(self, row: float, column: float):
        """Return the q coordinates of a given pixel.

        Inputs:
            row: float
                the row (vertical) coordinate of the pixel
            column: float
                the column (horizontal) coordinate of the pixel

        Coordinates are 0-based and calculated from the top left corner.
        """
        qrow = 4 * np.pi * np.sin(0.5 * np.arctan(
            float((row - self.header.beamcentery) * self.header.pixelsizey / self.header.distance))) / float(
            self.header.wavelength)
        qcol = 4 * np.pi * np.sin(0.5 * np.arctan((
                                                  column - self.header.beamcenterx) * self.header.pixelsizex / self.header.distance)) / self.header.wavelength
        return qrow, qcol

    def imshow(self, *args, show_crosshair=True, show_mask=True, show_qscale=True, axes=None, invalid_color='black',
               mask_opacity=0.8, show_colorbar=True, **kwargs):
        """Plot the matrix (imshow)

        Keyword arguments [and their default values]:

        show_crosshair [True]: if a cross-hair marking the beam position is to be
            plotted.
        show_mask [True]: if the mask is to be plotted.
        show_qscale [True]: if the horizontal and vertical axes are to be scaled into q
        axes [None]: the axes into which the image should be plotted. If None,
            defaults to the currently active axes (returned by plt.gca())
        invalid_color ['black']: the color for invalid (NaN or infinite) pixels
        mask_opacity [0.8]: the opacity of the overlaid mask (1 is fully opaque,
            0 is fully transparent)
        show_colorbar [True]: if a colorbar is to be added. Can be a boolean value
            (True or False) or an instance of matplotlib.axes.Axes, into which the
            color bar should be drawn.

        All other keywords are forwarded to plt.imshow() / matplotlib.Axes.imshow()

        Returns: the image instance returned by imshow()
        """
        kwargs_default = {'interpolation': 'nearest'}
        if 'origin' not in kwargs:
            kwargs['origin'] = None
        if kwargs['origin'] is None:
            kwargs['origin'] = matplotlib.rcParams['image.origin']

        if 'aspect' not in kwargs:
            kwargs['aspect'] = 'equal'

        if show_qscale:
            ymin, xmin = self.pixel_to_q(0, 0)
            ymax, xmax = self.pixel_to_q(*self.shape)
            if kwargs['origin'].upper() == 'UPPER':
                kwargs['extent'] = [xmin, xmax, ymax, ymin]
            else:
                kwargs['extent'] = [xmin, xmax, ymin, ymax]
            bcx = 0
            bcy = 0
        else:
            bcx = self.header.beamcenterx
            bcy = self.header.beamcentery
            xmin = 0
            xmax = self.shape[1]
            ymin = 0
            ymax = self.shape[0]
            if kwargs['origin'].upper() == 'UPPER':
                kwargs['extent'] = [0, self.shape[1], self.shape[0], 0]
            else:
                kwargs['extent'] = [0, self.shape[1], 0, self.shape[0]]
        if axes is None:
            axes = plt.gca()
        ret = axes.imshow(self.intensity, **kwargs)
        if show_mask:
            # workaround: because of the colour-scaling we do here, full one and
            #   full zero masks look the SAME, i.e. all the image is shaded.
            #   Thus if we have a fully unmasked matrix, skip this section.
            #   This also conserves memory.
            if (self.mask == 0).sum():  # there are some masked pixels
                # we construct another representation of the mask, where the masked pixels are 1.0, and the
                # unmasked ones will be np.nan. They will thus be not rendered.
                mf = np.ones(self.mask.shape, np.float)
                mf[self.mask != 0] = np.nan
                axes.imshow(mf, cmap=matplotlib.cm.gray_r, interpolation='nearest', alpha=mask_opacity,
                            extent=kwargs['extent'], origin=kwargs['origin'], aspect=kwargs['aspect'])
        if show_crosshair:
            ax = axes.axis()  # save zoom state
            axes.plot([xmin, xmax], [bcy] * 2, 'w-')
            axes.plot([bcx] * 2, [ymin, ymax], 'w-')
            axes.axis(ax)  # restore zoom state
        axes.set_axis_bgcolor(invalid_color)
        if show_colorbar:
            if isinstance(show_colorbar, matplotlib.axes.Axes):
                axes.figure.colorbar(
                        ret, cax=show_colorbar)
            else:
                # try to find a suitable colorbar axes: check if the plot target axes already
                # contains some images, then check if their colorbars exist as
                # axes.
                cax = [i.colorbar[1]
                       for i in axes.images if i.colorbar is not None]
                cax = [c for c in cax if c in c.figure.axes]
                if cax:
                    cax = cax[0]
                else:
                    cax = None
                axes.figure.colorbar(ret, cax=cax, ax=axes)
        axes.figure.canvas.draw()
        return ret

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
        return self

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
        return self

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

    def __copy__(self):
        c = type(self)()
        c.error = self.error
        c.intensity = self.intensity
        c.mask = self.mask
        c.header = self.header
        return c

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
                                         self.header.distance.err, self.header.pixelsizey.val,
                                         self.header.pixelsizex.val, self.header.beamcentery.val,
                                         self.header.beamcentery.err, self.header.beamcenterx.val,
                                         self.header.beamcenterx.err, (self.mask == 0).astype(np.uint8),
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
                                         self.header.pixelsizey.val, self.header.pixelsizex.val,
                                         self.header.beamcentery.val, self.header.beamcentery.err,
                                         self.header.beamcenterx.val, self.header.beamcenterx.err,
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
