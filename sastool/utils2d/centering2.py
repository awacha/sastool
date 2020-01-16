"""Beam center finding algorithms"""
from typing import Sequence, Tuple, Optional, Dict

import numpy as np
import scipy.optimize

from . import integrate2


class Centering:
    """Find the beam center on a scattering pattern using various algorithms

    Beam center coordinates are (row, column), starting from 0.

    mask: True if pixel is valid, False if invalid

    Various algorithms are implemented using cost function methods. The first argument of these is always the beam
    center coordinate to be tested. You can check other required arguments in the docstring of the cost function.

    Algorithms implemented:
        - coi: center of intensity
        - slices: matching of opposite slices of the scattering curves at 45°, 135°, -135° and -45°.
        - azimuthal: the azimuthal scattering curve in a given annulus must be flat (small standard deviation)
        - azimuthal_fold: the azimuthal scattering curve must be periodic to pi
        - peak_amplitude: the height of the peak fitted to a selected range of the scattering curve is maximized
        - peak_width: the width of the Lorentzian peak fitted to a selected range of the scattering curve is minimized
        - powerlaw: power-law fit to a selected range of the scattering curve must be good (Chi^2 parameter minimized).
    """
    matrix: np.ndarray
    mask: np.ndarray
    center: Tuple[float, float]
    algorithmdescription: Dict[str, str] = {
        'coi': 'Determine the center of gravity (intensity) of a selected part of an image. Typically an exposure '
               'is needed where the (attenuated) direct beam is visible.',
        'slices': 'Calculate scattering curves in angular sectors centered at 45°, 135°, 225° and 315° and try to '
                  'match the curves corresponding to opposite angles (45° - 225°; 135° - 315°).',
        'azimuthal': 'Calculate the azimuthal scattering curve in a given annulus and try to make it as flat as '
                     'possible (minimize the standard deviation of the intensity.',
        'azimuthal_fold': 'Calculate the azimuthal scattering curve in a given annulus and try to make it periodic to '
                          'π by matching one half of it to the other.',
        'peak_amplitude': 'Fit a Lorentzian peak on a selected interval of the radial scattering curve and maximize '
                          'the peak height (baseline + amplitude).',
        'peak_width': 'Fit a Lorentzian peak on a selected interval of the radial scattering curve and minimize the '
                      'peak width.',
        'powerlaw': 'Fit a power-law function on a selected interval of the radial scattering curve and minimize the '
                    'Chi^2 parameter (sum of squares divided by the number of degrees of freedom).'
    }
    algorithmname: Dict[str, str] = {
        'coi': 'Center of gravity',
        'slices': 'Opposite slices',
        'azimuthal': 'Flat azimuthal curve',
        'azimuthal_fold': 'π-periodic azimuthal curve',
        'peak_amplitude': 'Peak height',
        'peak_width': 'Peak width',
        'powerlaw': 'Goodness of power-law fit',
    }

    def __init__(self, matrix: np.ndarray, mask: np.ndarray, initialcenter: Sequence[float, float]):
        self.matrix = matrix
        self.mask = mask
        self.center = (initialcenter[0], initialcenter[1])

    def findcenter(self, method:str, **kwargs) -> Tuple[float, float]:
        costfunc = getattr(self, 'costfunc_{}'.format(method))
        result = scipy.optimize.minimize(costfunc, self.center, kwargs=kwargs)
        if not result.success:
            raise ValueError('Beam center finding failed with message: {}'.format(result.message))
        self.center = tuple(result.x)
        return self.center

    def costfunc_coi(self, center: Tuple[float, float], rowmin: float = -np.inf, rowmax: float = np.inf,
                     colmin: float = -np.inf, colmax: float = np.inf) -> float:
        """Cost function for finding the center of gravity (intensity)

        :param center: current center coordinate
        :type center: (float, float)
        :param rowmin: lowest row-coordinate to take into account
        :type rowmin: float
        :param rowmax: highest row-coordinate to take into account
        :type rowmax: float
        :param colmin: lowest column-coordinate to take into account
        :type colmin: float
        :param colmax: highest column-coordinate to take into account
        :type colmax: float
        :return: the cost value. The lower the best.
        :rtype: object
        """
        rows, cols = np.ogrid[0:self.matrix.shape[0], 0:self.matrix.shape[1]]
        # x and y coordinates of vectors pointing at each pixel
        y = rows - center[0]
        x = cols - center[1]
        mask = np.logical_and(self.mask, np.logical_and(np.logical_and(rows >= rowmin, rows <= rowmax),
                                                        np.logical_and(cols >= colmin, cols <= colmax)))
        wdx = (self.matrix * x)[mask].sum()
        wdy = (self.matrix * y)[mask].sum()
        return np.hypot(wdx, wdy)

    def costfunc_slices(self, center: Tuple[float, float], dmin=0, dmax=np.inf, sector_width=30):
        curves = []
        for angle in [45, 135, -135, -45]:
            curves.append(integrate2.fastradavg(
                self.matrix, self.mask, center[0], center[1], dmin, dmax, min(2, int((dmax - dmin) / 2)), angle,
                sector_width, sector_symmetric=False)
            )
        # each element of `curves` is (pixel, intensity, Area)
        validindices = np.logical_and(
            np.logical_and(np.isfinite(curves[0][0]), np.isfinite(curves[1][0])),
            np.logical_and(np.isfinite(curves[2][0]), np.isfinite(curves[3][0]))
        )
        if validindices.sum() < 2:
            raise ValueError('Not enough overlap between slices: maybe the beam center is off by much.')
        return ((curves[0][1] - curves[2][1]) ** 2 + (curves[1][1] - curves[3][1]) ** 2).sum() / validindices.sum()

    def costfunc_azimuthal(self, center: Tuple[float, float], dmin=0, dmax=np.inf, ntheta: int = 50):
        phi, intensity, area = integrate2.fastazimavg(self.matrix, self.mask, center[0], center[1], ntheta,
                                                      (dmin, dmax))
        return np.nanstd(intensity)

    def costfunc_azimuthal_fold(self, center: Tuple[float, float], dmin=0, dmax=np.inf, ntheta: int = 50):
        if ntheta % 2:
            raise ValueError('Argument `ntheta` must be even.')
        phi, intensity, area = integrate2.fastazimavg(self.matrix, self.mask, center[0], center[1], ntheta,
                                                      (dmin, dmax))
        return ((intensity[:ntheta // 2] - intensity[ntheta // 2:]) ** 2).sum() / ntheta * 2

    def costfunc_peak_amplitude(self, center: Tuple[float, float], dmin=0, dmax=np.inf):
        pix, intensity, area = integrate2.fastradavg(self.matrix, self.mask, center[0], center[1], dmin, dmax,
                                                     int(min(2, dmax - dmin)))
        # now get rid of possible NaNs
        area = area[np.isfinite(pix)]
        intensity = intensity[np.isfinite(pix)]
        pix = pix[np.isfinite(pix)]
        popt, pcov = scipy.optimize.curve_fit(
            self._lorentzian, pix, intensity, [
                0.5*(intensity[0]+intensity[-1]),   # first estimate for the baseline
                intensity.max() - 0.5*(intensity[0]+intensity[-1]),  # guess for the amplitude
                pix[intensity.argmax()],   # guess for the position
                pix.ptp()*0.3  # guess for the hwhm
            ])
        return - (popt[0] + popt[1]) # baseline + amplitude: negative sign because the lower the better.

    def costfunc_peak_width(self, center: Tuple[float, float], dmin=0, dmax=np.inf):
        pix, intensity, area = integrate2.fastradavg(self.matrix, self.mask, center[0], center[1], dmin, dmax,
                                                     int(min(2, dmax - dmin)))
        # now get rid of possible NaNs
        area = area[np.isfinite(pix)]
        intensity = intensity[np.isfinite(pix)]
        pix = pix[np.isfinite(pix)]
        popt, pcov = scipy.optimize.curve_fit(
            self._lorentzian, pix, intensity, [
                0.5*(intensity[0]+intensity[-1]),   # first estimate for the baseline
                intensity.max() - 0.5*(intensity[0]+intensity[-1]),  # guess for the amplitude
                pix[intensity.argmax()],   # guess for the position
                pix.ptp()*0.3  # guess for the hwhm
            ])
        return popt[3]  # hwhm: the smaller the better

    def costfunc_powerlaw(self, dmin=0, dmax=np.inf):
        pix, intensity, area = integrate2.fastradavg(self.matrix, self.mask, center[0], center[1], dmin, dmax,
                                                     int(min(2, dmax - dmin)))
        # now get rid of possible NaNs
        area = area[np.isfinite(pix)]
        intensity = intensity[np.isfinite(pix)]
        pix = pix[np.isfinite(pix)]
        popt, pcov = scipy.optimize.curve_fit(
            self._powerlaw, pix, intensity, [
                0,
                1,
                -4
            ])
        return ((intensity-self._powerlaw(pix, *popt))**2).sum()/(len(intensity)-3)

    def _lorentzian(self, x, baseline, amplitude, position, hwhm):
        return baseline + amplitude/(1+((x-position)/hwhm)**2)

    def _powerlaw(self, x, baseline, amplitude, exponent):
        return baseline + amplitude*x**exponent