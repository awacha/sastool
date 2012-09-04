'''
Created on Jul 25, 2012

@author: andris
'''

import numpy as np
from .easylsq import nlsq_fit

__all__ = ['findpeak']

def findpeak(x, y, dy=None, position=None, hwhm=None, baseline=None, amplitude=None, curve='Lorentz'):
    """Find a (positive) peak in the dataset.

    Inputs:
        x, y, dy: abscissa, ordinate and the error of the ordinate (can be None)
        position, hwhm, baseline, amplitude: first guesses for the named parameters
        curve: 'Gauss' or 'Lorentz' (default)
    Outputs:
        peak position, error of peak position, hwhm, error of hwhm, baseline,
            error of baseline, amplitude, error of amplitude.

    Notes:
        A Gauss or a Lorentz curve is fitted, depending on the value of 'curve'.
    """
    if position is None: position = x[y == y.max()]
    if hwhm is None: hwhm = 0.5 * (x.max() - x.min())
    if baseline is None: baseline = y.min()
    if amplitude is None: amplitude = y.max() - baseline
    if dy is None: dy = np.ones_like(x)
    if curve.upper() == 'GAUSS':
        def fitfunc(x_, amplitude_, position_, hwhm_, baseline_):
            return amplitude_ * np.exp(0.5 * (x_ - position_) ** 2 / hwhm_ ** 2) + baseline_
    elif curve.upper() == 'LORENTZ':
        def fitfunc(x_, amplitude_, position_, hwhm_, baseline_):
            return amplitude_ * hwhm_ ** 2 / (hwhm_ ** 2 + (position_ - x_) ** 2) + baseline_
    p, dp = nlsq_fit(x, y, dy, fitfunc,
                                     (amplitude, position, hwhm, baseline))
    return p[1], dp[1], abs(p[2]), dp[2], p[3], dp[3], p[0], dp[0]
