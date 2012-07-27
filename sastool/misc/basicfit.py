'''
Created on Jul 25, 2012

@author: andris
'''

import numpy as np
import scipy.optimize


def _nlsq_fit(x, y, dy, func, params_init, **kwargs):
    """Perform a non-linear least squares fit
    
    Inputs:
        x: one-dimensional numpy array of the independent variable
        y: one-dimensional numpy array of the dependent variable
        dy: absolute error (square root of the variance) of the dependent
            variable
        func: a callable with the signature
            func(x,par1,par2,par3,...)
        params_init: list or tuple of the first estimates of the
            parameters par1, par2, par3 etc. to be fitted
        
        other optional keyword arguments will be passed to leastsq().
                
    Outputs: p, dp, statdict where
        p: list of fitted values of par1, par2 etc.
        dp: list of estimated errors
    
    Notes:
        for the actual fitting, scipy.optimize.leastsq() is used.
    """
    def objectivefunc(params, x, y, dy):
        """The target function for leastsq()."""
        return (func(x, *(params.tolist())) - y) / dy
    #do the fitting
    par, cov, infodict, mesg, ier = scipy.optimize.leastsq(objectivefunc, np.array(params_init),
                                         (x, y, dy), full_output = True,
                                         **kwargs)
    #test if the covariance was singular (cov is None)
    if cov is None:
        cov = np.ones((len(par), len(par))) * np.nan #set it to a NaN matrix
    #assemble the statistics dictionary
    chi2 = (infodict['fvec'] ** 2).sum()
    dpar = np.sqrt((cov * (chi2 / (len(x) - len(par) - 1))).diagonal())
    #Pearson's correlation coefficients (usually 'r') in a matrix.
    return par, dpar

def findpeak(x, y, dy = None, position = None, hwhm = None, baseline = None, amplitude = None, curve = 'Lorentz'):
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
    p, dp = _nlsq_fit(x, y, dy, fitfunc,
                                     (amplitude, position, hwhm, baseline))
    return p[1], dp[1], abs(p[2]), dp[2], p[3], dp[3], p[0], dp[0]
