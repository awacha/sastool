"""Non-linear least squares fitting for Python

Created by: Andras Wacha 19.03.2012

This is just a wrapper around scipy.optimize.leastsq(). Its main goal is to make
extracting "traditional" statistical results (i.e. estimated error of the best
fit parameters, R^2, reduced Chi^2 etc) easier. It is very much like
scipy.optimize.curve_fit, except it allows for obtaining other parameters like
Chi^2, covariance matrix, R^2 etc.
"""

from scipy.optimize import leastsq
import numpy as np

def nlsq_fit(x, y, dy, func, params_init, **kwargs):
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
        statdict: dictionary of various statistical parameters:
            'DoF': Degrees of freedom
            'Chi2': Chi-squared
            'Chi2_reduced': Reduced Chi-squared
            'R2': Coefficient of determination
            'num_func_eval': number of function evaluations during fit.
            'func_value': the function evaluated in the best fitting parameters
            'message': status message from leastsq()
            'error_flag': integer status flag from leastsq() ('ier')
            'Covariance': covariance matrix (variances in the diagonal)
            'Correlation_coeffs': Pearson's correlation coefficients (usually
                denoted by 'r') in a matrix. The diagonal is unity. 
    
    Notes:
        for the actual fitting, scipy.optimize.leastsq() is used.
    """
    def objectivefunc(params, x, y, dy):
        """The target function for leastsq()."""
        return (func(x, *(params.tolist())) - y) / dy
    #do the fitting
    par, cov, infodict, mesg, ier=leastsq(objectivefunc, np.array(params_init), 
                                         (x, y, dy), full_output = True, 
                                         **kwargs)
    #test if the covariance was singular (cov is None)
    if cov is None:
        cov = np.ones((len(par), len(par))) * np.nan; #set it to a NaN matrix
    #calculate the Pearson's R^2 parameter (coefficient of determination)
    sserr = np.sum(((func(x, *(par.tolist())) - y) / dy) ** 2)
    sstot = np.sum((y - np.mean(y)) ** 2 / dy ** 2)
    r2 = 1 - sserr / sstot
    #assemble the statistics dictionary
    statdict = {'DoF' : len(x) - len(par) - 1, #degrees of freedom
                'Chi2' : (infodict['fvec'] ** 2).sum(),
                'R2' : r2,
                'num_func_eval' : infodict['nfev'],
                'func_value' : func(x, *(par.tolist())),
                'message' : mesg,
                'error_flag' : ier,
               }
    statdict['Chi2_reduced'] = statdict['Chi2'] / statdict['DoF']
    statdict['Covariance'] = cov * statdict['Chi2_reduced']
    #calculate the estimated errors of the fit parameters
    dpar = np.sqrt(statdict['Covariance'].diagonal())
    #Pearson's correlation coefficients (usually 'r') in a matrix.
    statdict['Correlation_coeffs'] = statdict['Covariance'] / np.outer(dpar,dpar)
    return par, dpar, statdict

