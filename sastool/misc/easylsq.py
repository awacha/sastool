"""Non-linear least squares fitting for Python

Created by: Andras Wacha 19.03.2012

This is just a wrapper around scipy.optimize.leastsq(). Its main goal is to make
extracting "traditional" statistical results (i.e. estimated error of the best
fit parameters, R^2, reduced Chi^2 etc) easier. It is very much like
scipy.optimize.curve_fit, except it allows for obtaining other parameters like
Chi^2, covariance matrix, R^2 etc.
"""

import collections
import time
from typing import Callable, Sequence

import numpy as np
import scipy.odr as odr
from scipy.optimize import leastsq

from .errorvalue import ErrorValue

__all__ = ['FixedParameter', 'nonlinear_leastsquares',
           'simultaneous_nonlinear_leastsquares', 'nlsq_fit',
           'simultaneous_nlsq_fit','nonlinear_odr']

class FixedParameter(float):
    def __str__(self):
        return 'FixedParameter(' + float.__str__(self) + ')'
    __unicode__ = __str__
    def __repr__(self):
        return 'FixedParameter(' + float.__repr__(self) + ')'


def hide_fixedparams(function: Callable, params: Sequence):
    def newfunc(x, *pars, **kwargs):
        return function(x, *resubstitute_fixedparams(pars, params), **kwargs)
    return newfunc, [p for p in params if not isinstance(p, FixedParameter)]


def resubstitute_fixedparams(params: Sequence, paramsorig: Sequence, covariance=None):
    if isinstance(paramsorig, tuple):
        paramsorig = list(paramsorig)
    elif isinstance(paramsorig, np.ndarray):
        paramsorig = paramsorig.tolist()
    paramsorig = paramsorig[:]
    if covariance is not None:
        cov1 = np.zeros((len(paramsorig), len(paramsorig)))
    indices_nonfixed = [i for i in range(len(paramsorig)) if not isinstance(paramsorig[i], FixedParameter)]
    for i in range(len(params)):
        paramsorig[indices_nonfixed[i]] = params[i]
        if covariance is not None:
            cov1[indices_nonfixed[i], indices_nonfixed[i]] = covariance[i, i]
            for j in range(i + 1, len(params)):
                cov1[indices_nonfixed[i], indices_nonfixed[j]] = covariance[i, j]
                cov1[indices_nonfixed[j], indices_nonfixed[i]] = covariance[j, i]
    if covariance is not None:
        return paramsorig, cov1
    else:
        return paramsorig


def nonlinear_leastsquares(x: np.ndarray, y: np.ndarray, dy: np.ndarray, func: Callable, params_init: np.ndarray,
                           verbose: bool = False, **kwargs):
    """Perform a non-linear least squares fit, return the results as
    ErrorValue() instances.

    Inputs:
        x: one-dimensional numpy array of the independent variable
        y: one-dimensional numpy array of the dependent variable
        dy: absolute error (square root of the variance) of the dependent
            variable. Either a one-dimensional numpy array or None. In the array
            case, if any of its elements is NaN, the whole array is treated as
            NaN (= no weighting)
        func: a callable with the signature
            func(x,par1,par2,par3,...)
        params_init: list or tuple of the first estimates of the
            parameters par1, par2, par3 etc. to be fitted
        `verbose`: if various messages useful for debugging should be printed on
            stdout.

        other optional keyword arguments will be passed to leastsq().

    Outputs: par1, par2, par3, ... , statdict
        par1, par2, par3, ...: fitted values of par1, par2, par3 etc
            as instances of ErrorValue.
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
        for the actual fitting, nlsq_fit() is used, which in turn delegates the
            job to scipy.optimize.leastsq().
    """
    newfunc, newparinit = hide_fixedparams(func, params_init)
    p, dp, statdict = nlsq_fit(x, y, dy, newfunc, newparinit, verbose, **kwargs)
    p, statdict['Covariance'] = resubstitute_fixedparams(p, params_init, statdict['Covariance'])
    dp, statdict['Correlation_coeffs'] = resubstitute_fixedparams(dp, [type(p_)(0) for p_ in params_init], statdict['Correlation_coeffs'])
    def convert(p_, dp_):
        if isinstance(p_, FixedParameter) or isinstance(dp_, FixedParameter):
            return p_
        else:
            return ErrorValue(p_, dp_)
    return tuple([convert(p_, dp_) for (p_, dp_) in zip(p, dp)] + [statdict])

def nonlinear_odr(x, y, dx, dy, func, params_init, **kwargs):
    """Perform a non-linear orthogonal distance regression, return the results as
    ErrorValue() instances.

    Inputs:
        x: one-dimensional numpy array of the independent variable
        y: one-dimensional numpy array of the dependent variable
        dx: absolute error (square root of the variance) of the independent
            variable. Either a one-dimensional numpy array or None. If None,
            weighting is disabled. Non-finite (NaN or inf) elements signify
            that the corresponding element in x is to be treated as fixed by
            ODRPACK.
        dy: absolute error (square root of the variance) of the dependent
            variable. Either a one-dimensional numpy array or None. If None,
            weighting is disabled.
        func: a callable with the signature
            func(x,par1,par2,par3,...)
        params_init: list or tuple of the first estimates of the
            parameters par1, par2, par3 etc. to be fitted

        other optional keyword arguments will be passed to leastsq().

    Outputs: par1, par2, par3, ... , statdict
        par1, par2, par3, ...: fitted values of par1, par2, par3 etc
            as instances of ErrorValue.
        statdict: dictionary of various statistical parameters:
            'DoF': Degrees of freedom
            'Chi2': Chi-squared
            'Chi2_reduced': Reduced Chi-squared
            'num_func_eval': number of function evaluations during fit.
            'func_value': the function evaluated in the best fitting parameters
            'message': status message from leastsq()
            'error_flag': integer status flag from leastsq() ('ier')
            'Covariance': covariance matrix (variances in the diagonal)
            'Correlation_coeffs': Pearson's correlation coefficients (usually
                denoted by 'r') in a matrix. The diagonal is unity.

    Notes:
        for the actual fitting, the module scipy.odr is used.
    """
    odrmodel=odr.Model(lambda pars, x: func(x,*pars))
    if dx is not None:
        # treat non-finite values as fixed
        xfixed=np.isfinite(dx)
    else:
        xfixed=None

    odrdata=odr.RealData(x, y, sx=dx,sy=dy, fix=xfixed)
    odrodr=odr.ODR(odrdata,odrmodel,params_init,ifixb=[not(isinstance(p,FixedParameter)) for p in params_init],
                   **kwargs)
    odroutput=odrodr.run()
    statdict=odroutput.__dict__.copy()
    statdict['Covariance']=odroutput.cov_beta
    statdict['Correlation_coeffs']=odroutput.cov_beta/np.outer(odroutput.sd_beta,odroutput.sd_beta)
    statdict['DoF']=len(x)-len(odroutput.beta)
    statdict['Chi2_reduced']=statdict['res_var']
    statdict['func_value']=statdict['y']
    statdict['Chi2']=statdict['sum_square']
    def convert(p_, dp_, pi):
        if isinstance(pi, FixedParameter):
            return FixedParameter(p_)
        else:
            return ErrorValue(p_, dp_)
    return tuple([convert(p_, dp_, pi) for (p_, dp_, pi) in zip(odroutput.beta, odroutput.sd_beta, params_init)] + [statdict])


def simultaneous_nonlinear_leastsquares(xs, ys, dys, func, params_inits, verbose=False, **kwargs):
    """Do a simultaneous nonlinear least-squares fit and return the fitted
    parameters as instances of ErrorValue.

    Input:
    ------
    `xs`: tuple of abscissa vectors (1d numpy ndarrays)
    `ys`: tuple of ordinate vectors (1d numpy ndarrays)
    `dys`: tuple of the errors of ordinate vectors (1d numpy ndarrays or Nones)
    `func`: fitting function (the same for all the datasets)
    `params_init`: tuples of *lists* or *tuples* (not numpy ndarrays!) of the
        initial values of the parameters to be fitted. The special value `None`
        signifies that the corresponding parameter is the same as in the
        previous dataset. Of course, none of the parameters of the first dataset
        can be None.
    `verbose`: if various messages useful for debugging should be printed on
        stdout.
    additional keyword arguments get forwarded to nlsq_fit()

    Output:
    -------
    `parset1, parset2 ...`: tuples of fitted parameters corresponding to curve1,
        curve2, etc. Each tuple contains the values of the fitted parameters
        as instances of ErrorValue, in the same order as they are in
        `params_init`.
    `statdict`: statistics dictionary. This is of the same form as in
        `nlsq_fit`, except that func_value is a sequence of one-dimensional
        np.ndarrays containing the best-fitting function values for each curve.
    """
    p, dp, statdict = simultaneous_nlsq_fit(xs, ys, dys, func, params_inits,
                                            verbose, **kwargs)
    params = [[ErrorValue(p_, dp_) for (p_, dp_) in zip(pcurrent, dpcurrent)]
              for (pcurrent, dpcurrent) in zip(p, dp)]
    return tuple(params + [statdict])

def nlsq_fit(x, y, dy, func, params_init, verbose=False, **kwargs):
    """Perform a non-linear least squares fit

    Inputs:
        x: one-dimensional numpy array of the independent variable
        y: one-dimensional numpy array of the dependent variable
        dy: absolute error (square root of the variance) of the dependent
            variable. Either a one-dimensional numpy array or None. In the array
            case, if any of its elements is NaN, the whole array is treated as
            NaN (= no weighting)
        func: a callable with the signature
            func(x,par1,par2,par3,...)
        params_init: list or tuple of the first estimates of the
            parameters par1, par2, par3 etc. to be fitted
        `verbose`: if various messages useful for debugging should be printed on
            stdout.

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
    if verbose:
        t0 = time.time()
        print("nlsq_fit starting.")
    func_orig = func
    params_init_orig = params_init
    func, params_init = hide_fixedparams(func_orig, params_init_orig)
    if (dy is None) or (dy == np.nan).sum() > 0 or (dy <= 0).sum() > 0:
        if verbose:
            print("nlsq_fit: no weighting")
        dy = None

    def objectivefunc(params, x, y, dy):
        """The target function for leastsq()."""
        if dy is None:
            return (func(x, *(params.tolist())) - y)
        else:
            return (func(x, *(params.tolist())) - y) / dy
    # do the fitting
    if verbose:
        print("nlsq_fit: now doing the fitting...")
        t1 = time.time()
    par, cov, infodict, mesg, ier = leastsq(objectivefunc,
                                            np.array(params_init),
                                            (x, y, dy), full_output=True,
                                            **kwargs)
    if verbose:
        print("nlsq_fit: fitting done in %.2f seconds." % (time.time() - t1))
        print("nlsq_fit: status from scipy.optimize.leastsq(): %d (%s)" % (ier, mesg))
        print("nlsq_fit: extracting statistics.")
    # test if the covariance was singular (cov is None)
    if cov is None:
        cov = np.ones((len(par), len(par))) * np.nan  # set it to a NaN matrix
    # calculate the Pearson's R^2 parameter (coefficient of determination)
    if dy is None:
        sserr = np.sum(((func(x, *(par.tolist())) - y)) ** 2)
        sstot = np.sum((y - np.mean(y)) ** 2)
    else:
        sserr = np.sum(((func(x, *(par.tolist())) - y) / dy) ** 2)
        sstot = np.sum((y - np.mean(y)) ** 2 / dy ** 2)
    r2 = 1 - sserr / sstot
    # assemble the statistics dictionary
    statdict = {'DoF' : len(x) - len(par),  # degrees of freedom
                'Chi2' : (infodict['fvec'] ** 2).sum(),
                'R2' : r2,
                'num_func_eval' : infodict['nfev'],
                'func_value' : func(x, *(par.tolist())),
                'message' : mesg,
                'error_flag' : ier,
               }
    statdict['Chi2_reduced'] = statdict['Chi2'] / statdict['DoF']
    statdict['Covariance'] = cov * statdict['Chi2_reduced']
    par, statdict['Covariance'] = resubstitute_fixedparams(par, params_init_orig, statdict['Covariance'])
    # calculate the estimated errors of the fit parameters
    dpar = np.sqrt(statdict['Covariance'].diagonal())
    # Pearson's correlation coefficients (usually 'r') in a matrix.
    statdict['Correlation_coeffs'] = statdict['Covariance'] / np.outer(dpar,
                                                                       dpar)
    if verbose:
        print("nlsq_fit: returning with results.")
        print("nlsq_fit: total time: %.2f sec." % (time.time() - t0))
    return par, dpar, statdict

def slice_covarmatrix(cov, indices):
    cov1 = np.zeros((len(indices), len(indices)), np.double)
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            cov1[i, j] = cov[indices[i], indices[j]]
            cov1[j, i] = cov[indices[j], indices[i]]
    return cov1

def simultaneous_nlsq_fit(xs, ys, dys, func, params_inits, verbose=False,
                             **kwargs):
    """Do a simultaneous nonlinear least-squares fit

    Input:
    ------
    `xs`: tuple of abscissa vectors (1d numpy ndarrays)
    `ys`: tuple of ordinate vectors (1d numpy ndarrays)
    `dys`: tuple of the errors of ordinate vectors (1d numpy ndarrays or Nones)
    `func`: fitting function (the same for all the datasets)
    `params_init`: tuples of *lists* or *tuples* (not numpy ndarrays!) of the
        initial values of the parameters to be fitted. The special value `None`
        signifies that the corresponding parameter is the same as in the
        previous dataset. Of course, none of the parameters of the first dataset
        can be None.
    `verbose`: if various messages useful for debugging should be printed on
        stdout.

    additional keyword arguments get forwarded to nlsq_fit()

    Output:
    -------
    `p`: tuple of a list of fitted parameters
    `dp`: tuple of a list of errors of the fitted parameters
    `statdict`: statistics dictionary. This is of the same form as in
        `nlsq_fit` except that func_value is a sequence of one-dimensional
        np.ndarrays containing the best-fitting function values for each curve.
    """
    if not isinstance(xs, collections.Sequence) or \
        not isinstance(ys, collections.Sequence) or \
        not isinstance(dys, collections.Sequence) or \
        not isinstance(params_inits, collections.Sequence):
        raise ValueError('Parameters `xs`, `ys`, `dys` and `params_inits` should be tuples or lists.')
    Ndata = len(xs)
    if len(ys) != Ndata or len(dys) != Ndata or len(params_inits) != Ndata:
        raise ValueError('Parameters `xs`, `ys`, `dys` and `params_inits` should have the same length.')

    if not all([isinstance(x, collections.Sequence) for x in params_inits]):
        raise ValueError('Elements of `params_inits` should be tuples or Python lists.')
    Ns = set([len(x) for x in params_inits])
    if len(Ns) != 1:
        raise ValueError('Elements of `params_inits` should have the same length.')
    Npar = Ns.pop()
    for i in range(Ndata):
        if dys[i] is None:
            dys[i] = np.ones(len(xs[i]), np.double) * np.nan
    # concatenate the x, y and dy vectors
    xcat = np.concatenate(xs)
    ycat = np.concatenate(ys)
    dycat = np.concatenate(dys)
    # find the start and end indices for each dataset in the concatenated datasets.
    lens = [len(x) for x in xs]
    starts = [int(sum(lens[:i])) for i in range(len(lens))]
    ends = [int(sum(lens[:i + 1])) for i in range(len(lens))]

    # flatten the initial parameter list. A single list is needed, where the
    # constrained parameters occur only once. Of course, we have to do some
    # bookkeeping to be able to find the needed parameters for each sub-range
    # later during the fit.
    paramcat = []  # this will be the concatenated list of parameters
    param_indices = []  # this will have the same structure as params_inits (i.e.
        # a tuple of tuples of ints). Each tuple corresponds to a dataset.
        # Each integer number in each tuple holds
        # the index of the corresponding fit parameter in the 
        # concatenated parameter list.
    for j in range(Ndata):  # for each dataset
        param_indices.append([])
        jorig = j
        for i in range(Npar):
            j = jorig
            while params_inits[j][i] is None and (j >= 0):
                j = j - 1
            if j < 0:
                raise ValueError('None of the parameters in the very first dataset should be `None`.')
            if jorig == j:  # not constrained parameter
                paramcat.append(params_inits[j][i])
                param_indices[jorig].append(len(paramcat) - 1)
            else:
                param_indices[jorig].append(param_indices[j][i])

    if verbose:
        print("Number of datasets for simultaneous fitting:", Ndata)
        print("Total number of data points:", len(xcat))
        print("Number of parameters in each dataset:", Npar)
        print("Total number of parameters:", Ndata * Npar)
        print("Number of independent parameters:", len(paramcat))
    # the flattened function
    def func_flat(x, *params):
        y = []
        for j in range(Ndata):
            if verbose > 1:
                print("Simultaneous fitting: evaluating function for dataset #", j, "/", Ndata)
            pars = [params[i] for i in param_indices[j]]
            y.append(func(x[starts[j]:ends[j]], *pars))
        return np.concatenate(tuple(y))

    # Now we reduced the problem to a single least-squares fit. Carry it out and
    # interpret the results.
    pflat, dpflat, statdictflat = nlsq_fit(xcat, ycat, dycat, func_flat, paramcat, verbose, **kwargs)
    for n in ['func_value', 'R2', 'Chi2', 'Chi2_reduced', 'DoF', 'Covariance', 'Correlation_coeffs']:
        statdictflat[n + '_global'] = statdictflat[n]
        statdictflat[n] = []
    p = []
    dp = []
    for j in range(Ndata):  # unpack the results
        p.append([pflat[i] for i in param_indices[j]])
        dp.append([dpflat[i] for i in param_indices[j]])
        statdictflat['func_value'].append(statdictflat['func_value_global'][starts[j]:ends[j]])
        if np.isfinite(dys[j]).all():
            statdictflat['Chi2'].append((((statdictflat['func_value'][-1] - ys[j]) / dys[j]) ** 2).sum())
            sstot = np.sum((ys[j] - np.mean(ys[j])) ** 2 / dys[j] ** 2)
        else:
            statdictflat['Chi2'].append(((statdictflat['func_value'][-1] - ys[j]) ** 2).sum())
            sstot = np.sum((ys[j] - np.mean(ys[j])) ** 2)
        sserr = statdictflat['Chi2'][-1]
        statdictflat['R2'].append(1 - sserr / sstot)
        statdictflat['DoF'].append(len(xs[j] - len(p[-1])))
        statdictflat['Covariance'].append(slice_covarmatrix(statdictflat['Covariance_global'], param_indices[j]))
        statdictflat['Correlation_coeffs'].append(slice_covarmatrix(statdictflat['Correlation_coeffs_global'], param_indices[j]))
        statdictflat['Chi2_reduced'].append(statdictflat['Chi2'][-1] / statdictflat['DoF'][-1])
    return p, dp, statdictflat


