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
import collections
from errorvalue import ErrorValue

class FixedParameter(float):
    def __str__(self):
        return 'FixedParameter('+float.__str__(self)+')'
    def __unicode__(self):
        return 'FixedParameter('+float.__unicode__(self)+')'
    def __repr__(self):
        return 'FixedParameter('+float.__repr__(self)+')'
    pass

def hide_fixedparams(function,params):
    def newfunc(x,*pars,**kwargs):
        return function(x,*resubstitute_fixedparams(pars,params),**kwargs)
    return newfunc,[p for p in params if not isinstance(p,FixedParameter)]

def resubstitute_fixedparams(params,paramsorig):
    if isinstance(paramsorig,tuple):
        paramsorig=list(paramsorig)
    elif isinstance(paramsorig,np.ndarray):
        paramsorig=paramsorig.tolist()
    paramsorig=paramsorig[:]
    for j,p in zip([i for i in range(len(paramsorig)) if not isinstance(paramsorig[i],FixedParameter)],params):
        paramsorig[j]=p
    return paramsorig

def nonlinear_leastsquares(x, y, dy, func, params_init, **kwargs):
    """Perform a non-linear least squares fit, return the results as
    ErrorValue() instances.

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
    newfunc,newparinit=hide_fixedparams(func,params_init)
    p, dp, statdict = nlsq_fit(x, y, dy, newfunc, newparinit, **kwargs)
    p=resubstitute_fixedparams(p,params_init)
    dp=resubstitute_fixedparams(dp,[type(p_)(0) for p_ in params_init])
    def convert(p_,dp_):
        if isinstance(p_,FixedParameter) or isinstance(dp_, FixedParameter):
            return p_
        else:
            return ErrorValue(p_,dp_)
    return tuple([convert(p_, dp_) for (p_, dp_) in zip(p, dp)] + [statdict])

def simultaneous_nonlinear_leastsquares(xs, ys, dys, func, params_inits, **kwargs):
    """Do a simultaneous nonlinear least-squares fit and return the fitted
    parameters as instances of ErrorValue.

    Input:
    ------
    `xs`: tuple of abscissa vectors (1d numpy ndarrays)
    `ys`: tuple of ordinate vectors (1d numpy ndarrays)
    `dys`: tuple of the errors of ordinate vectors (1d numpy ndarrays)
    `func`: fitting function (the same for all the datasets)
    `params_init`: tuples of *lists* or *tuples* (not numpy ndarrays!) of the
        initial values of the parameters to be fitted. The special value `None`
        signifies that the corresponding parameter is the same as in the
        previous dataset. Of course, none of the parameters of the first dataset
        can be None.
    additional keyword arguments get forwarded to nlsq_fit()

    Output:
    -------
    `parset1, parset2 ...`: tuples of fitted parameters corresponding to curve1,
        curve2, etc. Each tuple contains the values of the fitted parameters
        as instances of ErrorValue, in the same order as they are in `params_init`.
    `statdict`: statistics dictionary. This is of the same form as in `nlsq_fit`,
        except that func_value is a sequence of one-dimensional np.ndarrays containing
        the best-fitting function values for each curve.
    """
    p, dp, statdict = simultaneous_nlsq_fit(xs, ys, dys, func, params_inits, **kwargs)
    params = [[ErrorValue(p_, dp_) for (p_, dp_) in zip(pcurrent, dpcurrent)] for (pcurrent, dpcurrent) in zip(p, dp)]
    return tuple(params + [statdict])

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
    func_orig=func
    params_init_orig=params_init
    func,params_init=hide_fixedparams(func_orig,params_init_orig)
    
    def objectivefunc(params, x, y, dy):
        """The target function for leastsq()."""
        return (func(x, *(params.tolist())) - y) / dy
    #do the fitting
    par, cov, infodict, mesg, ier = leastsq(objectivefunc, np.array(params_init),
                                         (x, y, dy), full_output=True,
                                         **kwargs)
    #test if the covariance was singular (cov is None)
    if cov is None:
        cov = np.ones((len(par), len(par))) * np.nan #set it to a NaN matrix
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
    statdict['Correlation_coeffs'] = statdict['Covariance'] / np.outer(dpar, dpar)
    par=resubstitute_fixedparams(par,params_init_orig)
    dpar=resubstitute_fixedparams(dpar,[type(p)(0) for p in params_init_orig])
    return par, dpar, statdict

def simultaneous_nlsq_fit(xs, ys, dys, func, params_inits, **kwargs):
    """Do a simultaneous nonlinear least-squares fit

    Input:
    ------
    `xs`: tuple of abscissa vectors (1d numpy ndarrays)
    `ys`: tuple of ordinate vectors (1d numpy ndarrays)
    `dys`: tuple of the errors of ordinate vectors (1d numpy ndarrays)
    `func`: fitting function (the same for all the datasets)
    `params_init`: tuples of *lists* or *tuples* (not numpy ndarrays!) of the
        initial values of the parameters to be fitted. The special value `None`
        signifies that the corresponding parameter is the same as in the
        previous dataset. Of course, none of the parameters of the first dataset
        can be None.
    additional keyword arguments get forwarded to nlsq_fit()

    Output:
    -------
    `p`: tuple of a list of fitted parameters
    `dp`: tuple of a list of errors of the fitted parameters
    `statdict`: statistics dictionary. This is of the same form as in `nlsq_fit`,
        except that func_value is a sequence of one-dimensional np.ndarrays containing
        the best-fitting function values for each curve.
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

    #concatenate the x, y and dy vectors
    xcat = np.concatenate(xs)
    ycat = np.concatenate(ys)
    dycat = np.concatenate(dys)
    #find the start and end indices for each dataset in the concatenated datasets.
    lens = [len(x) for x in xs]
    starts = [int(sum(lens[:i])) for i in range(len(lens))]
    ends = [int(sum(lens[:i + 1])) for i in range(len(lens))]

    #flatten the initial parameter list. A single list is needed, where the
    # constrained parameters occur only once. Of course, we have to do some
    # bookkeeping to be able to find the needed parameters for each sub-range
    # later during the fit.
    paramcat = []  # this will be the concatenated list of parameters
    param_indices = [] # this will have the same structure as params_inits (i.e.
        # a tuple of tuples of ints). Each integer number holds
        # the index of the corresponding fit parameter in the 
        # concatenated parameter list.
    for j in range(Ndata): # for each dataset
        param_indices.append([])
        jorig = j
        for i in range(Npar):
            j = jorig
            while params_inits[j][i] is None and (j >= 0):
                j = j - 1
            if j < 0:
                raise ValueError('None of the parameters in the very first dataset should be `None`.')
            if jorig == j:  #not constrained parameter
                paramcat.append(params_inits[j][i])
                param_indices[jorig].append(len(paramcat) - 1)
            else:
                param_indices[jorig].append(param_indices[j][i])

    #the flattened function
    def func_flat(x, *params):
        y = []
        for j in range(Ndata):
            pars = [params[i] for i in param_indices[j]]
            y.append(func(x[starts[j]:ends[j]], *pars))
        return np.concatenate(tuple(y))

    #Now we reduced the problem to a single least-squares fit. Carry it out and
    # interpret the results.
    pflat, dpflat, statdictflat = nlsq_fit(xcat, ycat, dycat, func_flat, paramcat, **kwargs)
    p = []
    dp = []
    fval = []
    for j in range(Ndata):
        p.append([pflat[i] for i in param_indices[j]])
        dp.append([dpflat[i] for i in param_indices[j]])
        fval.append(statdictflat['func_value'][starts[j]:ends[j]])
    statdictflat['func_value'] = fval
    return p, dp, statdictflat

