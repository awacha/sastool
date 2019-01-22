'''
Created on Jul 25, 2012

@author: andris
'''

import warnings

import numpy as np
import scipy.optimize
from scipy.linalg import svd

from .easylsq import nlsq_fit
from .errorvalue import ErrorValue

__all__ = ['findpeak', 'findpeak_single', 'findpeak_multi']

def findpeak(x, y, dy=None, position=None, hwhm=None, baseline=None, amplitude=None, curve='Lorentz'):
    """Find a (positive) peak in the dataset.
    
    This function is deprecated, please consider using findpeak_single() instead.

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
    warnings.warn('Function findpeak() is deprecated, please use findpeak_single() instead.', DeprecationWarning)
    pos, hwhm, baseline, ampl = findpeak_single(x, y, dy, position, hwhm, baseline, amplitude, curve)
    return pos.val, pos.err, hwhm.val, hwhm.err, baseline.val, baseline.err, ampl.val, ampl.err


def findpeak_single(x, y, dy=None, position=None, hwhm=None, baseline=None, amplitude=None, curve='Lorentz',
                    return_stat=False, signs=(-1, 1), return_x=None):
    """Find a (positive or negative) peak in the dataset.

    Inputs:
        x, y, dy: abscissa, ordinate and the error of the ordinate (can be None)
        position, hwhm, baseline, amplitude: first guesses for the named parameters
        curve: 'Gauss' or 'Lorentz' (default)
        return_stat: return fitting statistics from easylsq.nlsq_fit()
        signs: a tuple, can be (1,), (-1,), (1,-1). Will try these signs for the peak amplitude
        return_x: abscissa on which the fitted function form has to be evaluated

    Outputs:
        peak position, hwhm, baseline, amplitude[, stat][, peakfunction]

        where:
            peak position, hwhm, baseline, amplitude are ErrorValue instances.
            stat is the statistics dictionary, returned only if return_stat is True
            peakfunction is the fitted peak evaluated at return_x if it is not None.

    Notes:
        A Gauss or a Lorentz curve is fitted, depending on the value of 'curve'. The abscissa
        should be sorted, ascending.
    """
    y_orig=y
    if dy is None: dy = np.ones_like(x)
    if curve.upper().startswith('GAUSS'):
        def fitfunc(x_, amplitude_, position_, hwhm_, baseline_):
            return amplitude_ * np.exp(-0.5 * (x_ - position_) ** 2 / hwhm_ ** 2) + baseline_
    elif curve.upper().startswith('LORENTZ'):
        def fitfunc(x_, amplitude_, position_, hwhm_, baseline_):
            return amplitude_ * hwhm_ ** 2 / (hwhm_ ** 2 + (position_ - x_) ** 2) + baseline_
    else:
        raise ValueError('Invalid curve type: {}'.format(curve))
    results=[]
    # we try fitting a positive and a negative peak and return the better fit (where R2 is larger)
    for sign in signs:
        init_params={'position':position,'hwhm':hwhm,'baseline':baseline,'amplitude':amplitude}
        y = y_orig * sign
        if init_params['position'] is None: init_params['position'] = x[y == y.max()][0]
        if init_params['hwhm'] is None: init_params['hwhm'] = 0.5 * (x.max() - x.min())
        if init_params['baseline'] is None: init_params['baseline'] = y.min()
        if init_params['amplitude'] is None: init_params['amplitude'] = y.max() - init_params['baseline']
        results.append(nlsq_fit(x, y, dy, fitfunc, (init_params['amplitude'],
                                                   init_params['position'],
                                                   init_params['hwhm'],
                                                   init_params['baseline']))+(sign,))
    max_R2=max([r[2]['R2'] for r in results])
    p,dp,stat,sign=[r for r in results if r[2]['R2']==max_R2][0]
    retval = [ErrorValue(p[1], dp[1]), ErrorValue(abs(p[2]), dp[2]), sign * ErrorValue(p[3], dp[3]),
              sign * ErrorValue(p[0], dp[0])]
    if return_stat:
        stat['func_value'] = stat['func_value'] * sign
        retval.append(stat)
    if return_x is not None:
        retval.append(sign * fitfunc(return_x, p[0], p[1], p[2], p[3]))
    return tuple(retval)

def findpeak_multi(x, y, dy, N, Ntolerance, Nfit=None, curve='Lorentz', return_xfit=False, return_stat=False):
    """Find multiple peaks in the dataset given by vectors x and y.

    Points are searched for in the dataset where the N points before and
    after have strictly lower values than them. To get rid of false
    negatives caused by fluctuations, Ntolerance is introduced. It is the
    number of outlier points to be tolerated, i.e. points on the left-hand
    side of the peak where the growing tendency breaks or on the right-hand
    side where the diminishing tendency breaks. Increasing this number,
    however gives rise to false positives.

    Inputs:
        x, y, dy: vectors defining the data-set. dy can be None.
        N, Ntolerance: the parameters of the peak-finding routines
        Nfit: the number of points on the left and on the right of
            the peak to be used for least squares refinement of the
            peak positions.
        curve: the type of the curve to be fitted to the peaks. Can be
            'Lorentz' or 'Gauss'
        return_xfit: if the abscissa used for fitting is to be returned.
        return_stat: if the fitting statistics is to be returned for each
            peak.
            
    Outputs:
        position, hwhm, baseline, amplitude, (xfit): lists
        
    Notes:
        Peaks are identified where the curve grows N points before and 
        decreases N points after. On noisy curves Ntolerance may improve
        the results, i.e. decreases the 2*N above mentioned criteria.
    """
    if Nfit is None:
        Nfit = N
    # find points where the curve grows for N points before them and
    # decreases for N points after them. To accomplish this, we create
    # an indicator array of the sign of the first derivative.
    sgndiff = np.sign(np.diff(y))
    xdiff = x[:-1]  # associate difference values to the lower 'x' value.
    pix = np.arange(len(x) - 1)  # pixel coordinates create an indicator
    # array as the sum of sgndiff shifted left and right.  whenever an
    # element of this is 2*N, it fulfills the criteria above.
    indicator = np.zeros(len(sgndiff) - 2 * N)
    for i in range(2 * N):
        indicator += np.sign(N - i) * sgndiff[i:-2 * N + i]
    # add the last one, since the indexing is different (would be
    # [2*N:0], which is not what we want)
    indicator += -sgndiff[2 * N:]
    # find the positions (indices) of the peak. The strict criteria is
    # relaxed somewhat by using the Ntolerance value. Note the use of
    # 2*Ntolerance, since each outlier point creates two outliers in
    # sgndiff (-1 insted of +1 and vice versa).
    peakpospix = pix[N:-N][indicator >= 2 * N - 2 * Ntolerance]
    ypeak = y[peakpospix]
    # Now refine the found positions by least-squares fitting. But
    # first we have to sort out other non-peaks, i.e. found points
    # which have other found points with higher values in their [-N,N]
    # neighbourhood.
    pos = []; ampl = []; hwhm = []; baseline = []; xfit = []; stat = []
    dy1 = None
    for i in range(len(ypeak)):
        if not [j for j in list(range(i + 1, len(ypeak))) + list(range(0, i)) if abs(peakpospix[j] - peakpospix[i]) <= N and ypeak[i] < ypeak[j]]:
            # only leave maxima.
            idx = peakpospix[i]
            if dy is not None:
                dy1 = dy[(idx - Nfit):(idx + Nfit + 1)]
            xfit_ = x[(idx - Nfit):(idx + Nfit + 1)]
            pos_, hwhm_, baseline_, ampl_, stat_ = findpeak_single(xfit_, y[(idx - Nfit):(idx + Nfit + 1)], dy1, position=x[idx], return_stat=True)
            
            stat.append(stat_)
            xfit.append(xfit_)
            pos.append(pos_)
            ampl.append(ampl_)
            hwhm.append(hwhm_)
            baseline.append(baseline_)
    results = [pos, hwhm, baseline, ampl]
    if return_xfit:
        results.append(xfit)
    if return_stat:
        results.append(stat)
    return tuple(results)


def findpeak_asymmetric(x, y, dy=None, curve='Lorentz', return_x=None, init_parameters=None):
    """Find an asymmetric Lorentzian peak.

    Inputs:
        x: numpy array of the abscissa
        y: numpy array of the ordinate
        dy: numpy array of the errors in y (or None if not present)
        curve: string (case insensitive): if starts with "Lorentz",
            a Lorentzian curve will be fitted. If starts with "Gauss",
            a Gaussian will be fitted. Otherwise error.
        return_x: numpy array of the x values at which the best
            fitting peak function should be evaluated and returned
        init_parameters: either None, or a list of [amplitude, center,
            hwhm_left, hwhm_right, baseline]: initial parameters to
            start fitting from.

    Results: center, hwhm_left, hwhm_right, baseline, amplitude [, y_fitted]
        The fitted parameters are returned as floats if dy was None or
        ErrorValue instances if dy was not None.
        y_fitted is only returned if return_x was not None

    Note:
        1) The dataset must contain only the peak.
        2) A positive peak will be fitted
        3) The peak center must be in the given range
    """
    idx = np.logical_and(np.isfinite(x), np.isfinite(y))
    if dy is not None:
        idx = np.logical_and(idx, np.isfinite(dy))
    x=x[idx]
    y=y[idx]
    if dy is not None:
        dy=dy[idx]
    if curve.lower().startswith('loren'):
        lorentzian = True
    elif curve.lower().startswith('gauss'):
        lorentzian = False
    else:
        raise ValueError('Unknown peak type {}'.format(curve))

    def peakfunc(pars, x, lorentzian=True):
        x0, sigma1, sigma2, C, A = pars
        result = np.empty_like(x)
        if lorentzian:
            result[x < x0] = A * sigma1 ** 2 / (sigma1 ** 2 + (x0 - x[x < x0]) ** 2) + C
            result[x >= x0] = A * sigma2 ** 2 / (sigma2 ** 2 + (x0 - x[x >= x0]) ** 2) + C
        else:
            result[x < x0] = A * np.exp(-(x[x < x0] - x0) ** 2 / (2 * sigma1 ** 2))
            result[x >= x0] = A * np.exp(-(x[x >= x0] - x0) ** 2 / (2 * sigma1 ** 2))
        return result

    def fitfunc(pars, x, y, dy, lorentzian=True):
        yfit = peakfunc(pars, x, lorentzian)
        if dy is None:
            return yfit - y
        else:
            return (yfit - y) / dy
    if init_parameters is not None:
        pos, hwhmleft, hwhmright, baseline, amplitude = [float(x) for x in init_parameters]
    else:
        baseline = y.min()
        amplitude = y.max() - baseline
        hwhmleft = hwhmright = (x.max() - x.min()) * 0.5
        pos = x[np.argmax(y)]
    #print([pos,hwhm,hwhm,baseline,amplitude])
    result = scipy.optimize.least_squares(fitfunc, [pos, hwhmleft, hwhmright, baseline, amplitude],
                                          args=(x, y, dy, lorentzian),
                                          bounds=([x.min(), 0, 0, -np.inf, 0],
                                                  [x.max(), np.inf, np.inf, np.inf, np.inf]))
#    print(result.x[0], result.x[1], result.x[2], result.x[3], result.x[4], result.message, result.success)
    if not result.success:
        raise RuntimeError('Error while peak fitting: {}'.format(result.message))
    if dy is None:
        ret = (result.x[0], result.x[1], result.x[2], result.x[3], result.x[4])
    else:
        # noinspection PyTupleAssignmentBalance
        _, s, VT = svd(result.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(result.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s ** 2, VT)
        ret = tuple([ErrorValue(result.x[i], pcov[i, i] ** 0.5) for i in range(5)])
    if return_x is not None:
        ret = ret + (peakfunc([float(x) for x in ret], return_x, lorentzian),)
    return ret
