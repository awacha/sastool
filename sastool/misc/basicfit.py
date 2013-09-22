'''
Created on Jul 25, 2012

@author: andris
'''

import numpy as np
from .easylsq import nlsq_fit
from .errorvalue import ErrorValue
import warnings

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

def findpeak_single(x, y, dy=None, position=None, hwhm=None, baseline=None, amplitude=None, curve='Lorentz', return_stat=False):
    """Find a (positive or negative) peak in the dataset.

    Inputs:
        x, y, dy: abscissa, ordinate and the error of the ordinate (can be None)
        position, hwhm, baseline, amplitude: first guesses for the named parameters
        curve: 'Gauss' or 'Lorentz' (default)
        return_stat: return fitting statistics from easylsq.nlsq_fit()
        
    Outputs:
        peak position, hwhm, baseline, amplitude as ErrorValue instances and the statistics 
        dictionary if requested.

    Notes:
        A Gauss or a Lorentz curve is fitted, depending on the value of 'curve'. The abscissa
        should be sorted, ascending.
    """
    # decide if the peak is positive or negative. Positive if the left and right hand sides
    # of the interval are more below the maximum than above the minimum. Negative if they are
    # more above the minimum than below the maximum.
    if ((y.max() - y[0]) >= (y[0] - y.min())) and ((y.max() - y[-1]) >= (y[-1] - y.min())):
        sign = +1
    elif ((y.max() - y[0]) <= (y[0] - y.min())) and ((y.max() - y[-1]) <= (y[-1] - y.min())):
        sign = -1
    else:
        warnings.warn('Cannot safely determine if the fitted peak should be positive or negative. Assuming positive.')
        sign = +1
    # switch peak direction to up.
    y = y * sign
    if position is None: position = x[y == y.max()][0]
    if hwhm is None: hwhm = 0.5 * (x.max() - x.min())
    if baseline is None: baseline = y.min()
    if amplitude is None: amplitude = y.max() - baseline
    if dy is None: dy = np.ones_like(x)
    if curve.upper().startswith('GAUSS'):
        def fitfunc(x_, amplitude_, position_, hwhm_, baseline_):
            return amplitude_ * np.exp(-0.5 * (x_ - position_) ** 2 / hwhm_ ** 2) + baseline_
    elif curve.upper().startswith('LORENTZ'):
        def fitfunc(x_, amplitude_, position_, hwhm_, baseline_):
            return amplitude_ * hwhm_ ** 2 / (hwhm_ ** 2 + (position_ - x_) ** 2) + baseline_
    p, dp, stat = nlsq_fit(x, y, dy, fitfunc,
                                     (amplitude, position, hwhm, baseline))
    if return_stat:
        stat['func_value'] = stat['func_value'] * sign
        return ErrorValue(p[1], dp[1]), ErrorValue(abs(p[2]), dp[2]), sign * ErrorValue(p[3], dp[3]), sign * ErrorValue(p[0], dp[0]), stat
    else:
        return ErrorValue(p[1], dp[1]), ErrorValue(abs(p[2]), dp[2]), sign * ErrorValue(p[3], dp[3]), sign * ErrorValue(p[0], dp[0])


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
        if not [j for j in range(i + 1, len(ypeak)) + range(0, i) if abs(peakpospix[j] - peakpospix[i]) <= N and ypeak[i] < ypeak[j]]:
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
