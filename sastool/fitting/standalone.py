"""Stand-alone fitting utilities"""

import numpy as np
from sastool.misc import easylsq
from scipy.special import gamma, psi
from sastool.classes import ErrorValue


__all__ = ['fit_shullroess']

def fit_shullroess(q, Intensity, Error, R0=None, r=None):
    """Do a Shull-Roess fitting on the scattering data.

    Inputs:
        q: np.ndarray[ndim=1]
            vector of the q values (4*pi*sin(theta)/lambda)
        Intensity: np.ndarray[ndim=1]
            Intensity vector
        Error: np.ndarray[ndim=1]
            Error of the intensity (absolute uncertainty, 1sigma)
        R0: scalar
            first guess for the mean radius (None to autodetermine, default)
        r: np.ndarray[ndim=1]
            vector of the abscissa of the resulting size distribution (None to
            autodetermine, default)

    Output:
        A: ErrorValue
            the fitted value of the intensity scaling factor
        r0: the r0 parameter of the maxwellian size distribution
        n: the n parameter of the maxwellian size distribution
        r: the abscissa of the fitted size distribution
        maxw: the size distribution
        stat: the statistics dictionary, returned by nlsq_fit()

    Note: This first searches for r0, which best linearizes the
            log(Intensity) vs. log(q**2+3/r0**2) relation.
            After this is found, the parameters of the fitted line give the
            parameters of a Maxwellian-like particle size distribution function.
            After it a proper least squares fitting is carried out, using the
            obtained values as initial parameters.
    """
    q = np.array(q)
    Intensity = np.array(Intensity)
    Error = np.array(Error)
    if R0 is None:
        r0s = np.linspace(1, 2 * np.pi / q.min(), 1000)
        def naive_fit_chi2(q, Intensity, r0):
            p = np.polyfit(np.log(q ** 2 + 3 / r0 ** 2), np.log(Intensity), 1)
            return ((np.polyval(p, q) - Intensity) ** 2).sum() / (len(q) - 3)
        chi2 = np.array([naive_fit_chi2(q, Intensity, r0) for r0 in r0s.tolist()])
        R0 = r0s[chi2 == chi2.min()][0]
    def naive_fit(q, Intensity, r0):
        p = np.polyfit(np.log(q ** 2 + 3 / r0 ** 2), np.log(Intensity), 1)
        return np.exp(p[1]), -2 * p[0] - 4
    K, n = naive_fit(q, Intensity, R0)
    def SR_function(q, A, r0, n):
        return A * (q ** 2 + 3 / r0 ** 2) ** (-(n + 4.) * 0.5)
    p, dp, statdict = easylsq.nlsq_fit(q, Intensity, Error, SR_function, (K, R0, n))
    n = ErrorValue(p[2], dp[2])
    r0 = ErrorValue(p[1], dp[1])
    A = ErrorValue(p[0], dp[0])
    if r is None:
        r = np.linspace(np.pi / q.max(), np.pi / q.min(), 1000)
    return A, r0, n, r, maxwellian(r, r0, n), statdict

def maxwellian(r, r0, n):
    """Maxwellian-like distribution of spherical particles
    
    Inputs:
    -------
        r: np.ndarray or scalar
            radii
        r0: positive scalar or ErrorValue
            mean radius
        n: positive scalar or ErrorValue
            "n" parameter
    
    Output:
    -------
        the distribution function and its uncertainty as an ErrorValue containing arrays.
        The uncertainty of 'r0' and 'n' is taken into account.
        
    Notes:
    ------
        M(r)=2*r^n/r0^(n+1)*exp(-r^2/r0^2) / gamma((n+1)/2)
    """
    r0 = ErrorValue(r0)
    n = ErrorValue(n)

    expterm = np.exp(-r ** 2 / r0.val ** 2)
    dmaxdr0 = -2 * r ** n.val * r0.val ** (-n.val - 4) * ((n.val + 1) * r0.val ** 2 - 2 * r ** 2) * expterm / gamma((n.val + 1) * 0.5)
    dmaxdn = -r ** n.val * r0.val ** (-n.val - 1) * expterm * (2 * np.log(r0.val) - 2 * np.log(r) + psi((n.val + 1) * 0.5)) / gamma((n.val + 1) * 0.5)

    maxwellian = 2 * r ** n.val * r0.val ** (-n.val - 1) * expterm / gamma((n.val + 1) * 0.5)
    dmaxwellian = (dmaxdn ** 2 * n.err ** 2 + dmaxdr0 ** 2 * r0.err ** 2) ** 0.5
    return ErrorValue(maxwellian, dmaxwellian)
