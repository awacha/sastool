'''
Created on Jul 25, 2012

@author: andris
'''
import numpy as np
from sastool.misc import easylsq
from scipy.special import gamma, psi
from sastool.classes import ErrorValue

__all__ = ['fit_shullroess']

def fit_shullroess(q, Intensity, Error, R0=None, r=None):
    """Do a Shull-Roess fitting on the scattering data.

    Inputs:

    Output:

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
    r0 = ErrorValue(r0)
    n = ErrorValue(n)

    expterm = np.exp(-r ** 2 / r0.val ** 2)
    dmaxdr0 = -2 * r ** n.val * r0.val ** (-n.val - 4) * ((n.val + 1) * r0.val ** 2 - 2 * r ** 2) * expterm / gamma((n.val + 1) * 0.5)
    dmaxdn = -r ** n.val * r0.val ** (-n.val - 1) * expterm * (2 * np.log(r0.val) - 2 * np.log(r) + psi((n.val + 1) * 0.5)) / gamma((n.val + 1) * 0.5)

    maxwellian = 2 * r ** n.val * r0.val ** (-n.val - 1) * expterm / gamma((n.val + 1) * 0.5)
    dmaxwellian = (dmaxdn ** 2 * n.err ** 2 + dmaxdr0 ** 2 * r0.err ** 2) ** 0.5
    return ErrorValue(maxwellian, dmaxwellian)
