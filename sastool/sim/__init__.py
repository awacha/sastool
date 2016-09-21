"""Small-angle scattering simulation subpackage for SASTOOL"""

import numpy as np

from .c_sim import *

__all__ = []


def Fsphere(q, R):
    """Scattering factor of a sphere, normalized to F(q=0)=N_electrons

    Inputs:
        q: scalar or one-dimensional vector of q values
        R: scalar or one-dimensional vector of radii

    Outputs:
        the values of the form factor amplitude in a `np.ndarray`. Its shape is
        that of ``np.outer(q, R)`` and is scaled such as F(q = 0) = V
    """
    qR = np.outer(q, R)
    q1 = np.outer(q, np.ones_like(R))
    return 4 * np.pi / q1 ** 3 * (np.sin(qR) - qR * np.cos(qR))


def FGaussProfile(q, R, sigma):
    """Form factor of a radial layer with a Gaussian radial profile, normalized
    to F(q=0) = V

    Inputs:
    -------
    q: scalar or a one-dimensional np.ndarray
        the values of the scattering vector.
    R: scalar or a one-dimensional np.ndarray
        the values of the mean value of the radial Gauss profile
    sigma: scalar
        HWHM of the radial Gauss profile

    Outputs:
    --------
        the values of the form factor amplitude in a np.ndarray. Its shape is
        that of ``np.outer(q, R)`` and it is scaled such as F(q=0) = V
    """
    qR = np.outer(q, R)
    q1 = np.outer(q, np.ones_like(R))
    Rdivq = np.outer(1.0 / q, R)
    return 4 * np.pi * np.sqrt(2 * np.pi) * sigma / (Rdivq * np.sin(qR) + sigma ** 2 * np.cos(qR)) * np.exp(-q1 ** 2 * sigma ** 2 * 0.5)
