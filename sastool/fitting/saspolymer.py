import numpy as np
from scipy.special import gamma, gammainc, sinc

__all__ = []

def DebyeChain(q, Rg):
    """Scattering form-factor intensity of a Gaussian chain (Debye)

    Inputs:
    -------
        ``q``: independent variable
        ``Rg``: radius of gyration

    Formula:
    --------
        ``2*(exp(-a)-1+a)/a^2`` where ``a=(q*Rg)^2``
    """
    a = (q * Rg) ** 2
    return 2 * (np.exp(-a) - 1 + a) / a ** 2

def ExcludedVolumeChain(q, Rg, nu):
    """Scattering intensity of a generalized excluded-volume Gaussian chain

    Inputs:
    -------
        ``q``: independent variable
        ``Rg``: radius of gyration
        ``nu``: excluded volume exponent

    Formula:
    --------
        ``(u^(1/nu)*gamma(0.5/nu)*gammainc_lower(0.5/nu,u)-
            gamma(1/nu)*gammainc_lower(1/nu,u)) / (nu*u^(1/nu))``
        where ``u = q^2*Rg^2*(2*nu+1)*(2*nu+2)/6`` is the reduced scattering
        variable, ``gamma(x)`` is the gamma function and ``gammainc_lower(x,t)``
        is the lower incomplete gamma function.

    Literature:
    -----------
        SASFit manual 6. nov. 2010. Equation (3.60b)
    """
    u = (q * Rg) ** 2 * (2 * nu + 1) * (2 * nu + 2) / 6.
    return (u ** (0.5 / nu) * gamma(0.5 / nu) * gammainc(0.5 / nu, u) -
            gamma(1. / nu) * gammainc(1. / nu, u)) / (nu * u ** (1. / nu))

def BorueErukhimovich(q, C, r0, s, t):
    """Borue-Erukhimovich model of microphase separation in polyelectrolytes

    Inputs:
    -------
        ``q``: independent variable
        ``C``: scaling factor
        ``r0``: typical el.stat. screening length
        ``s``: dimensionless charge concentration
        ``t``: dimensionless temperature

    Formula:
    --------
        ``C*(x^2+s)/((x^2+s)(x^2+t)+1)`` where ``x=q*r0``

    Literature:
    -----------
        o Borue and Erukhimovich. Macromolecules (1988) 21 (11) 3240-3249
        o Shibayama and Tanaka. J. Chem. Phys (1995) 102 (23) 9392
        o Moussaid et. al. J. Phys II (France) (1993) 3 (4) 573-594
        o Ermi and Amis. Macromolecules (1997) 30 (22) 6937-6942
    """
    x = q * r0
    return C * (x ** 2 + s) / ((x ** 2 + s) * (x ** 2 + t) + 1)

def BorueErukhimovich_Powerlaw(q, C, r0, s, t, nu):
    """Borue-Erukhimovich model ending in a power-law.

    Inputs:
    -------
        ``q``: independent variable
        ``C``: scaling factor
        ``r0``: typical el.stat. screening length
        ``s``: dimensionless charge concentration
        ``t``: dimensionless temperature
        ``nu``: excluded volume parameter

    Formula:
    --------
        ``C*(x^2+s)/((x^2+s)(x^2+t)+1)`` where ``x=q*r0`` if ``q<qsep``
        ``A*q^(-1/nu)``if ``q>qsep``
        ``A`` and ``qsep`` are determined from conditions of smoothness at the
        cross-over.
    """
    def get_xsep(alpha, s, t):
        A = alpha + 2
        B = 2 * s * alpha + t * alpha + 4 * s
        C = s * t * alpha + alpha + alpha * s ** 2 + alpha * s * t - 2 + 2 * s ** 2
        D = alpha * s ** 2 * t + alpha * s
        r = np.roots([A, B, C, D])
        #print "get_xsep: ", alpha, s, t, r
        return r[r > 0][0] ** 0.5
    get_B = lambda C, xsep, s, t, nu:C * (xsep ** 2 + s) / ((xsep ** 2 + s) * (xsep ** 2 + t) + 1) * xsep ** (1.0 / nu)
    x = q * r0
    xsep = np.real_if_close(get_xsep(-1.0 / nu, s, t))
    A = get_B(C, xsep, s, t, nu)
    return np.piecewise(q, (x < xsep, x >= xsep),
                        (lambda a:BorueErukhimovich(a, C, r0, s, t),
                         lambda a:A * (a * r0) ** (-1.0 / nu)))
