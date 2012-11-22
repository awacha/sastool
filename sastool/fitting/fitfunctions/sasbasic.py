import numpy as np
from .basic import Powerlaw
from scipy.special import sinc, sici

__all__ = ['Fsphere', 'Guinier', 'Guinier_thickness', 'Guinier_crosssection',
           'GuinierPorod', 'PorodGuinier', 'PorodGuinierPorod',
           'DampedPowerlaw', 'LogNormSpheres', 'PowerlawPlusConstant', 'PowerlawGuinierPorodConst']

# Helper functions
def Fsphere(q, R):
    """Scattering form-factor amplitude of a sphere normalized to F(q=0)=V

    Inputs:
    -------
        ``q``: independent variable
        ``R``: sphere radius

    Formula:
    --------
        ``4*pi/q^3 * (sin(qR) - qR*cos(qR))``
    """
    return 4 * np.pi / q ** 3 * (np.sin(q * R) - q * R * np.cos(q * R))

def Guinier(q, G, Rg):
    """Guinier scattering

    Inputs:
    -------
        ``q``: independent variable
        ``G``: factor
        ``Rg``: radius of gyration

    Formula:
    --------
        ``G*exp(-(q^2*Rg^2)/3)``
    """
    return G * np.exp(-(q * Rg) ** 2 / 3.0)

def Guinier_thickness(q, G, Rg):
    """Guinier scattering of a thin lamella

    Inputs:
    -------
        ``q``: independent variable
        ``G``: factor
        ``Rg``: radius of gyration of the thickness

    Formula:
    --------
        ``G/q^2 * exp(-q^2*Rg^2)``
    """
    return G / q ** 2 * np.exp(-q ** 2 * Rg ** 2)

def Guinier_crosssection(q, G, Rg):
    """Guinier scattering of a long rod

    Inputs:
    -------
        ``q``: independent variable
        ``G``: factor
        ``Rg``: radius of gyration of the cross-section

    Formula:
    --------
        ``G/q * exp(-q^2*Rg^2/2)``
    """
    return G / q * np.exp(-q ** 2 * Rg ** 2 / 2.0)

def GuinierPorod(q, G, Rg, alpha):
    """Empirical Guinier-Porod scattering

    Inputs:
    -------
        ``q``: independent variable
        ``G``: factor of the Guinier-branch
        ``Rg``: radius of gyration
        ``alpha``: power-law exponent

    Formula:
    --------
        ``G * exp(-q^2*Rg^2/2)`` if ``q<q_sep`` and ``a*q^alpha`` otherwise.
        ``q_sep`` and ``a`` are determined from conditions of smoothness at
        the cross-over.

    Literature:
    -----------
        B. Hammouda: A new Guinier-Porod model. J. Appl. Crystallogr. (2010) 43,
            716-719.
    """
    qsep = (3 * (-alpha) * 0.5) ** 0.5 / Rg
    a = G * np.exp(alpha * 0.5) * qsep ** (-alpha)
    return np.piecewise(q, (q < qsep, q >= qsep),
                        (lambda x:Guinier(x, G, Rg), lambda x:Powerlaw(x, a, alpha))
                        )

def PorodGuinier(q, a, alpha, Rg):
    """Empirical Porod-Guinier scattering

    Inputs:
    -------
        ``q``: independent variable
        ``a``: factor of the power-law branch
        ``alpha``: power-law exponent
        ``Rg``: radius of gyration

    Formula:
    --------
        ``G * exp(-q^2*Rg^2/2)`` if ``q>q_sep`` and ``a*q^alpha`` otherwise.
        ``q_sep`` and ``G`` are determined from conditions of smoothness at
        the cross-over.

    Literature:
    -----------
        B. Hammouda: A new Guinier-Porod model. J. Appl. Crystallogr. (2010) 43,
            716-719.
    """
    qsep = (3 * (-alpha) * 0.5) ** 0.5 / Rg
    G = a * np.exp(-alpha * 0.5) * qsep ** alpha
    return np.piecewise(q, (q > qsep, q <= qsep),
                        (lambda x:Guinier(x, G, Rg), lambda x:Powerlaw(x, a, alpha))
                        )

def PorodGuinierPorod(q, a, alpha, Rg, beta):
    """Empirical Porod-Guinier-Porod scattering

    Inputs:
    -------
        ``q``: independent variable
        ``a``: factor of the first power-law branch
        ``alpha``: exponent of the first power-law branch
        ``Rg``: radius of gyration
        ``beta``: exponent of the second power-law branch

    Formula:
    --------
        ``a*q^alpha`` if ``q<q_sep1``. ``G * exp(-q^2*Rg^2/2)`` if
        ``q_sep1<q<q_sep2`` and ``b*q^beta`` if ``q_sep2<q``.
        ``q_sep1``, ``q_sep2``, ``G`` and ``b`` are determined from conditions
        of smoothness at the cross-overs.

    Literature:
    -----------
        B. Hammouda: A new Guinier-Porod model. J. Appl. Crystallogr. (2010) 43,
            716-719.
    """
    qsep = (3 * (-alpha) * 0.5) ** 0.5 / Rg
    G = a * np.exp(-alpha * 0.5) * qsep ** alpha
    return np.piecewise(q, (q < qsep, q >= qsep),
                        (lambda x:Powerlaw(x, a, alpha), lambda x:GuinierPorod(x, G, Rg, beta))
                        )

def DampedPowerlaw(q, a, alpha, sigma):
    """Damped power-law

    Inputs:
    -------
        ``q``: independent variable
        ``a``: factor
        ``alpha``: exponent
        ``sigma``: hwhm of the damping Gaussian

    Formula:
    --------
        ``a*q^alpha*exp(-q^2/(2*sigma^2))``
    """
    return a * q ** alpha * np.exp(-q ** 2 / (2 * sigma ** 2))

def LogNormSpheres(q, A, mu, sigma, N=1000):
    """Scattering of a population of non-correlated spheres (radii from a log-normal distribution)

    Inputs:
    -------
        ``q``: independent variable
        ``A``: scaling factor
        ``mu``: expectation of ``ln(R)``
        ``sigma``: hwhm of ``ln(R)``

    Non-fittable inputs:
    --------------------
        ``N``: the (integer) number of spheres

    Formula:
    --------
        The integral of ``F_sphere^2(q,R) * P(R)`` where ``P(R)`` is a
        log-normal distribution of the radii.

    """
    Rmin = 0
    Rmax = np.exp(mu + 3 * sigma)
    R = np.linspace(Rmin, Rmax, N + 1)[1:]
    P = 1 / np.sqrt(2 * np.pi * sigma ** 2 * R ** 2) * np.exp(-(np.log(R) - mu) ** 2 / (2 * sigma ** 2))
    def Fsphere_outer(q, R):
        qR = np.outer(q, R)
        q1 = np.outer(q, np.ones_like(R))
        return 4 * np.pi / q1 ** 3 * (np.sin(qR) - qR * np.cos(qR))
    I = (Fsphere_outer(q, R) ** 2 * np.outer(np.ones_like(q), P))
    return A * I.sum(1) / P.sum()

def PowerlawGuinierPorodConst(q, A, alpha, G, Rg, beta, C):
    """Sum of a Power-law, a Guinier-Porod curve and a constant.
    
    Inputs:
    -------
        ``q``: independent variable (momentum transfer)
        ``A``: scaling factor of the power-law
        ``alpha``: power-law exponent
        ``G``: scaling factor of the Guinier-Porod curve
        ``Rg``: Radius of gyration
        ``beta``: power-law exponent of the Guinier-Porod curve
        ``C``: additive constant
    
    Formula:
    --------
        ``A*q^alpha + GuinierPorod(q,G,Rg,beta) + C``
    """
    return PowerlawPlusConstant(q, A, alpha, C) + GuinierPorod(q, G, Rg, beta)

def PowerlawPlusConstant(q, A, alpha, C):
    """A power-law curve plus a constant
    
    Inputs:
    -------
        ``q``: independent variable (momentum transfer)
        ``A``: scaling factor
        ``alpha``: exponent
        ``C``: additive constant
    
    Formula:
    --------
        ``A*q^alpha + C``
    """
    return A * q ** alpha + C
