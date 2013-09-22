import numpy as np
from .basic import Powerlaw
from scipy.special import sinc, sici

__all__ = ['Fsphere', 'Guinier', 'Guinier_thickness', 'Guinier_crosssection',
           'GuinierPorod', 'PorodGuinier', 'PorodGuinierPorod', 'GuinierPorodMulti', 'PorodGuinierMulti',
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
        ``G * exp(-q^2*Rg^2/3)`` if ``q<q_sep`` and ``a*q^alpha`` otherwise.
        ``q_sep`` and ``a`` are determined from conditions of smoothness at
        the cross-over.

    Literature:
    -----------
        B. Hammouda: A new Guinier-Porod model. J. Appl. Crystallogr. (2010) 43,
            716-719.
    """
    return GuinierPorodMulti(q, G, Rg, alpha)

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
        ``G * exp(-q^2*Rg^2/3)`` if ``q>q_sep`` and ``a*q^alpha`` otherwise.
        ``q_sep`` and ``G`` are determined from conditions of smoothness at
        the cross-over.

    Literature:
    -----------
        B. Hammouda: A new Guinier-Porod model. J. Appl. Crystallogr. (2010) 43,
            716-719.
    """
    return PorodGuinierMulti(q, a, alpha, Rg)

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
        ``a*q^alpha`` if ``q<q_sep1``. ``G * exp(-q^2*Rg^2/3)`` if
        ``q_sep1<q<q_sep2`` and ``b*q^beta`` if ``q_sep2<q``.
        ``q_sep1``, ``q_sep2``, ``G`` and ``b`` are determined from conditions
        of smoothness at the cross-overs.

    Literature:
    -----------
        B. Hammouda: A new Guinier-Porod model. J. Appl. Crystallogr. (2010) 43,
            716-719.
    """
    return PorodGuinierMulti(q, a, alpha, Rg, beta)

def GuinierPorodGuinier(q, G, Rg1, alpha, Rg2):
    """Empirical Guinier-Porod-Guinier scattering
    
    Inputs:
    -------
        ``q``: independent variable
        ``G``: factor for the first Guinier-branch
        ``Rg1``: the first radius of gyration
        ``alpha``: the power-law exponent
        ``Rg2``: the second radius of gyration
    
    Formula:
    --------
        ``G*exp(-q^2*Rg1^2/3)`` if ``q<q_sep1``.
        ``A*q^alpha`` if ``q_sep1 <= q  <=q_sep2``.
        ``G2*exp(-q^2*Rg2^2/3)`` if ``q_sep2<q``.
        The parameters ``A``,``G2``, ``q_sep1``, ``q_sep2`` are determined
        from conditions of smoothness at the cross-overs.
        
    Literature:
    -----------
        B. Hammouda: A new Guinier-Porod model. J. Appl. Crystallogr. (2010) 43,
            716-719.
        
    """
    return GuinierPorodMulti(q, G, Rg1, alpha, Rg2)


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

def _PG_qsep(alpha, Rg):
    return (-1.5 * alpha) ** 0.5 / np.abs(Rg)

def _PG_G(alpha, Rg, A):
    return A * np.exp(-alpha * 0.5) * (-1.5 * alpha) ** (alpha * 0.5) * Rg ** (-alpha)

def _PG_A(alpha, Rg, G):
    return G * Rg ** alpha * np.exp(alpha * 0.5) * (-1.5 * alpha) ** (-alpha * 0.5)

def _PGcs_qsep(alpha, Rg):
    return (1 - alpha) ** 0.5 / Rg

def _PGcs_G(alpha, Rg, A):
    return A * np.exp(0.5 - 0.5 * alpha) * Rg ** (1 - alpha) * (1 - alpha) ** (0.5 * alpha - 0.5)

def _PGcs_A(alpha, Rg, G):
    return G * np.exp(0.5 * alpha - 0.5) * Rg ** (alpha - 1) * (1 - alpha) ** (0.5 - 0.5 * alpha)

def _PGt_qsep(alpha, Rg):
    return ((2 - alpha) * 0.5) ** 0.5 / Rg

def _PGt_G(alpha, Rg, A):
    return A * (1 - 0.5 * alpha) ** (0.5 * alpha - 1) * np.exp(1 - 0.5 * alpha) * Rg ** (2 - alpha)

def _PGt_A(alpha, Rg, G):
    return G * (1 - 0.5 * alpha) ** (1 - 0.5 * alpha) * np.exp(0.5 * alpha - 1) * Rg ** (alpha - 2)

def GuinierPorodMulti(q, G, *Rgsalphas):
    """Empirical multi-part Guinier-Porod scattering
    
    Inputs:
    -------
        ``q``: independent variable
        ``G``: factor for the first Guinier-branch
        other arguments: [Rg1, alpha1, Rg2, alpha2, Rg3 ...] the radii of 
        gyration and power-law exponents of the consecutive parts
    
    Formula:
    --------
        The intensity is a piecewise function with continuous first derivatives.
        The separating points in ``q`` between the consecutive parts and the
        intensity factors of them (except the first) are determined from 
        conditions of smoothness (continuity of the function and its first
        derivative) at the border points of the intervals. Guinier-type
        (``G*exp(-q^2*Rg1^2/3)``) and Power-law type (``A*q^alpha``) parts
        follow each other in alternating sequence.
        
    Literature:
    -----------
        B. Hammouda: A new Guinier-Porod model. J. Appl. Crystallogr. (2010) 43,
            716-719.
    """
    scalefactor = G
    funcs = [lambda q: Guinier(q, G, Rgsalphas[0])]
    indices = np.ones_like(q, dtype=np.bool)
    constraints = []
    for i in range(1, len(Rgsalphas)):
        if i % 2:
            # Rgsalphas[i] is an exponent, Rgsalphas[i-1] is a radius of gyration
            qsep = _PG_qsep(Rgsalphas[i], Rgsalphas[i - 1])
            scalefactor = _PG_A(Rgsalphas[i], Rgsalphas[i - 1], scalefactor)
            funcs.append(lambda q, a=scalefactor, alpha=Rgsalphas[i]: Powerlaw(q, a, alpha))
        else:
            # Rgsalphas[i] is a radius of gyration, Rgsalphas[i-1] is a power-law exponent
            qsep = _PG_qsep(Rgsalphas[i - 1], Rgsalphas[i])
            scalefactor = _PG_G(Rgsalphas[i - 1], Rgsalphas[i], scalefactor)
            funcs.append(lambda q, G=scalefactor, Rg=Rgsalphas[i]: Guinier(q, G, Rg))
        # this belongs to the previous 
        constraints.append(indices & (q < qsep))
        indices[q < qsep] = False
    constraints.append(indices)
    return np.piecewise(q, constraints, funcs)

def PorodGuinierMulti(q, A, *alphasRgs):
    """Empirical multi-part Porod-Guinier scattering
    
    Inputs:
    -------
        ``q``: independent variable
        ``A``: factor for the first Power-law-branch
        other arguments: [alpha1, Rg1, alpha2, Rg2, alpha3 ...] the radii of 
        gyration and power-law exponents of the consecutive parts
    
    Formula:
    --------
        The intensity is a piecewise function with continuous first derivatives.
        The separating points in ``q`` between the consecutive parts and the
        intensity factors of them (except the first) are determined from 
        conditions of smoothness (continuity of the function and its first
        derivative) at the border points of the intervals. Guinier-type
        (``G*exp(-q^2*Rg1^2/3)``) and Power-law type (``A*q^alpha``) parts
        follow each other in alternating sequence.
        
    Literature:
    -----------
        B. Hammouda: A new Guinier-Porod model. J. Appl. Crystallogr. (2010) 43,
            716-719.
    """
    scalefactor = A
    funcs = [lambda q: Powerlaw(q, A, alphasRgs[0])]
    indices = np.ones_like(q, dtype=np.bool)
    constraints = []
    for i in range(1, len(alphasRgs)):
        if i % 2:
            # alphasRgs[i] is a radius of gyration, alphasRgs[i-1] is a power-law exponent
            qsep = _PG_qsep(alphasRgs[i - 1], alphasRgs[i])
            scalefactor = _PG_G(alphasRgs[i - 1], alphasRgs[i], scalefactor)
            funcs.append(lambda q, G=scalefactor, Rg=alphasRgs[i]: Guinier(q, G, Rg))
        else:
            # alphasRgs[i] is an exponent, alphasRgs[i-1] is a radius of gyration
            qsep = _PG_qsep(alphasRgs[i], alphasRgs[i - 1])
            scalefactor = _PG_A(alphasRgs[i], alphasRgs[i - 1], scalefactor)
            funcs.append(lambda q, a=scalefactor, alpha=alphasRgs[i]: a * q ** alpha)
        # this belongs to the previous 
        constraints.append(indices & (q < qsep))
        indices[q < qsep] = False
    constraints.append(indices)
    return np.piecewise(q, constraints, funcs)

class GeneralGuinierPorod(object):
    """Factory class for generalized piecewise Guinier-Power law functions.
    """ 
    def __init__(self, *parts):
        """Initialize the newly created object.
        
        Inputs: the type of the consecutive parts as strings. Can be:
            'Power', 'Guinier', 'Guinier_cross', 'Guinier_thick'
            
        Note that a Guinier-type part should be followed by a power-law
        and vice verse.
        """
        if 'GG' in (''.join(p[0] for p in parts)).upper():
            raise ValueError('Two Guinier curves cannot follow each other!')
        if 'PP' in (''.join(p[0] for p in parts)).upper():
            raise ValueError('Two power-law curves cannot follow each other!')
        self._parts = parts
    def __call__(self, q, factor, *Rgsalphas):
        if len(self._parts) != len(Rgsalphas):
            raise ValueError('Invalid number of arguments! Expected: %d. Got: %d.' % (len(self._parts), len(Rgsalphas)))
        for i in range(len(self._parts)):
            pass
             
