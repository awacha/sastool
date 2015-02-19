import numpy as np
from .basic import Powerlaw
from scipy.special import sinc, sici

__all__ = ['Fsphere', 'Guinier', 'Guinier_thickness', 'Guinier_crosssection',
           'GuinierPorod', 'PorodGuinier', 'PorodGuinierPorod', 'GuinierPorodMulti', 'PorodGuinierMulti',
           'DampedPowerlaw', 'LogNormSpheres', 'GaussSpheres', 'PowerlawPlusConstant', 'PowerlawGuinierPorodConst', 'GeneralGuinier', 'GeneralGuinierPorod']

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
    return GeneralGuinier(q, G, Rg, 3)

def GeneralGuinier(q, G, Rg, s):
    """Generalized Guinier scattering

    Inputs:
    -------
        ``q``: independent variable
        ``G``: factor
        ``Rg``: radius of gyration
        ``s``: dimensionality parameter (can be 1, 2, 3)

    Formula:
    --------
        ``G/q**(3-s)*exp(-(q^2*Rg^2)/s)``
    """
    return G / q ** (3 - s) * np.exp(-(q * Rg) ** 2 / s)


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
    return GeneralGuinier(q, G, Rg, 1)

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
    return GeneralGuinier(q, G, Rg, 2)

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

def GaussSpheres(q, A, R0, sigma, N=1000, weighting='intensity'):
    """Scattering of a population of non-correlated spheres (radii from a gaussian distribution)

    Inputs:
    -------
        ``q``: independent variable
        ``A``: scaling factor
        ``R0``: expectation of ``R``
        ``sigma``: hwhm of ``R``
        ``weighting``: 'intensity' (default), 'volume' or 'number'

    Non-fittable inputs:
    --------------------
        ``N``: the (integer) number of spheres

    Formula:
    --------
        The integral of ``F_sphere^2(q,R) * P(R)`` where ``P(R)`` is a
        gaussian (normal) distribution of the radii.

    """
    Rmin = max(0, R0 - 3 * sigma)
    Rmax = R0 + 3 * sigma
    R = np.linspace(Rmin, Rmax, N + 1)[1:]
    P = 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(R - R0) ** 2 / (2 * sigma ** 2))
    def Fsphere_outer(q, R):
        qR = np.outer(q, R)
        return 3 / qR ** 3 * (np.sin(qR) - qR * np.cos(qR))
    V=R**3*4*np.pi/3.
    if weighting=='intensity':
        P=P*V*V
    elif weighting=='volume':
        P=P*V
    elif weighting=='number':
        pass
    else:
        raise ValueError('Invalid weighting: '+str(weighting))    
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

def _PGgen_qsep(alpha, Rg, s):
    return ((-3 * s + s ** 2 - alpha * s) * 0.5) ** 0.5 / Rg

def _PGgen_GtoAfac(alpha,Rg, s):
    return _PGgen_qsep(alpha,Rg,s)**(s-3-alpha)*np.exp(-(s-3-alpha)*0.5)

def _PGgen_A(alpha, Rg, s, G):
    return G * _PGgen_GtoAfac(alpha,Rg,s)
    return G * Rg ** (alpha + s - 3) * (0.5 * (3 * s - s ** 2 - alpha * s)) ** (0.5 * (3 - alpha - s)) * np.exp(0.5 * (alpha + s - 3))

def _PGgen_G(alpha, Rg, s, A):
    return A/_PGgen_GtoAfac(alpha,Rg,s)
    return A * Rg ** (3 - alpha - s) * (0.5 * (3 * s - s ** 2 - alpha * s)) ** (0.5 * (alpha - 3 + s)) * np.exp(0.5 * (3 - alpha - s))

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
            qsep = _PGgen_qsep(Rgsalphas[i], Rgsalphas[i - 1], 3)
            scalefactor = _PGgen_A(Rgsalphas[i], Rgsalphas[i - 1], 3, scalefactor)
            funcs.append(lambda q, a=scalefactor, alpha=Rgsalphas[i]: Powerlaw(q, a, alpha))
        else:
            # Rgsalphas[i] is a radius of gyration, Rgsalphas[i-1] is a power-law exponent
            qsep = _PGgen_qsep(Rgsalphas[i - 1], Rgsalphas[i], 3)
            scalefactor = _PGgen_G(Rgsalphas[i - 1], Rgsalphas[i], 3, scalefactor)
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
            qsep = _PGgen_qsep(alphasRgs[i - 1], alphasRgs[i], 3)
            scalefactor = _PGgen_G(alphasRgs[i - 1], alphasRgs[i], 3, scalefactor)
            funcs.append(lambda q, G=scalefactor, Rg=alphasRgs[i]: Guinier(q, G, Rg))
        else:
            # alphasRgs[i] is an exponent, alphasRgs[i-1] is a radius of gyration
            qsep = _PGgen_qsep(alphasRgs[i], alphasRgs[i - 1], 3)
            scalefactor = _PGgen_A(alphasRgs[i], alphasRgs[i - 1], 3, scalefactor)
            funcs.append(lambda q, a=scalefactor, alpha=alphasRgs[i]: a * q ** alpha)
        # this belongs to the previous
        constraints.append(indices & (q < qsep))
        indices[q < qsep] = False
    constraints.append(indices)
    return np.piecewise(q, constraints, funcs)

def GeneralGuinierPorod(q, factor, *args, **kwargs):
    """Empirical generalized multi-part Guinier-Porod scattering

    Inputs:
    -------
        ``q``: independent variable
        ``factor``: factor for the first branch
        other arguments (*args): the defining arguments of the consecutive
             parts: radius of gyration (``Rg``) and dimensionality
             parameter (``s``) for Guinier and exponent (``alpha``) for
             power-law parts.
        supported keyword arguments:
            ``startswithguinier``: True if the first segment is a Guinier-type
            scattering (this is the default) or False if it is a power-law

    Formula:
    --------
        The intensity is a piecewise function with continuous first derivatives.
        The separating points in ``q`` between the consecutive parts and the
        intensity factors of them (except the first) are determined from
        conditions of smoothness (continuity of the function and its first
        derivative) at the border points of the intervals. Guinier-type
        (``G*q**(3-s)*exp(-q^2*Rg1^2/s)``) and Power-law type (``A*q^alpha``)
        parts follow each other in alternating sequence. The exact number of
        parts is determined from the number of positional arguments (*args).

    Literature:
    -----------
        B. Hammouda: A new Guinier-Porod model. J. Appl. Crystallogr. (2010) 43,
            716-719.
    """
    if kwargs.get('startswithguinier', True):
        funcs = [lambda q, A = factor:GeneralGuinier(q, A, args[0], args[1])]
        i = 2
        guiniernext = False
    else:
        funcs = [lambda q, A = factor: Powerlaw(q, A, args[0])]
        i = 1
        guiniernext = True
    indices = np.ones_like(q, dtype=np.bool)
    constraints = []
    while i < len(args):
        if guiniernext:
            # args[i] is a radius of gyration, args[i+1] is a dimensionality parameter, args[i-1] is a power-law exponent
            qsep = _PGgen_qsep(args[i - 1], args[i], args[i + 1])
            factor = _PGgen_G(args[i - 1], args[i], args[i + 1], factor)
            funcs.append(lambda q, G=factor, Rg=args[i], s=args[i + 1]: GeneralGuinier(q, G, Rg, s))
            guiniernext = False
            i += 2
        else:
            # args[i] is an exponent, args[i-2] is a radius of gyration, args[i-1] is a dimensionality parameter
            qsep = _PGgen_qsep(args[i], args[i - 2], args[i - 1])
            factor = _PGgen_A(args[i], args[i - 2], args[i - 1], factor)
            funcs.append(lambda q, a=factor, alpha=args[i]: a * q ** alpha)
            guiniernext = True
            i += 1
        # this belongs to the previous
        constraints.append(indices & (q < qsep))
        indices[q < qsep] = False
    constraints.append(indices)
    return np.piecewise(q, constraints, funcs)

