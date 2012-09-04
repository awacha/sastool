import numpy as np

__all__ = ['Linear', 'Sine', 'Cosine', 'Square', 'Cube', 'Powerlaw',
           'Exponential', 'Lorentzian', 'Gaussian']

def Linear(x, a, b):
    """First-order polynomial

    Inputs:
    -------
        ``x``: independent variable
        ``a``: slope
        ``b``: offset

    Formula:
    --------
        ``a*x+b``
    """
    return a * x + b

def Sine(x, a, omega, phi, y0):
    """Sine function

    Inputs:
    -------
        ``x``: independent variable
        ``a``: amplitude
        ``omega``: circular frequency
        ``phi``: phase
        ``y0``: offset

    Formula:
    --------
        ``a*sin(x*omega + phi)+y0``
    """
    return a * np.sin(x * omega + phi) + y0

def Cosine(x, a, omega, phi, y0):
    """Cosine function

    Inputs:
    -------
        ``x``: independent variable
        ``a``: amplitude
        ``omega``: circular frequency
        ``phi``: phase
        ``y0``: offset

    Formula:
    --------
        ``a*cos(x*omega + phi)+y0``
    """
    return a * np.cos(x * omega + phi) + y0

def Square(x, a, b, c):
    """Second order polynomial

    Inputs:
    -------
        ``x``: independent variable
        ``a``: coefficient of the second-order term
        ``b``: coefficient of the first-order term
        ``c``: additive constant

    Formula:
    --------
        ``a*x^2 + b*x + c``
    """
    return a * x ** 2 + b * x + c

def Cube(x, a, b, c, d):
    """Third order polynomial

    Inputs:
    -------
        ``x``: independent variable
        ``a``: coefficient of the third-order term
        ``b``: coefficient of the second-order term
        ``c``: coefficient of the first-order term
        ``d``: additive constant

    Formula:
    --------
        ``a*x^3 + b*x^2 + c*x + d``
    """
    return a * x ** 3 + b * x ** 2 + c * x + d

def Powerlaw(x, a, alpha):
    """Power-law function

    Inputs:
    -------
        ``x``: independen variable
        ``a``: scaling factor
        ``alpha``: exponent

    Formula:
    --------
        ``a*x^alpha``
    """
    return a * x ** alpha

def Exponential(x, a, tau, y0):
    """Exponential function

    Inputs:
    -------
        ``x``: independen variable
        ``a``: scaling factor
        ``tau``: time constant
        ``y0``: additive constant

    Formula:
    --------
        ``a*exp(x/tau)+y0``
    """
    return np.exp(x / tau) * a + y0

def Lorentzian(x, a, x0, sigma, y0):
    """Lorentzian peak

    Inputs:
    -------
        ``x``: independen variable
        ``a``: scaling factor (extremal value)
        ``x0``: center
        ``sigma``: half width at half maximum
        ``y0``: additive constant

    Formula:
    --------
        ``a/(1+((x-x0)/sigma)^2)+y0``
    """
    return a / (1 + ((x - x0) / sigma) ** 2) + y0

def Gaussian(x, a, x0, sigma, y0):
    """Gaussian peak

    Inputs:
    -------
        ``x``: independen variable
        ``a``: scaling factor (extremal value)
        ``x0``: center
        ``sigma``: half width at half maximum
        ``y0``: additive constant

    Formula:
    --------
        ``a*exp(-(x-x0)^2)/(2*sigma^2)+y0``
    """
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + y0
