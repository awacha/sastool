'''
Created on Feb 14, 2012

@author: andris

Corrections for 2D small-angle scattering images
'''
import numpy as np

def twotheta(matrix, bcx, bcy, pixsizeperdist):
    """Calculate the two-theta matrix for a scattering matrix

    Inputs:
        matrix: only the shape of it is needed
        bcx, bcy: beam position (counting from 0; x is row, y is column index)
        pixsizeperdist: the pixel size divided by the sample-to-detector
            distance

    Outputs:
        the two theta matrix, same shape as 'matrix'.
    """
    col, row = np.meshgrid(list(range(matrix.shape[1])), list(range(matrix.shape[0])))
    return np.arctan(np.sqrt((row - bcx) ** 2 + (col - bcy) ** 2) * pixsizeperdist)


def solidangle(twotheta, sampletodetectordistance, pixelsize=None):
    """Solid-angle correction for two-dimensional SAS images

    Inputs:
        twotheta: matrix of two-theta values
        sampletodetectordistance: sample-to-detector distance
        pixelsize: the pixel size in mm

    The output matrix is of the same shape as twotheta. The scattering intensity
        matrix should be multiplied by it.
    """
    if pixelsize is None:
        pixelsize = 1
    return sampletodetectordistance ** 2 / np.cos(twotheta) ** 3 / pixelsize ** 2


def solidangle_errorprop(twotheta, dtwotheta, sampletodetectordistance, dsampletodetectordistance, pixelsize=None):
    """Solid-angle correction for two-dimensional SAS images with error propagation

    Inputs:
        twotheta: matrix of two-theta values
        dtwotheta: matrix of absolute error of two-theta values
        sampletodetectordistance: sample-to-detector distance
        dsampletodetectordistance: absolute error of sample-to-detector distance

    Outputs two matrices of the same shape as twotheta. The scattering intensity
        matrix should be multiplied by the first one. The second one is the propagated
        error of the first one.
    """
    SAC = solidangle(twotheta, sampletodetectordistance, pixelsize)
    if pixelsize is None:
        pixelsize = 1
    return (SAC,
            (sampletodetectordistance * (4 * dsampletodetectordistance ** 2 * np.cos(twotheta) ** 2 +
                                        9 * dtwotheta ** 2 * sampletodetectordistance ** 2 * np.sin(twotheta) ** 2) ** 0.5
             / np.cos(twotheta) ** 4) / pixelsize ** 2)

def angledependentabsorption(twotheta, transmission):
    """Correction for angle-dependent absorption of the sample

    Inputs:
        twotheta: matrix of two-theta values
        transmission: the transmission of the sample (I_after/I_before, or
            exp(-mu*d))

    The output matrix is of the same shape as twotheta. The scattering intensity
        matrix should be multiplied by it. Note, that this does not corrects for
        sample transmission by itself, as the 2*theta -> 0 limit of this matrix
        is unity. Twotheta==0 and transmission==1 cases are handled correctly
        (the limit is 1 in both cases).
    """
    cor = np.ones(twotheta.shape)
    if transmission == 1:
        return cor
    mud = -np.log(transmission);

    cor[twotheta > 0] = transmission * mud * (1 - 1 / np.cos(twotheta[twotheta > 0])) / (np.exp(-mud / np.cos(twotheta[twotheta > 0])) - np.exp(-mud))
    return cor

def _calc_angledependentabsorption_error(twotheta, dtwotheta, transmission, dtransmission):
    # calculated using sympy
    return ((transmission * np.cos(twotheta) - np.exp(np.log(transmission) / np.cos(twotheta)) *
             np.log(transmission) * np.cos(twotheta) + np.exp(np.log(transmission) / np.cos(twotheta))
             * np.log(transmission) - np.exp(np.log(transmission) / np.cos(twotheta)) * np.cos(twotheta)) ** 2
             * (transmission ** 2 * dtwotheta ** 2 * np.log(transmission) ** 2 * np.sin(twotheta) ** 2
                 + dtransmission ** 2 * np.sin(twotheta) ** 4 - 3 * dtransmission ** 2 * np.sin(twotheta) ** 2
                 - 2 * dtransmission ** 2 * np.cos(twotheta) ** 3 + 2 * dtransmission ** 2) /
            (transmission - np.exp(np.log(transmission) / np.cos(twotheta))) ** 4) ** 0.5 * \
            np.abs(np.cos(twotheta)) ** (-3.0)

try:
    import sympy
    tth, dtth, T, dT = sympy.symbols('tth dtth T dT')
    mud = -sympy.log(T)
    corr = sympy.exp(-mud) * mud * (1 - 1 / sympy.cos(tth)) / (sympy.exp(-mud / sympy.cos(tth)) - sympy.exp(-mud))
    dcorr = ((sympy.diff(corr, T) ** 2 * dT ** 2 + sympy.diff(corr, tth) ** 2 * dtth ** 2)) ** 0.5
    _calc_angledependentabsorption_error = sympy.lambdify((tth, dtth, T, dT), dcorr, "numpy")
    del sympy, tth, dtth, T, dT, mud, corr, dcorr
except ImportError:
    pass

def angledependentabsorption_errorprop(twotheta, dtwotheta, transmission, dtransmission):
    """Correction for angle-dependent absorption of the sample with error propagation

    Inputs:
        twotheta: matrix of two-theta values
        dtwotheta: matrix of absolute error of two-theta values
        transmission: the transmission of the sample (I_after/I_before, or
            exp(-mu*d))
        dtransmission: the absolute error of the transmission of the sample

    Two matrices are returned: the first one is the correction (intensity matrix
        should be multiplied by it), the second is its absolute error.
    """
    # error propagation formula calculated using sympy
    return (angledependentabsorption(twotheta, transmission),
            _calc_angledependentabsorption_error(twotheta, dtwotheta, transmission, dtransmission))

def angledependentairtransmission(twotheta, mu_air, sampletodetectordistance):
    """Correction for the angle dependent absorption of air in the scattered
    beam path.

    Inputs:
            twotheta: matrix of two-theta values
            mu_air: the linear absorption coefficient of air
            sampletodetectordistance: sample-to-detector distance

    1/mu_air and sampletodetectordistance should have the same dimension

    The scattering intensity matrix should be multiplied by the resulting
    correction matrix."""
    return np.exp(mu_air * sampletodetectordistance / np.cos(twotheta))

def angledependentairtransmission_errorprop(twotheta, dtwotheta, mu_air,
                                            dmu_air, sampletodetectordistance,
                                            dsampletodetectordistance):
    """Correction for the angle dependent absorption of air in the scattered
    beam path, with error propagation

    Inputs:
            twotheta: matrix of two-theta values
            dtwotheta: absolute error matrix of two-theta
            mu_air: the linear absorption coefficient of air
            dmu_air: error of the linear absorption coefficient of air
            sampletodetectordistance: sample-to-detector distance
            dsampletodetectordistance: error of the sample-to-detector distance

    1/mu_air and sampletodetectordistance should have the same dimension

    The scattering intensity matrix should be multiplied by the resulting
    correction matrix."""
    return (np.exp(mu_air * sampletodetectordistance / np.cos(twotheta)),
            np.sqrt(dmu_air ** 2 * sampletodetectordistance ** 2 *
                    np.exp(2 * mu_air * sampletodetectordistance / np.cos(twotheta))
                    / np.cos(twotheta) ** 2 + dsampletodetectordistance ** 2 *
                    mu_air ** 2 * np.exp(2 * mu_air * sampletodetectordistance /
                                         np.cos(twotheta)) /
                    np.cos(twotheta) ** 2 + dtwotheta ** 2 * mu_air ** 2 *
                    sampletodetectordistance ** 2 *
                    np.exp(2 * mu_air * sampletodetectordistance / np.cos(twotheta))
                     * np.sin(twotheta) ** 2 / np.cos(twotheta) ** 4)
            )
