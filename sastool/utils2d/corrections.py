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
    col, row = np.meshgrid(range(matrix.shape[1]), range(matrix.shape[0]))
    return np.arctan(np.sqrt((row - bcx) ** 2 + (col - bcy) ** 2) * pixsizeperdist)

def solidangle(twotheta, sampletodetectordistance):
    """Solid-angle correction for two-dimensional SAS images
    
    Inputs:
        twotheta: matrix of two-theta values
        sampletodetectordistance: sample-to-detector distance
    
    The output matrix is of the same shape as twotheta. The scattering intensity
        matrix should be multiplied by it.
    """
    return sampletodetectordistance ** 2 / np.cos(twotheta) ** 3

def angledependentabsorption(twotheta, transmission):
    """Correction for angle-dependent absorption of the sample
    
    Inputs:
        twotheta: matrix of two-theta values
        transmission: the transmittance of the sample (I_after/I_before, or
            exp(-mu*d))
    
    The output matrix is of the same shape as twotheta. The scattering intensity
        matrix should be multiplied by it. Note, that this does not corrects for
        sample transmission by itself, as the 2*theta -> 0 limit of this matrix
        is unity.
    """
    mud = -np.log(transmission);
    cor = np.ones(twotheta.shape)
    
    cor[twotheta > 0] = transmission * mud * (1 - 1 / np.cos(twotheta[twotheta > 0])) / (np.exp(-mud / np.cos(twotheta[twotheta > 0])) - np.exp(-mud))
    return cor
