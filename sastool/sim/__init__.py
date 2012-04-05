"""Small-angle scattering simulation subpackage for SASTOOL"""

from _sim import *
import numpy as np

def Fsphere(q,R):
    """Scattering factor of a sphere
    
    Inputs:
        q: scalar or one-dimensional vector of q values
        R: scalar or one-dimensional vector of radii
        
    Outputs:
        The scattering factor.
    """
    qR=np.outer(q,R)
    q1=np.outer(q,np.ones_like(R))
    return 3./q1**3*(np.sin(qR)-qR*np.cos(qR))