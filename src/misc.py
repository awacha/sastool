'miscellaneous utilities'
import numpy as np

def normalize_listargument(arg):
    """Check if arg is an iterable (list, tuple, set, dict, np.ndarray, except
        string!). If not, make a list of it. Numpy arrays are flattened and
        converted to lists."""
    if isinstance(arg,np.ndarray):
        return arg.flatten()
    if isinstance(arg,basestring):
        return [arg]
    if isinstance(arg,list) or isinstance(arg,tuple) or isinstance(arg,dict) or isinstance(arg,set):
        return arg
    return [arg]


def energycalibration(energymeas,energycalib,energy1):
    """Do energy calibration.
    
    Inputs:
        energymeas: vector of measured (apparent) energies
        energycalib: vector of theoretical energies corresponding to the measured ones
        energy1: vector or matrix or a scalar of apparent energies to calibrate.
        
    Output:
        the calibrated energy/energies, in the same form as energy1 was supplied
        
    Note:
        to do backward-calibration (theoretical -> apparent), swap energymeas
        and energycalib on the parameter list.
    """
    energymeas=normalize_listargument(energymeas)
    energycalib=normalize_listargument(energycalib)
    if len(energymeas)==1: # in this case, only do a shift.
        poly=[1,energycalib[0]-energymeas[0]]
    else: # if more energy values are given, do a linear fit.
        poly=np.lib.polynomial.polyfit(energymeas,energycalib)
    return np.lib.polynomial.polyval(poly,energy1)
