'miscellaneous utilities'
import numpy as np
import os
import sys

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

def findfileindirs(filename,dirs=[],use_pythonpath=True):
    """Find file in multiple directories."""
    dirs=normalize_listargument(dirs)
    if not dirs: #dirs is empty
        dirs=['.']
    if use_pythonpath:
        dirs.extend(sys.path)
    for d in dirs:
        if os.path.exists(os.path.join(d,filename)):
            return os.path.join(d,filename)
    raise IOError('File %s not found in any of the directories.')

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
