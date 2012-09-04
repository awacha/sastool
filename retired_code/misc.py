def energycalibration(energymeas, energycalib, energy1, degree = None):
    """Do energy calibration.
    
    Inputs:
        energymeas: vector of measured (apparent) energies
        energycalib: vector of theoretical energies corresponding to the measured ones
        energy1: vector or matrix or a scalar of apparent energies to calibrate.
        degree: degree of polynomial. If None, defaults to len(energymeas)-1.
        
    Output:
        the calibrated energy/energies, in the same form as energy1 was supplied
        
    Note:
        to do backward-calibration (theoretical -> apparent), swap energymeas
        and energycalib on the parameter list.
    """
    energymeas = normalize_listargument(energymeas)
    energycalib = normalize_listargument(energycalib)
    if degree is None:
        degree = len(energymeas) - 1
    if len(energymeas) == 1: # in this case, only do a shift.
        poly = [1, energycalib[0] - energymeas[0]]
    else: # if more energy values are given, do a linear fit.
        poly = np.polyfit(energymeas, energycalib, degree)
    return np.polyval(poly, energy1)
