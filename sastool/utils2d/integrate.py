from _integrate import radint 

def radintpix(data, dataerr, bcx, bcy, mask=None, pix=None, returnavgpix, phi0,
              dphi, returnmask, symmetric_sector, doslice):
    """Radial integration (averaging) on the detector plane
    
    Inputs:
        data: scattering pattern matrix (np.ndarray, dtype: np.double)
        dataerr: error matrix (np.ndarray, dtype: np.double; or None)
        bcx, bcy: beam position, counting from 1
        mask: mask matrix (np.ndarray, dtype: np.uint8)
        pix: pixel distance values (abscissa) from origin. If None,
            auto-determine.
        returnavgpix: if the averaged pixel values should be returned
        phi0: starting angle (radian) for sector integration. If doslice is True,
            this is the angle of the slice.
        dphi: angular width (radian) of the sector or width (pixels) of the
            slice. If negative or zero, full radial average is requested.
        returnmask: if the effective mask matrix is to be returned
        symmetric_sector: the sector defined by phi0+pi is also to be used for
            integration.
        doslice: if slicing is to be done instead of sector averaging.
    
    Outputs: pix, Intensity, [Error], Area, [mask]
        Error is only returned if dataerr is not None
        mask is only returned if returnmask is True
    
    Relies heavily (completely) on radint().
    """
    return radint(data, dataerr, -1, -1, -1, bcx, bcy, mask, pix, returnavgpix,
                  phi0, dphi, returnmask, symmetric_sector, doslice, False)
