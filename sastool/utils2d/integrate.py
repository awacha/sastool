import numpy as np

from .c_integrate import autoabscissa, autoqscale, azimint, bin2D, calculateDmatrix, groupsum, polartransform, radint, \
    radint_errorprop, radint_fullq, radint_fullq_errorprop, radint_nsector, twodimfromonedim


def radintpix(data, dataerr, bcx, bcy, mask=None, pix=None, returnavgpix=False,
              phi0=0, dphi=0, returnmask=False, symmetric_sector=False,
              doslice=False, errorpropagation=2, autoqrange_linear=True):
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
        autoqrange_linear: if the automatically determined q-range is to be
            linspace-d. Otherwise log10 spacing will be applied.

    Outputs: pix, Intensity, [Error], Area, [mask]
        Error is only returned if dataerr is not None
        mask is only returned if returnmask is True

    Relies heavily (completely) on radint().
    """
    if isinstance(data, np.ndarray):
        data = data.astype(np.double)
    if isinstance(dataerr, np.ndarray):
        dataerr = dataerr.astype(np.double)
    if isinstance(mask, np.ndarray):
        mask = mask.astype(np.uint8)
    return radint(data, dataerr, -1, -1, -1,
                  1.0 * bcx, 1.0 * bcy, mask, pix, returnavgpix,
                  phi0, dphi, returnmask, symmetric_sector, doslice, False, errorpropagation, autoqrange_linear)


def azimintpix(data, dataerr, bcx, bcy, mask=None, Ntheta=100, pixmin=0,
               pixmax=np.inf, returnmask=False, errorpropagation=2):
    """Azimuthal integration (averaging) on the detector plane

    Inputs:
        data: scattering pattern matrix (np.ndarray, dtype: np.double)
        dataerr: error matrix (np.ndarray, dtype: np.double; or None)
        bcx, bcy: beam position, counting from 1
        mask: mask matrix (np.ndarray, dtype: np.uint8)
        Ntheta: Number of points in the abscissa (azimuth angle)
        pixmin: smallest distance from the origin in pixels
        pixmax: largest distance from the origin in pixels
        returnmask: if the effective mask matrix is to be returned

    Outputs: theta, Intensity, [Error], Area, [mask]
        Error is only returned if dataerr is not None
        mask is only returned if returnmask is True

    Relies heavily (completely) on azimint().
    """
    if isinstance(data, np.ndarray):
        data = data.astype(np.double)
    if isinstance(dataerr, np.ndarray):
        dataerr = dataerr.astype(np.double)
    if isinstance(mask, np.ndarray):
        mask = mask.astype(np.uint8)
    return azimint(data, dataerr, -1, -1,
                   - 1, bcx, bcy, mask, Ntheta, pixmin,
                   pixmax, returnmask, errorpropagation)
