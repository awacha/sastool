from typing import Optional, Tuple

import numpy as np

from . import c_radavg2


def radavg(data: np.ndarray, dataerr: Optional[np.ndarray], mask: Optional[np.ndarray],
           wavelength: float, wavelength_err: float,
           distance: float, distance_err: float,
           pixelsize: float, pixelsize_err: float,
           center_row: float, center_row_err: float,
           center_col: float, center_col_err: float,
           qbins: Optional[np.ndarray],
           errorprop: int = 3, qerrorprop: int = 3,
           sector_direction: Optional[float] = None,
           sector_width: Optional[float] = None,
           sector_symmetric: bool = True):
    """
    Perform radial averaging on a scattering pattern.

    Inputs:
        data (np.ndarray, two dimensions, double dtype): scattering pattern
        error (np.ndarray, two dimensions, double dtype): uncertainties of the scattering pattern
        mask (np.ndarray, two dimensions, uint8 dtype): mask matrix
        wavelength (double): X-ray wavelength, in nm
        wavelength_unc (double): uncertainty of the X-ray wavelength, in nm
        distance (double): sample-to-detector distance, in mm
        distance_unc (double): uncertainty of the sample-to-detector distance, in mm
        pixelsize (double): the length of the edge of a square pixel, in mm
        pixelsize_unc (double): uncertainty of the pixel size, in mm
        center_row (double): beam center position, row coordinate, in pixel units
        center_row_unc (double): uncertainty of the beam center row coordinate, in pixel units
        center_col (double): beam center position, column coordinate, in pixel units
        center_col_unc (double): uncertainty of the beam center column coordinate, in pixel units
        qbincenters (np.ndarray, one dimensions, double dtype): centers of the q-bins, 1/nm
        errorprop (int, 0-3 inclusive): error propagation type for intensities (see below)
        qerrorprop (int, 0-3 inclusive): error propagation type for q (see below)
        sector_direction (double or None): direction of the sector (in degrees)
        sector_width (double or None): full angular width of the sector (in degrees)
        sector_symmetric (bool): if the opposite sector is needed, too

    Returns: q, Intensity, Error, qError, Area, pixel
        (all one-dimensional np.ndarrays, length of `qbincenters`)
        q (dtype: double): scattering variable
        Intensity (dtype: double): intensity
        Error (dtype: double): propagated uncertainty of the intensity
        qError (dtype: double): propagated uncertainty of q
        Area (dtype: uint32): number of pixels falling into the bins
        pixel (dtype: double): pixel coordinate of the bin (simple mean)

    Requirements:
        - `qbincenters` must be in ascending order and must not contain NaNs or infs.
        - `data`, `error` and `mask` must be of the same type
        - values of 0 in `mask` indicate invalid pixels. All other values correspond to valid ones.
        - beam center coordinates are expressed in pixels, starting from 0.

    Error propagation types (parameters `errorprop` and `qerrorprop`):
        0: values falling into the same bin are considered as independent samples from the same quantity.
           The bin mean is a weighted mean of the values using 1/sigma^2 as weight. Error is 1/sqrt(sum(sigma^2)).
        1: linear: error is simply the mean of errors
        2: squared: error is the square root of the mean of squared errors
        3: conservative: either `squared` or the RMS of all values, whichever is larger.

    Units:
        - wavelength and qbincenters must be compatible (nm vs 1/nm or Angström vs. 1/Angström, etc.)
        - beam centers are expected in pixel units
        - distance and pixel size must be expressed in the same units (mm, cm, etc.)

    Binning:
        - Bin centers are given in `qbincenters`. B
        - Bin edges are determined as the mean of two neighbouring bin centers.
        - The left edge of the first bin is the first element in `qbincenters`.
        - The right edge of the last bin is the  last element in `qbincenters`.

    Sector averaging:
        - if both `sector_direction` and `sector_width` is defined (not None), the averaging is limited to azimuthal
          sectors. Otherwise a full azimuthal averaging is performed.
        - `sector_direction` is the azimuth angle corresponding to the center of the sector. If the pixel
          (row=0, column=0) is top left, then the zero angle points towards right. The positive angular direction is
          counterclockwise.
        - `sector_width` is the full angular width of the sector
        - if `sector_symmetric` is True, the other sector (at angle `sector_direction` + 180°) is also used.
    """
    if qbins is None:
        qbins = c_radavg2.autoq(mask, wavelength, distance, pixelsize, center_row, center_col)

    if sector_width is not None and sector_direction is not None:
        mask = c_radavg2.maskforsectors(mask, center_row, center_col,
                                        phicenter=sector_direction / 180 * np.pi,
                                        phihalfwidth=sector_width / 180 * np.pi * 0.5,
                                        symmetric=sector_symmetric)

    q, I, E, qE, A, p = c_radavg2.radavg(data=data,
                                         dataerr=dataerr if dataerr is not None else np.ones_like(data),
                                         mask=mask,
                                         wavelength=wavelength, wavelength_unc=wavelength_err,
                                         distance=distance, distance_unc=distance_err,
                                         pixelsize=pixelsize, pixelsize_unc=pixelsize_err,
                                         center_row=center_row, center_row_unc=center_row_err,
                                         center_col=center_col, center_col_unc=center_col_err,
                                         qbincenters=qbins,
                                         errorprop=errorprop,
                                         qerrorprop=qerrorprop
                                         )

    return q, I, E, qE, A, p


def fastradavg(data: np.ndarray, mask: np.ndarray, center_row: float, center_col: float,
               dmin: float, dmax: float, N: int,
               sector_direction: Optional[float] = None, sector_width: Optional[float] = None,
               sector_symmetric: Optional[bool] = True):
    """
    Fast radial averaging

    Inputs:
        data (np.ndarray, two dimensions, dtype: double) scattering pattern
        mask (np.ndarray, two dimensions, dtype: uint8) mask matrix
        center_row (double): row coordinate of the beam center
        center_col (double): column coordinate of the beam center
        dmin (double): smallest pixel for the abscissa
        dmax (double): largest pixel for the abscissa
        N (Py_ssize_t): number of pixels
        sector_direction (double or None): direction of the sector (in degrees)
        sector_width (double or None): full angular width of the sector (in degrees)
        sector_symmetric (bool): if the opposite sector is needed, too

    Outputs: pixel, Intensity, Area
        (all one-dimensional np.ndarrays of length `N`)
        pixel (dtype: double): pixel coordinate of the bin
        Intensity (dtype: double): intensity of the bin
        Area (dtype: uint32): number of pixels in the bin

    Notes:
        - this is a reduced version of `radint`. It is significantly faster by omitting several features:
        - no error propagation is performed
        - treats only pixels (not the "q" scattering variable)
        - the abscissa range is not arbitrary: linearly spaced bins between `dmin` and `dmax`.

    Sector averaging:
        - if both `sector_direction` and `sector_width` is defined (not None), the averaging is limited to azimuthal
          sectors. Otherwise a full azimuthal averaging is performed.
        - `sector_direction` is the azimuth angle corresponding to the center of the sector. If the pixel
          (row=0, column=0) is top left, then the zero angle points towards right. The positive angular direction is
          counterclockwise.
        - `sector_width` is the full angular width of the sector
        - if `sector_symmetric` is True, the other sector (at angle `sector_direction` + 180°) is also used.

    """
    if sector_direction is not None and sector_width is not None:
        mask = c_radavg2.maskforsectors(mask, center_row, center_col,
                                        phicenter=sector_direction / 180 * np.pi,
                                        phihalfwidth=sector_width / 180 * np.pi * 0.5,
                                        symmetric=sector_symmetric)
    p, I, A = c_radavg2.fastradavg(data=data, mask=mask,
                                   center_row=center_row, center_col=center_col,
                                   dmin=dmin, dmax=dmax, N=N)
    return p, I, A


def azimavg(data: np.ndarray, dataerr: Optional[np.ndarray], mask: Optional[np.ndarray],
            wavelength: float,
            distance: float,
            pixelsize: float,
            center_row: float, center_row_err: float,
            center_col: float, center_col_err: float,
            N: int,
            errorprop: int = 3, phierrorprop: int = 3,
            interval: Optional[Tuple[float, float]] = None,
            limitsinq: bool = True):
    """
    Perform azimuthal averaging on a scattering pattern.

    Inputs:
        data (np.ndarray, two dimensions, double dtype): scattering pattern
        error (np.ndarray, two dimensions, double dtype): uncertainties of the scattering pattern
        mask (np.ndarray, two dimensions, uint8 dtype): mask matrix
        wavelength (double): X-ray wavelength, in nm
        distance (double): sample-to-detector distance, in mm
        pixelsize (double): the length of the edge of a square pixel, in mm
        center_row (double): beam center position, row coordinate, in pixel units
        center_row_unc (double): uncertainty of the beam center row coordinate, in pixel units
        center_col (double): beam center position, column coordinate, in pixel units
        center_col_unc (double): uncertainty of the beam center column coordinate, in pixel units
        N (int): number of bins
        errorprop (int, 0-3 inclusive): error propagation type for intensities (see below)
        qerrorprop (int, 0-3 inclusive): error propagation type for q (see below)
        interval (2-tuple of floats): lower and upper bounds of the annulus for limiting the averaging
        limitsinq (bool): the two numbers in `interval` are q values (True) or pixel values (False)

    Returns: phi, Intensity, Error, phiError, Area, qmean, qstd
        (all one-dimensional np.ndarrays, length of `qbincenters`)
        phi (dtype: double): azimuth angle in radians, 0 to 2*pi
        Intensity (dtype: double): intensity
        Error (dtype: double): propagated uncertainty of the intensity
        phiError (dtype: double): propagated uncertainty of phi (radians)
        Area (dtype: uint32): number of pixels falling into the bins
        qmean (dtype: double): average of q values in the bin (simple mean)
        qstd (dtype: double): sample standard deviation of the q values in the bin

    Requirements:
        - `data`, `error` and `mask` must be of the same type
        - values of 0 in `mask` indicate invalid pixels. All other values correspond to valid ones.
        - beam center coordinates are expressed in pixels, starting from 0.

    Error propagation types (parameters `errorprop` and `qerrorprop`):
        0: values falling into the same bin are considered as independent samples from the same quantity.
           The bin mean is a weighted mean of the values using 1/sigma^2 as weight. Error is 1/sqrt(sum(sigma^2)).
        1: linear: error is simply the mean of errors
        2: squared: error is the square root of the mean of squared errors
        3: conservative: either `squared` or the RMS of all values, whichever is larger.

    Units:
        - beam centers are expected in pixel units
        - distance and pixel size must be expressed in the same units (mm, cm, etc.)
        - angles are in radians

    Binning:
        - bins go from 0 (included) to 2*pi (excluded).
        - The first bin is centered on 0 rad
        - the bin width is 2*pi/N.
    """
    if interval is not None:
        if limitsinq:
            interval = [np.tan(2 * np.arcsin(q / 4 / np.pi)) * distance / pixelsize for q in interval]
        mask = c_radavg2.maskforannulus(mask, center_row, center_col, interval[0], interval[1])

    phi, intensity, error, phierror, area, qmean, qstd = c_radavg2.azimavg(
        data=data,
        error=dataerr if dataerr is not None else np.ones_like(data),
        mask=mask,
        wavelength=wavelength,
        distance=distance,
        pixelsize=pixelsize,
        center_row=center_row, center_row_unc=center_row_err,
        center_col=center_col, center_col_unc=center_col_err,
        N=N, errorprop=errorprop, phierrorprop=phierrorprop
    )
    return phi, intensity, error, phierror, area, qmean, qstd


def fastazimavg(data: np.ndarray, mask: np.ndarray, center_row: float, center_col: float, N: int,
                interval: Optional[Tuple[float, float]] = None):
    """
    Fast azimuthal averaging

    Inputs:
        data (np.ndarray, two dimensions, dtype: double) scattering pattern
        mask (np.ndarray, two dimensions, dtype: uint8) mask matrix
        center_row (double): row coordinate of the beam center
        center_col (double): column coordinate of the beam center
        N (Py_ssize_t): number of bins
        interval (2-tuple of floats): lower and upper bounds of the annulus for limiting the averaging (pixel units)

    Outputs: phi, Intensity, Area
        (all one-dimensional np.ndarrays of length `N`)
        phi (dtype: double): azimuth angle of the bin
        Intensity (dtype: double): intensity of the bin
        Area (dtype: uint32): number of pixels in the bin
    """
    if interval is not None:
        mask = c_radavg2.maskforannulus(mask, center_row, center_col, interval[0], interval[1])
    phi, intensity, area = c_radavg2.fastazimavg(data, mask, center_row, center_col, N)
    return phi, intensity, area
