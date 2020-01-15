# cython: cdivision=True, wraparound=False, boundscheck=False, language_level=3, embedsignature=True
import numpy as np
cimport numpy as np
from libc.math cimport  sqrt, atan, sin, cos, M_PI, NAN, floor, HUGE_VAL, atan2, fabs
from libc.stdint cimport uint32_t, uint8_t
cimport numpy as np
import numpy as np
from libc.math cimport sqrt, atan, sin, cos, M_PI, NAN, floor, HUGE_VAL, atan2, fabs
from libc.stdint cimport uint32_t, uint8_t

def autoq(uint8_t[:,:] mask, double wavelength, double distance, double pixelsize, double center_row, double center_col,
          bint linspacing=True, Py_ssize_t N=-1):
    """Determine q-scale automatically

    Inputs:
        mask (np.ndarray, two dimensions, dtype: uint8): mask matrix (1 valid, 0 invalid).
        wavelength (double): wavelength in nm (or Angstroem, this determines the unit of the returned q values)
        distance (double): sample-detector distance in mm
        pixelsize (double): pixel size in mm
        center_row, center_col (double): beam position in pixel (starting from 0)
        linspacing (bool): if linear spacing is expected. Otherwise log10 spacing.
        N (Py_ssize_t): number of points. If nonpositive, auto-determined.

    Output: the q scale in a numpy vector.
    """
    cdef:
        double r2min, r2max, r2, qmin, qmax
        Py_ssize_t irow, icol

    r2min = HUGE_VAL
    r2max = 0
    for irow in range(mask.shape[0]):
        for icol in range(mask.shape[1]):
            if mask[irow, icol]==0:
                continue
            r2 = ((irow - center_row)**2 + (icol - center_col)**2)
            if r2 > r2max:
                r2max = r2
            if r2 < r2min:
                r2min = r2
    qmin = 4 * M_PI * sin(0.5 * atan(sqrt(r2min)*pixelsize/distance))/wavelength
    qmax = 4 * M_PI * sin(0.5 * atan(sqrt(r2max)*pixelsize/distance))/wavelength
    print('Minimum radius: ', sqrt(r2min))
    print('Maximum radius: ', sqrt(r2max))
    if N <= 0:
        N = <Py_ssize_t>(r2max**0.5 - r2min**0.5)+1

    if linspacing:
        return np.linspace(qmin, qmax, N)
    else:
        return np.logspace(np.log10(qmin), np.log10(qmax), N)


def radavg(double[:,:] data, double[:,:] error, uint8_t[:,:] mask,
           double wavelength, double wavelength_unc,
           double distance, double distance_unc,
           double pixelsize, double pixelsize_unc,
           double center_row, double center_row_unc,
           double center_col, double center_col_unc,
           double[:] qbincenters,
           int errorprop=3, int qerrorprop=3
          ):
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

    Returns: q, Intensity, Error, qError, Area, pixel
        (all one-dimensional np.ndarrays, length of `qbincenters`)
        q (dtype: double): scattering variable
        Intensity (dtype: double): intensity
        Error (dtype: double): propagated uncertainty of the intensity
        qError (dtype: double): propagated uncertainty of q
        Area (dtype: uint32): number of pixels falling into the bins
        pixel (dtype: double): pixel coordinate of the bin (simple mean)

    Notes:
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
    """
    cdef:
        double[:] Intensity
        double[:] Intensity2
        double[:] Error
        double[:] q
        double[:] q2
        double[:] qError
        double[:] qmax
        uint32_t[:] Area
        double[:] pixel
        Py_ssize_t irow, icolumn, ibin
        double currentq, pixelradius2, tgtwotheta2, qfac, row, col, sinth
        double currentq_unc, pixelradius_unc2, tgtwotheta_unc2, qfac_unc2, row_unc2, col_unc2, sinth_unc2
        double dist_relunc2, pixelsize_relunc2
        double error_propagated
        double error_statistics
    # initialize output arrays
    Intensity = np.zeros(len(qbincenters), dtype=np.double)
    Intensity2 = np.zeros(len(qbincenters), dtype=np.double)
    Error = np.zeros(len(qbincenters), dtype=np.double)
    q = np.zeros(len(qbincenters), dtype=np.double)
    q2 = np.zeros(len(qbincenters), dtype=np.double)
    qError = np.zeros(len(qbincenters), dtype=np.double)
    Area = np.zeros(len(qbincenters), dtype=np.uint32)
    pixel = np.zeros(len(qbincenters), dtype=np.double)
    qmax = np.zeros(len(qbincenters), dtype=np.double)
    # the prefactor of "q": speed things up.
    qfac = 4*M_PI/wavelength
    qfac_unc2 = 16*M_PI*M_PI*wavelength_unc**2/wavelength**4
    # the uncertainty of the center-corrected row and column coordinate does not change
    row_unc2 = 0.25 + center_row_unc**2  # the uncertainty of the pixel coordinate is assumed to be 0.5 pixel
    col_unc2 = 0.25 + center_col_unc**2
    # these are for speeding things up
    dist_relunc2 = distance_unc**2/distance**2
    pixelsize_relunc2 = pixelsize_unc**2/pixelsize**2
    # set upper bin limits
    for ibin in range(len(qbincenters)-1):
        qmax[ibin] = 0.5*(qbincenters[ibin] + qbincenters[ibin+1])
    qmax[len(qbincenters)-1] = qbincenters[len(qbincenters)-1]

    # categorize pixels into q bins.
    for irow in range(data.shape[0]):
        for icolumn in range(data.shape[1]):
            #print(irow, icolumn)
            if mask[irow, icolumn] == 0:
                # this pixel is masked
                continue
            row = irow-center_row
            col = icolumn-center_col
            # using sqrt(x**2+y**2) was faster than hypot(x,y)
            pixelradius2 = row**2 + col**2
            # propagated squared uncertainty of (irow-center_row) is 0.25 + dcenter_row**2
            # propagated squared uncertainty of sqrt(A**2+B**2) is (A**2*dA**2 + B**2+dB**)/(A**2+B**2).
            #
            pixelradius_unc2 = (row_unc2*row**2 + col_unc2*col**2)/pixelradius2
            tgtwotheta2 = pixelradius2*pixelsize**2/distance**2
            tgtwotheta_unc2 = tgtwotheta2 * (pixelradius_unc2/pixelradius2 +
                                               dist_relunc2 +
                                               pixelsize_relunc2
                                              )
            # by employing trigonometric identities we get an algebraic form for tg(2theta) -> sin(theta).
            # While this is faster than the obvious sin(0.5*arctan(tg2theta)), it only works for 0 < 2theta < pi/2.
            # This is usually true in SAXS.
            sinth = (0.5*(1-1/(tgtwotheta2+1)**0.5))**0.5
            sinth_unc2 = 1/16. * tgtwotheta2 / (tgtwotheta2+1)**3 / (1-1/(tgtwotheta2+1)**0.5)*tgtwotheta_unc2**2
            currentq = qfac * sinth
            currentq_unc2 = currentq**2 * (qfac_unc2 / qfac**2 + sinth_unc2 / sinth**2)
            # Now find the q-bin
            if currentq<qbincenters[0]:
                # underflow
                continue
            if currentq > qmax[len(qbincenters)-1]:
                # overflow
                continue
            for ibin in range(len(qbincenters)):
                if qmax[ibin] > currentq:
                    # this pixel belongs to the current q-bin
                    break
            # now bin the pixel
            if errorprop > 0: # 1,2,3
                Intensity[ibin] += data[irow, icolumn]
                if errorprop == 1:
                    Error[ibin] += error[irow, icolumn]
                else:
                    Error[ibin] += error[irow, icolumn]**2
                if errorprop == 3:
                    Intensity2[ibin] += data[irow, icolumn]**2
            else: # 0
                Intensity[ibin] += data[irow, icolumn] / error[irow, icolumn]**2
                Error[ibin] += 1/error[irow, icolumn]**2
            if qerrorprop > 0: # 1, 2 or 3
                q[ibin] += currentq
                if qerrorprop == 1:
                    qError[ibin] += sqrt(currentq_unc2)
                else:
                    qError[ibin] += currentq_unc2
                if qerrorprop == 3:
                    q2[ibin] += currentq**2
            else: # 0
                q[ibin] += currentq / currentq_unc2
                qError[ibin] += currentq_unc2
            Area[ibin] += 1
            pixel[ibin] += sqrt(pixelradius2)

    # normalize the bins
    for ibin in range(len(qbincenters)):
        if not Area[ibin]:
            # no pixels in this bin: set everything to NaN
            Intensity[ibin] = NAN
            Error[ibin] = NAN
            q[ibin] = NAN
            qError[ibin] = NAN
            pixel[ibin] = NAN
            continue
        if errorprop >0:
            if errorprop == 1:
                Error[ibin] /= Area[ibin]
            elif errorprop == 2:
                Error[ibin] = sqrt(Error[ibin])/Area[ibin]
            else: # errorprop == 3
                error_propagated = sqrt(Error[ibin]/Area[ibin])
                if Area[ibin] < 2:
                    error_statistics = 0
                else:
                    # sample standard deviation: sqrt((sum(I**2) - N*mean(I)**2)/(N-1))
                    # in our case, sum(I**2) is simply Intensity2[ibin].
                    # mean(I) = Intensity[ibin] / Area[ibin], thus N*mean(I)**2 is Intensity[ibin]**2/Area[ibin]
                    error_statistics = sqrt(
                        (Intensity2[ibin] - Intensity[ibin]**2/Area[ibin])/(Area[ibin]-1))
                Error[ibin] = max(error_propagated, error_statistics)
            Intensity[ibin] /= Area[ibin]
        else:
            Intensity[ibin] /= Error[ibin]
            Error[ibin] = 1/sqrt(Error[ibin])
        if qerrorprop >0:
            if qerrorprop == 1:
                qError[ibin] /= Area[ibin]
            elif qerrorprop == 2:
                qError[ibin] = sqrt(qError[ibin])/Area[ibin]
            else: # qerrorprop == 3
                error_propagated = sqrt(qError[ibin]/Area[ibin])
                if Area[ibin] < 2:
                    error_statistics = 0
                else:
                    # sample standard deviation: sqrt((sum(q**2) - N*mean(q)**2)/(N-1))
                    # in our case, sum(q**2) is simply q2[ibin].
                    # mean(q) = q[ibin] / Area[ibin], thus N*mean(q)**2 is q[ibin]**2/Area[ibin]
                    error_statistics = sqrt(
                        (q2[ibin] - q[ibin]**2/Area[ibin])/(Area[ibin]-1))
                qError[ibin] = max(error_propagated, error_statistics)
            q[ibin] /= Area[ibin]
        else:
            q[ibin] /= qError[ibin]
            qError[ibin] = 1/sqrt(qError[ibin])
        pixel[ibin] /= Area[ibin]
    return np.array(q), np.array(Intensity), np.array(Error), np.array(qError), np.array(Area), np.array(pixel)

def fastradavg(double[:,:] data, uint8_t[:,:] mask,
               double center_row, double center_col,
               double dmin, double dmax, Py_ssize_t N):
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

    Outputs: pixel, Intensity, Area
        (all one-dimensional np.ndarrays of length `N`)
        pixel (dtype: double): pixel coordinate of the bin
        Intensity (dtype: double): intensity of the bin
        Area (dtype: uint32): number of pixels in the bin
    """
    cdef:
        Py_ssize_t i, j, ibin
        double[:] pixel
        double[:] Intensity
        uint32_t[:] Area
        double r

    pixel = np.zeros(N, np.double)
    Intensity = np.zeros(N, np.double)
    Area = np.zeros(N, np.uint32)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if mask[i,j] ==0:
                continue
            r = sqrt((i-center_row)**2 + (j-center_col)**2)
            ibin = <Py_ssize_t>(floor((r-dmin)/(dmax-dmin)*N))
            if ibin>=0 and ibin < N:
                pixel[ibin] += r
                Intensity[ibin] += data[i,j]
                Area[ibin] += 1
    for ibin in range(N):
        if Area[ibin] == 0:
            pixel[ibin] = NAN
            Intensity[ibin] = NAN
        else:
            pixel[ibin] /= Area[ibin]
            Intensity[ibin] /= Area[ibin]
    return np.array(pixel), np.array(Intensity), np.array(Area)

def maskforsectors(uint8_t[:,:] mask, double center_row, double center_col, double phicenter, double phiwidth, bint symmetric=False):
    """
    Calculate a mask which can be used to limit radial averaging to a sector.

    Inputs:
        mask (np.ndarray, two dimensional, dtype: uint8): mask matrix
        center_row (double): row coordinate of the beam center
        center_col (double): column coordinate of the beam center
        phicenter (double): azimuth angle of the center of the sector expressed in radians
        phiwidth (double): full angular width of the sector (expressed in radians)
        symmetric (bool): if the opposite sector should also be considered

    Outputs:
        the sector mask: pixels which do not lie in the desired sectors are masked. All pixels which are masked in the
            input mask will also remain masked.

    Notes:
        If the (row=0, column=0) point of the mask is top left, then the direction designated by phi=0 points to the
        right. The positive direction of angles is counterclockwise.
    """
    cdef:
        Py_ssize_t irow, icol
        uint8_t[:,:] maskout
        double row, col, r
        double rowref, colref
        double sindeltaphi, cosdeltaphi

    # row and column coordinates of the direction vector of the sector center
    rowref = -sin(phicenter)
    colref = cos(phicenter)

    maskout = np.empty_like(mask)
    for irow in range(mask.shape[0]):
        row = irow-center_row
        for icol in range(mask.shape[1]):
            if mask[irow, icol] == 0:
                maskout[irow, icol] = 0
                continue
            col = icol-center_col
            r = sqrt(row**2+col**2)
            sindeltaphi = (rowref*col - row*colref) / r
            cosdeltaphi = (rowref*row + colref*col) / r
            deltaphi = atan2(sindeltaphi, cosdeltaphi)
            if fabs(deltaphi) <= 0.5*phiwidth:
                maskout[irow, icol] = 1
            elif symmetric and (fabs(deltaphi) >= (np.pi - 0.5*phiwidth)):
                maskout[irow, icol] = 1
            else:
                maskout[irow, icol] = 0
    return np.array(maskout)

def maskforannulus(uint8_t[:,:] mask, double center_row, double center_col, double pixmin, double pixmax):
    """
    Calculate a mask which can be used to limit radial averaging to a sector.

    Inputs:
        mask (np.ndarray, two dimensional, dtype: uint8): mask matrix
        center_row (double): row coordinate of the beam center
        center_col (double): column coordinate of the beam center
        pixmin (double): azimuth angle of the center of the sector expressed in radians
        pixmax (double): full angular width of the sector (expressed in radians)

    Outputs:
        the updated mask: pixels which do not lie in the desired annulus are masked. All pixels which are masked in the
            input mask will also remain masked.
    """
    cdef:
        Py_ssize_t irow, icol
        uint8_t[:,:] maskout
        double row, col, r


    maskout = np.empty_like(mask)
    for irow in range(mask.shape[0]):
        row = irow-center_row
        for icol in range(mask.shape[1]):
            if mask[irow, icol] == 0:
                maskout[irow, icol] = 0
                continue
            col = icol-center_col
            r = sqrt(row**2+col**2)
            maskout[irow, icol] = (r>=pixmin) & (r<=pixmax)
    return np.array(maskout)

def azimavg(double[:,:] data, double[:,:] error, uint8_t[:,:] mask,
            double wavelength,
            double distance,
            double pixelsize,
            double center_row, double center_row_unc,
            double center_col, double center_col_unc,
            Py_ssize_t N=100,
            int errorprop=3, int phierrorprop=3
            ):
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
        N (Py_ssize_t): number of points in the output
        errorprop (int, 0-3 inclusive): error propagation type for intensities (see below)
        phierrorprop (int, 0-3 inclusive): error propagation type for the azimuth angle (see below)

    Returns: phi, Intensity, Error, phiError, Area, qmean, qstd
        (all one-dimensional np.ndarrays, length of `qbincenters`)
        phi (dtype: double): azimuth angle, from 0 to 2*pi
        Intensity (dtype: double): intensity
        Error (dtype: double): propagated uncertainty of the intensity
        phiError (dtype: double): propagated uncertainty of the azimuth angle
        Area (dtype: uint32): number of pixels falling into the bins
        qmean (dtype: double): mean q values for each bin
        qstd (dtype: double): sample standard deviation of the q values in each bin

    Notes:
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

    Binning:
        - bins go from 0 (included) to 2*pi (excluded).
        - The first bin is centered on 0 rad
        - the bin width is 2*pi/N.
    """
    cdef:
        double[:] Intensity
        double[:] Intensity2
        double[:] Error
        double[:] phi
        double[:] phi2
        double[:] phiError
        double[:] q
        double[:] q2
        double[:] qmean
        double[:] qstd
        uint32_t[:] Area
        double[:] pixel
        Py_ssize_t irow, icolumn, ibin
        double currentphi, currentphi_unc, pixelradius, halfbinwidth

        double currentq, tgtwotheta2, qfac, row, col, sinth
        double row_unc2, col_unc2, sinth_unc2
        double error_propagated
        double error_statistics
    # initialize output arrays
    Intensity = np.zeros(N, dtype=np.double)
    Intensity2 = np.zeros(N, dtype=np.double)
    Error = np.zeros(N, dtype=np.double)
    phi = np.zeros(N, dtype=np.double)
    phi2 = np.zeros(N, dtype=np.double)
    phiError = np.zeros(N, dtype=np.double)
    q = np.zeros(N, dtype=np.double)
    q2 = np.zeros(N, dtype=np.double)
    qmean = np.zeros(N, dtype=np.double)
    qstd = np.zeros(N, dtype=np.double)
    Area = np.zeros(N, dtype=np.uint32)
    pixel = np.zeros(N, dtype=np.double)
    # the prefactor of "q": speed things up.
    qfac = 4*M_PI/wavelength
    # the uncertainty of the center-corrected row and column coordinate does not change
    row_unc2 = 0.25 + center_row_unc**2  # the uncertainty of the pixel coordinate is assumed to be 0.5 pixel
    col_unc2 = 0.25 + center_col_unc**2


    halfbinwidth = M_PI/N
    # categorize pixels into phi bins.
    for irow in range(data.shape[0]):
        for icolumn in range(data.shape[1]):
            #print(irow, icolumn)
            if mask[irow, icolumn] == 0:
                # this pixel is masked
                continue
            row = irow-center_row
            col = icolumn-center_col
            currentphi = atan2(-row, col)  # -pi <= currentphi <= pi
            if currentphi < -halfbinwidth:
                currentphi += 2* M_PI
            # now (-binwidth/2 <= currentphi < 2*M_PI-binwidth/2
            ibin = <Py_ssize_t>((currentphi + halfbinwidth)/(2*M_PI) * N)
            if ibin > N-1:
                continue
            currentphi_unc = fabs(row/col)/fabs(1+(row/col)**2) * sqrt(row_unc2/row**2 + col_unc2/col**2)
            tgtwotheta2 = (row**2+col**2)*pixelsize**2/distance**2
            # by employing trigonometric identities we get an algebraic form for tg(2theta) -> sin(theta).
            # While this is faster than the obvious sin(0.5*arctan(tg2theta)), it only works for 0 < 2theta < pi/2.
            # This is usually true in SAXS.
            sinth = (0.5*(1-1/(tgtwotheta2+1)**0.5))**0.5
            currentq = qfac * sinth
            # now bin the pixel
            if errorprop > 0: # 1,2,3
                Intensity[ibin] += data[irow, icolumn]
                if errorprop == 1:
                    Error[ibin] += error[irow, icolumn]
                else:
                    Error[ibin] += error[irow, icolumn]**2
                if errorprop == 3:
                    Intensity2[ibin] += data[irow, icolumn]**2
            else: # 0
                Intensity[ibin] += data[irow, icolumn] / error[irow, icolumn]**2
                Error[ibin] += 1/error[irow, icolumn]**2
            if phierrorprop > 0: # 1, 2 or 3
                phi[ibin] += currentphi
                if phierrorprop == 1:
                    phiError[ibin] += currentphi_unc
                else:
                    phiError[ibin] += currentphi_unc**2
                if phierrorprop == 3:
                    phi2[ibin] += currentphi**2
            else: # 0
                phi[ibin] += currentphi / currentphi_unc**2
                phiError[ibin] += currentphi_unc**2
            Area[ibin] += 1
            q[ibin] += currentq
            q2[ibin] += currentq**2

    # normalize the bins
    for ibin in range(N):
        if not Area[ibin]:
            # no pixels in this bin: set everything to NaN
            Intensity[ibin] = NAN
            Error[ibin] = NAN
            phi[ibin] = NAN
            phiError[ibin] = NAN
            qmean[ibin] = NAN
            qstd[ibin] = NAN
            continue
        if errorprop >0:
            if errorprop == 1:
                Error[ibin] /= Area[ibin]
            elif errorprop == 2:
                Error[ibin] = sqrt(Error[ibin])/Area[ibin]
            else: # errorprop == 3
                error_propagated = sqrt(Error[ibin]/Area[ibin])
                if Area[ibin] < 2:
                    error_statistics = 0
                else:
                    # sample standard deviation: sqrt((sum(I**2) - N*mean(I)**2)/(N-1))
                    # in our case, sum(I**2) is simply Intensity2[ibin].
                    # mean(I) = Intensity[ibin] / Area[ibin], thus N*mean(I)**2 is Intensity[ibin]**2/Area[ibin]
                    error_statistics = sqrt(
                        (Intensity2[ibin] - Intensity[ibin]**2/Area[ibin])/(Area[ibin]-1))
                Error[ibin] = max(error_propagated, error_statistics)
            Intensity[ibin] /= Area[ibin]
        else:
            Intensity[ibin] /= Error[ibin]
            Error[ibin] = 1/sqrt(Error[ibin])
        if phierrorprop >0:
            if phierrorprop == 1:
                phiError[ibin] /= Area[ibin]
            elif phierrorprop == 2:
                phiError[ibin] = sqrt(phiError[ibin])/Area[ibin]
            else: # phierrorprop == 3
                error_propagated = sqrt(phiError[ibin]/Area[ibin])
                if Area[ibin] < 2:
                    error_statistics = 0
                else:
                    # sample standard deviation: sqrt((sum(phi**2) - M*mean(phi)**2)/(M-1))
                    # in our case, sum(phi**2) is simply phi2[ibin].
                    # mean(phi) = phi[ibin] / Area[ibin], thus M*mean(phi)**2 is phi[ibin]**2/Area[ibin]
                    error_statistics = sqrt(
                        (phi2[ibin] - phi[ibin]**2/Area[ibin])/(Area[ibin]-1))
                phiError[ibin] = max(error_propagated, error_statistics)
            phi[ibin] /= Area[ibin]
        else:
            phi[ibin] /= phiError[ibin]
            phiError[ibin] = 1/sqrt(phiError[ibin])
        qmean[ibin] = q[ibin] / Area[ibin]
        if Area[ibin] <2:
            qstd[ibin] = 0
        else:
            qstd[ibin] = sqrt(q2[ibin] - q[ibin]**2/Area[ibin])/(Area[ibin]-1)
    return np.array(phi), np.array(Intensity), np.array(Error), np.array(phiError), np.array(Area), np.array(qmean), np.array(qstd)

def fastazimavg(double[:,:] data, uint8_t[:,:] mask,
                double center_row, double center_col,
                Py_ssize_t N):
    """
    Fast azimuthal averaging

    Inputs:
        data (np.ndarray, two dimensions, dtype: double) scattering pattern
        mask (np.ndarray, two dimensions, dtype: uint8) mask matrix
        center_row (double): row coordinate of the beam center
        center_col (double): column coordinate of the beam center
        N (Py_ssize_t): number of bins

    Outputs: phi, Intensity, Area
        (all one-dimensional np.ndarrays of length `N`)
        phi (dtype: double): azimuth angle of the bin
        Intensity (dtype: double): intensity of the bin
        Area (dtype: uint32): number of pixels in the bin
    """
    cdef:
        Py_ssize_t i, j, ibin
        double[:] pixel
        double[:] Intensity
        uint32_t[:] Area
        double r
        double halfbinwidth

    halfbinwidth= M_PI/N

    phi = np.zeros(N, np.double)
    Intensity = np.zeros(N, np.double)
    Area = np.zeros(N, np.uint32)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if mask[i,j] ==0:
                continue
            row = i-center_row
            col = j-center_col
            currentphi = atan2(-row, col)  # -pi <= currentphi <= pi
            if currentphi < - halfbinwidth:
                currentphi += 2* M_PI
            # now (-binwidth/2 <= currentphi < 2*M_PI-binwidth/2
            ibin = <Py_ssize_t>((currentphi + halfbinwidth)/(2*M_PI) * N)
            if ibin > N-1:
                continue
            phi[ibin] += currentphi
            Intensity[ibin] += data[i,j]
            Area[ibin] += 1
    for ibin in range(N):
        if Area[ibin] == 0:
            phi[ibin] = NAN
            Intensity[ibin] = NAN
        else:
            phi[ibin] /= Area[ibin]
            Intensity[ibin] /= Area[ibin]
    return np.array(phi), np.array(Intensity), np.array(Area)

