# pylint: disable-msg-cat=WCREFI
#cython: boundscheck=False
#cython: embedsignature=True
import numpy as np
cimport numpy as np
from libc.stdlib cimport *
from libc.math cimport *

cdef extern from "math.h":
    int isfinite(double)
    double INFINITY
    double floor(double)
    double ceil(double)
    double fmod(double,double)
    double fabs(double)


cdef inline double gaussian(double x0, double sigma, double x):
    return 1/sqrt(2*M_PI*sigma*sigma)*exp(-(x-x0)**2/(2*sigma**2))

def polartransform(np.ndarray[np.double_t, ndim=2] data not None,
                   np.ndarray[np.double_t, ndim=1] r,
                   np.ndarray[np.double_t, ndim=1] phi,
                   double origx, double origy):
    """Calculates a matrix of a polar representation of the image ("azimuthal
    regrouping").

    Inputs:
        data: the 2D matrix
        r: vector of polar radii
        phi: vector of polar angles (degrees)
        origx: the x (row) coordinate of the origin, starting from 0
        origy: the y (column) coordinate of the origin, starting from 0
    Outputs:
        pdata: a matrix of len(phi) rows and len(r) columns which contains the
            polar representation of the image.
    """
    cdef np.ndarray[np.double_t,ndim=2] pdata
    cdef Py_ssize_t lenphi,lenr
    cdef Py_ssize_t i,j
    cdef double x,y
    cdef Nrows,Ncols

    Nrows=data.shape[0]
    Ncols=data.shape[1]
    lenphi=len(phi)
    lenr=len(r)
    pdata=np.zeros((lenphi,lenr))

    for i from 0<=i<lenphi:
        for j from 0<=j<lenr:
            x=origx+r[j]*cos(phi[i]);
            y=origy+r[j]*sin(phi[i]);
            if (x>=0) and (y>=0) and (x<Nrows) and (y<Ncols):
                pdata[i,j]=data[x,y];
    return pdata

def autoqscale(double wavelength, double distance, double xres, double yres,
               double bcxa, double bcya,
               np.ndarray[np.uint8_t, ndim=2] mask not None):
    """Determine q-scale automatically

    Inputs:
        wavelength: wavelength in Angstroem
        distance: sample-detector distance in mm
        xres, yres: pixel size in mm
        bcxa, bcya: beam position (starting from 0)
        mask: mask matrix (1 means masked, 0 unmasked).

    Output: the q scale in a numpy vector. If either wavelength or distance or xres
        or yres is nonpositive, pixel vector is returned, which is guaranteed to
        be spaced by 1 pixels.
    """
    #determine the q-scale to be used automatically.
    cdef double qmin
    cdef double qmax
    cdef Py_ssize_t ix,iy,M,N
    cdef bint flagq

    flagq=(wavelength>0 and distance>0 and xres>0 and yres>0)
    M=mask.shape[0]; N=mask.shape[1];
    qmin=1e20; qmax=-10
    for ix from 0<=ix<M:
        for iy from 0<=iy<N:
            if mask[ix,iy]:
                continue
            x=((ix-bcxa)*xres)
            y=((iy-bcya)*yres)
            if flagq:
                q1=4*M_PI*sin(0.5*atan(sqrt(x*x+y*y)/distance))/wavelength
            else:
                q1=ceil(sqrt(x*x+y*y));
            if (q1>qmax):
                qmax=q1
            if (q1<qmin):
                qmin=q1
    if flagq:
        return np.linspace(qmin,qmax,sqrt(M*M+N*N)/2)
    else:
        return np.arange(qmin,qmax);

def radint_testarrays(np.ndarray[np.double_t,ndim=2] data,
                           np.ndarray[np.double_t, ndim=2] dataerr,
                           np.ndarray[np.uint8_t, ndim=2] mask):
    cdef Py_ssize_t M,N
    if (data is None):
        return (0,0);
    M=data.shape[0];
    N=data.shape[1];
    if (dataerr is not None) and \
        (dataerr.shape[0]!=(M) or dataerr.shape[1]!=(N)):
        return (0,0);
    if (mask is not None) and (mask.shape[0]!=(M) or mask.shape[1]!=(N)):
        return (0,0);
    return (M,N)

def radint_getres(res):
    #resolution
    cdef double xr, yr
    xr=-10;
    try:
        xr=res[0]; #exception may be raised here if res is not indexable.
        yr=res[1]; #or here if res has only one element.
    except:
        if xr<0: #first kind of exception occurred
            xr=res;
        yr=xr;
    return (xr,yr)

def radint(np.ndarray[np.double_t,ndim=2] data not None,
           np.ndarray[np.double_t,ndim=2] dataerr,
           double wavelength, double distance, res,
           double bcx, double bcy,
           np.ndarray[np.uint8_t, ndim=2] mask,
           np.ndarray[np.double_t, ndim=1] q=None,
           bint returnavgq=False, double phi0=0, double dphi=0,
           returnmask=False, bint symmetric_sector=False,
           bint doslice=False, bint returnpixel=False,
           int errorpropagation=2):
    """ Radial averaging of scattering images.

    Inputs:
        data: the intensity matrix
        dataerr: the error (standard deviation) matrix (of the same size as
            'data'). Or None to disregard it.
        wavelength: the real photon/neutron wavelength (Angstroem)
        distance: the distance from the sample to the detector
        res: pixel size. Either a vector (tuple) of two or a scalar. Must be
            expressed in the same units as the 'distance' parameter.
        bcx: the coordinate of the beam center along the first axis (row
            coordinates), starting from 0
        bcy: the coordinate of the beam center along the second axis (column
            coordiantes), starting from 0
        mask: the mask matrix (of the same size as 'data'). Nonzero is masked,
            zero is not masked. None to omit.
        q: the q (or pixel) points at which the integration is requested, in
            1/Angstroem (or pixel) units. If None, optimum range will be chosen
            automagically by taking the mask and the geometry into account.
        returnavgq: if True, returns the average q (or pixel) value for each
            bin, i.e. the average of the q (or pixel distance) values
            corresponding to the centers of the pixels which fell into each bin.
            False by default.
        phi0: starting angle if sector integration is requested. Expressed
            in radians.
        dphi: arc angle if sector integration is requested. Expressed in
            radians. OR, if sliceorsector is True, this is the width of the
            slice, in pixels. 0 implies simple radial integration without
            constraints.
        returnmask: True if the effective mask matrix is to be returned
            (0 for pixels taken into account, 1 for all the others).
        symmetric_sector: True if the mirror pair of the sector should be taken
            into account as well for sector integration. Pixels falling into
            sectors of width dphi, starting at phi0 and pi+phi0 will be used.
        doslice: True if slice, False (default) if sector (or just radial)
            integration is preferred. In the former case, dphi is interpreted as
            the width of the slice in pixel units.
        returnpixel: return pixel coordinates for integrated bins. False by
            default.
        errorpropagation: an integer number determining the type of error
            propagation. Can be:
            0: intensities falling to the same q-bin are assumed to be independent
                measurements of the same quantity, thus they will be weighted by
                the inverse squared of the error bars, and the error will be the
                inverse of the sum of these inverses.
            1: error bars are simply averaged, then divided by sqrt(# of pixels
                belonging to the bin).
            2: squared error propagation of independent quantities

    If any of 'wavelength', 'distance' or 'res' is zero or negative, pixel-based
    integration is done, with q denoting pixel everywhere.

    Outputs: q, Intensity, Error, Area, [effective mask], [pixel]
    """
    cdef double xres,yres
    cdef Py_ssize_t N,M
    cdef np.ndarray[np.double_t, ndim=1] qout
    cdef np.ndarray[np.double_t, ndim=1] Intensity
    cdef np.ndarray[np.double_t, ndim=1] Error
    cdef np.ndarray[np.double_t, ndim=1] Area
    cdef Py_ssize_t ix,iy
    cdef Py_ssize_t l
    cdef double x,y,q1
    cdef double * qmax
    cdef double *weight
    cdef double w
    cdef double rho
    cdef double phi0a,dphia
    cdef Py_ssize_t Numq
    cdef np.ndarray[np.uint8_t,ndim=2] maskout
    cdef double symmetric_sector_periodicity
    cdef double sinphi0,cosphi0
    cdef np.ndarray[np.double_t, ndim=1] pixelout
    cdef bint flagmask, flagerror, flagq

    #Process input data
    (xres,yres)=radint_getres(res)
    #array shapes
    M,N= radint_testarrays(data,dataerr,mask)
    if (M<=0) or (N<=0):
        raise ValueError('data, dataerr and mask should be of the same shape')
    flagerror=(dataerr is not None);
    flagmask=(mask is not None);
    #if any of wavelength, distance, res is nonpositive, pixel-based integration.
    flagq=((wavelength>0) and (distance>0) and (xres>0) and (yres>0));

    phi0a=phi0; dphia=dphi;
    # the equation of the line of the slice is x*sin(phi0)-y*cos(phi0)==0.
    # this is the normalized equation, ie. sin(phi0)**2+(-cos(phi0))**2=1,
    # therefore the signed distance of the point x,y from this line is
    # simply x*sin(phi0)-y*cos(phi0). We will use this to determine if a
    # point falls into this slice or not.
    sinphi0=sin(phi0a); cosphi0=cos(phi0a);
    if symmetric_sector:
        symmetric_sector_periodicity=1
    else:
        symmetric_sector_periodicity=2

    if returnmask:
        maskout=np.ones((data.shape[0],data.shape[1]),dtype=np.uint8)
    # if the q-scale was not supplied, create one.
    if q is None:
        if not flagmask:
            mask=np.zeros((data.shape[0],data.shape[1]),dtype=np.uint8)
        q=autoqscale(wavelength, distance, xres, yres, bcx, bcy, mask);
        if not flagmask:
            mask=None
    Numq=len(q)
    # initialize the output vectors
    Intensity=np.zeros(Numq,dtype=np.double)
    Error=np.zeros(Numq,dtype=np.double)
    Area=np.zeros(Numq,dtype=np.double)
    qout=np.zeros(Numq,dtype=np.double)
    pixelout=np.zeros(Numq,dtype=np.double)
    # set the upper bounds of the q-bins in qmax
    qmax=<double *>malloc(Numq*sizeof(double))
    weight=<double *>malloc(Numq*sizeof(double))
    for l from 0<=l<Numq:
        #initialize the weight and the qmax array.
        weight[l]=0
        if l==Numq-1:
            qmax[l]=q[Numq-1]
        else:
            qmax[l]=0.5*(q[l]+q[l+1])
    #loop through pixels
    for ix from 0<=ix<M: #rows
        for iy from 0<=iy<N: #columns
            if flagmask and (mask[ix,iy]): #if the pixel is masked, disregard it.
                continue
            if not isfinite(data[ix,iy]):
                #disregard nonfinite (NaN or inf) pixels.
                continue
            if flagerror and not isfinite(dataerr[ix,iy]):
                #disregard nonfinite (NaN or inf) pixels.
                continue
            if flagerror and errorpropagation==0 and dataerr[ix,iy]<=0:
                continue
            # coordinates of the pixel in length units (mm)
            x=((ix-bcx)*xres)
            y=((iy-bcy)*yres)
            if dphia>0: #slice or sector averaging.
                if doslice and fabs(sinphi0*x/xres-cosphi0*y/yres)>dphia:
                    continue
                if (not doslice) and fmod(atan2(y,x)-phi0a+M_PI*10,symmetric_sector_periodicity*M_PI)>dphia:
                    continue
            #normalized distance of the pixel from the origin
            if flagq:
                rho=sqrt(x*x+y*y)/distance
                q1=4*M_PI*sin(0.5*atan(rho))/wavelength
            else:
                q1=sqrt(x*x/xres/xres+y*y/yres/yres);
            if q1<q[0]: #q underflow
                continue
            if q1>q[Numq-1]: #q overflow
                continue
            if flagq:
                #weight, corresponding to the Jacobian (integral with respect to q,
                # not on the detector plane)
                w=(2*M_PI/wavelength/distance)**2*(2+rho**2+2*sqrt(1+rho**2))/( (1+rho**2+sqrt(1+rho**2))**2*sqrt(1+rho**2) )
            else:
                w=1;
            for l from 0<=l<Numq: # Find the q-bin
                if (q1>qmax[l]):
                    #not there yet
                    continue
                #we reach this point only if q1 is in the l-th bin. Calculate
                # the contributions of this pixel to the weighted average.
                if flagerror:
                    if errorpropagation==2:
                        Error[l]+=dataerr[ix,iy]**2*w
                    elif errorpropagation==1:
                        Error[l]+=dataerr[ix,iy]*w
                    else:
                        Error[l]+=1/dataerr[ix,iy]**2*w
                        w/=dataerr[ix,iy]**2
                qout[l]+=q1*w
                Intensity[l]+=data[ix,iy]*w
                Area[l]+=1
                weight[l]+=w
                if returnmask:
                    maskout[ix,iy]=0
                if returnpixel:
                    pixelout[l]+=sqrt((ix-bcx)**2+(iy-bcy)**2)*w
                break #avoid counting this pixel into higher q-bins.
    #normalize the results
    for l from 0<=l<Numq:
        if (weight[l]>0) and (Area[l]>0):
            qout[l]/=weight[l]
            Intensity[l]/=weight[l]
            if flagerror:
                if Error[l]==0:
                    pass
                elif errorpropagation==2:
                    Error[l]=sqrt(Error[l]/weight[l])
                elif errorpropagation==1:
                    Error[l]=Error[l]/weight[l]/Area[l]**0.5
                else:
                    Error[l]=sqrt(1/(Error[l]/weight[l]))
            pixelout[l]/=weight[l]
    #cleanup memory
    free(qmax)
    free(weight)
    #prepare return values
    if not returnavgq:
        qout=q
    output=[qout,Intensity]
    if flagerror:
        output.append(Error)
    output.append(Area)
    if returnmask:
        output.append(maskout)
    if returnpixel:
        output.append(pixelout)

    return tuple(output)

def radint_nsector(np.ndarray[np.double_t,ndim=2] data not None,
           np.ndarray[np.double_t,ndim=2] dataerr,
           double wavelength, double distance, res,
           double bcx, double bcy,
           np.ndarray[np.uint8_t, ndim=2] mask,
           np.ndarray[np.double_t, ndim=1] q=None,
           bint returnavgq=False, double phi0=0, double dphi=0,
           Py_ssize_t Nsector=4,
           returnmask=False, bint doslice=False, bint returnpixel=False,
           int errorpropagation =2):
    """ Radial averaging of scattering images: several sectors at once.

    Inputs:
        data: the intensity matrix
        dataerr: the error (standard deviation) matrix (of the same size as
            'data'). Or None to disregard it.
        wavelength: the real photon/neutron wavelength (Angstroem)
        distance: the distance from the sample to the detector
        res: pixel size. Either a vector (tuple) of two or a scalar. Must be
            expressed in the same units as the 'distance' parameter.
        bcx: the coordinate of the beam center along the first axis (row
            coordinates), starting from 0
        bcy: the coordinate of the beam center along the second axis (column
            coordiantes), starting from 0
        mask: the mask matrix (of the same size as 'data'). Nonzero is masked,
            zero is not masked. None to omit.
        q: the q (or pixel) points at which the integration is requested, in
            1/Angstroem (or pixel) units. If None, optimum range will be chosen
            automagically by taking the mask and the geometry into account.
        returnavgq: if True, returns the average q (or pixel) value for each
            bin, i.e. the average of the q (or pixel distance) values
            corresponding to the centers of the pixels which fell into each bin.
            False by default.
        phi0: starting angle if sector integration is requested. Expressed
            in radians.
        dphi: arc angle if sector integration is requested. Expressed in
            radians. OR, if sliceorsector is True, this is the width of the
            slice, in pixels. 0 implies simple radial integration without
            constraints.
        Nsector: number of sectors or slices. This many sectors of angular width
            dphi will be chosen in a rotation-symmetric manner, starting from
            phi0.
        returnmask: True if the effective mask matrix is to be returned
            (0 for pixels taken into account, 1 for all the others).
        doslice: True if slice, False (default) if sector (or just radial)
            integration is preferred. In the former case, dphi is interpreted as
            the width of the slice in pixel units.
        returnpixel: return pixel coordinates for integrated bins. False by
            default.
        errorpropagation: an integer number determining the type of error
            propagation. Can be:
            0: intensities falling to the same q-bin are assumed to be independent
                measurements of the same quantity, thus they will be weighted by
                the inverse squared of the error bars, and the error will be the
                inverse of the sum of these inverses.
            1: error bars are simply averaged, then divided by sqrt(# of pixels
                belonging to the bin).
            2: squared error propagation of independent quantities

    If any of 'wavelength', 'distance' or 'res' is zero or negative, pixel-based
    integration is done, with q denoting pixel everywhere.

    Outputs: q, Intensity, Error, Area, [effective mask], [pixel]
    """
    cdef double xres,yres
    cdef Py_ssize_t N,M
    cdef np.ndarray[np.double_t, ndim=2] qout
    cdef np.ndarray[np.double_t, ndim=2] Intensity
    cdef np.ndarray[np.double_t, ndim=2] Error
    cdef np.ndarray[np.double_t, ndim=2] Area
    cdef Py_ssize_t ix,iy
    cdef Py_ssize_t l
    cdef double x,y,q1
    cdef double * qmax
    cdef double *weight
    cdef double w
    cdef double rho
    cdef double phi0a,dphia
    cdef Py_ssize_t Numq
    cdef np.ndarray[np.uint8_t,ndim=2] maskout
    cdef double symmetric_sector_periodicity
    cdef double *sinphi0
    cdef double *cosphi0
    cdef np.ndarray[np.double_t, ndim=2] pixelout
    cdef bint flagmask, flagerror, flagq
    cdef Py_ssize_t sector_idx

    #Process input data
    (xres,yres)=radint_getres(res)
    #array shapes
    M,N= radint_testarrays(data,dataerr,mask)
    if (M<=0) or (N<=0):
        raise ValueError('data, dataerr and mask should be of the same shape')
    flagerror=(dataerr is not None);
    flagmask=(mask is not None);
    #if any of wavelength, distance, res is nonpositive, pixel-based integration.
    flagq=((wavelength>0) and (distance>0) and (xres>0) and (yres>0));

    phi0a=phi0; dphia=dphi;
    # the equation of the line of the slice is x*sin(phi0)-y*cos(phi0)==0.
    # this is the normalized equation, ie. sin(phi0)**2+(-cos(phi0))**2=1,
    # therefore the signed distance of the point x,y from this line is
    # simply x*sin(phi0)-y*cos(phi0). We will use this to determine if a
    # point falls into this slice or not.
    sinphi0=<double*>malloc(Nsector*sizeof(double))
    cosphi0=<double*>malloc(Nsector*sizeof(double))
    for l from 0<=l<Nsector:
        sinphi0[l]=sin(phi0a+2*M_PI/Nsector*l)
        cosphi0[l]=cos(phi0a+2*M_PI/Nsector*l)

    if returnmask:
        maskout=np.ones((data.shape[0],data.shape[1]),dtype=np.uint8)
    # if the q-scale was not supplied, create one.
    if q is None:
        if not flagmask:
            mask=np.zeros((data.shape[0],data.shape[1]),dtype=np.uint8)
        q=autoqscale(wavelength, distance, xres, yres, bcx, bcy, mask);
        if not flagmask:
            mask=None
    Numq=len(q)
    # initialize the output vectors
    Intensity=np.zeros((Numq,Nsector),dtype=np.double)
    Error=np.zeros((Numq,Nsector),dtype=np.double)
    Area=np.zeros((Numq,Nsector),dtype=np.double)
    qout=np.zeros((Numq,Nsector),dtype=np.double)
    pixelout=np.zeros((Numq,Nsector),dtype=np.double)
    # set the upper bounds of the q-bins in qmax
    qmax=<double *>malloc(Numq*sizeof(double))
    weight=<double *>malloc(Numq*Nsector*sizeof(double))
    for l from 0<=l<Numq:
        #initialize the weight and the qmax array.
        weight[l]=0
        if l==Numq-1:
            qmax[l]=q[Numq-1]
        else:
            qmax[l]=0.5*(q[l]+q[l+1])
    sector_idx=0;
    #loop through pixels
    for ix from 0<=ix<M: #rows
        for iy from 0<=iy<N: #columns
            if flagmask and (mask[ix,iy]): #if the pixel is masked, disregard it.
                continue
            if not isfinite(data[ix,iy]):
                #disregard nonfinite (NaN or inf) pixels.
                continue
            if flagerror and not isfinite(dataerr[ix,iy]):
                #disregard nonfinite (NaN or inf) pixels.
                continue
            if flagerror and errorpropagation==0 and dataerr[ix,iy]<=0:
                continue
            # coordinates of the pixel in length units (mm)
            x=((ix-bcx)*xres)
            y=((iy-bcy)*yres)
            #find the sector this point falls into. Trick: start the iteration
            # with the last found sector. Maybe this reduces the number of
            # iterations needed for finding the appropriate sector index.
            for l from sector_idx<=l<(sector_idx+Nsector):
                if doslice and fabs(sinphi0[l%Nsector]*x/xres-cosphi0[l%Nsector]*y/yres)>dphia:
                    break
                if (not doslice) and fmod(atan2(y,x)-phi0a-2*M_PI/Nsector*(l%Nsector)+M_PI*10*Nsector,2*M_PI)<dphia:
                    break
            #print l,
            if l>=(sector_idx+Nsector): #point does not fall in any of the sectors
                continue
            sector_idx=l%Nsector # normalize the sector index.
            #print sector_idx,
            #normalized distance of the pixel from the origin
            if flagq:
                rho=sqrt(x*x+y*y)/distance
                q1=4*M_PI*sin(0.5*atan(rho))/wavelength
            else:
                q1=sqrt(x*x/xres/xres+y*y/yres/yres);
            if q1<q[0]: #q underflow
                continue
            if q1>q[Numq-1]: #q overflow
                continue
            if flagq:
                #weight, corresponding to the Jacobian (integral with respect to q,
                # not on the detector plane)
                w=(2*M_PI/wavelength/distance)**2*(2+rho**2+2*sqrt(1+rho**2))/( (1+rho**2+sqrt(1+rho**2))**2*sqrt(1+rho**2) )
            else:
                w=1;
            for l from 0<=l<Numq: # Find the q-bin
                if (q1>qmax[l]):
                    #not there yet
                    continue
                #we reach this point only if q1 is in the l-th bin. Calculate
                # the contributions of this pixel to the weighted average.
                if flagerror:
                    if errorpropagation==2:
                        Error[l, sector_idx]+=dataerr[ix,iy]**2*w
                    elif errorpropagation==1:
                        Error[l,sector_idx]+=dataerr[ix,iy]*w
                    else:
                        Error[l,sector_idx]+=1/dataerr[ix,iy]**2
                        w/=dataerr[ix,iy]**2
                qout[l,sector_idx]+=q1*w
                Intensity[l,sector_idx]+=data[ix,iy]*w
                if flagerror:
                    Error[l,sector_idx]+=dataerr[ix,iy]**2*w
                Area[l,sector_idx]+=1
                weight[l*Nsector+sector_idx]+=w
                if returnmask:
                    maskout[ix,iy]=0
                if returnpixel:
                    pixelout[l,sector_idx]+=sqrt((ix-bcx)**2+(iy-bcy)**2)*w
                break #avoid counting this pixel into higher q-bins.
    #normalize the results
    for sector_idx from 0<=sector_idx<Nsector:
        for l from 0<=l<Numq:
            if (weight[l*Nsector+sector_idx]>0) and (Area[l*Nsector+sector_idx]>0):
                qout[l,sector_idx]/=weight[l*Nsector+sector_idx]
                Intensity[l,sector_idx]/=weight[l*Nsector+sector_idx]
                if flagerror:
                    if Error[l, sector_idx]==0:
                        pass
                    elif errorpropagation==2:
                        Error[l,sector_idx]=sqrt(Error[l,sector_idx]/weight[l*Nsector+sector_idx])
                    elif errorpropagation==1:
                        Error[l,sector_idx]=Error[l,sector_idx]/weight[l*Nsector+sector_idx]/Area[l*Nsector+sector_idx]**0.5
                    else:
                        Error[l,sector_idx]=sqrt(1/(Error[l, sector_idx]/weight[l*Nsector+sector_idx]))
                pixelout[l,sector_idx]/=weight[l*Nsector+sector_idx]
    #cleanup memory
    free(qmax)
    free(weight)
    free(sinphi0)
    free(cosphi0)
    #prepare return values
    if not returnavgq:
        for l from 0<=l<Nsector:
            qout[:,l]=q
    output=[qout,Intensity]
    if flagerror:
        output.append(Error)
    output.append(Area)
    if returnmask:
        output.append(maskout)
    if returnpixel:
        output.append(pixelout)

    return tuple(output)



def radint_fullq(np.ndarray[np.double_t,ndim=2] data not None,
           np.ndarray[np.double_t,ndim=2] dataerr,
           double wavelength, double distance, res,
           double bcx, double bcy,
           np.ndarray[np.uint8_t, ndim=2] mask,
           np.ndarray[np.double_t, ndim=1] q=None,
           bint returnavgq=False,
           returnmask=False,int errorpropagation =2):
    """ Radial averaging of scattering images.

    Inputs:
        data: the intensity matrix
        dataerr: the error (standard deviation) matrix (of the same size as
            'data'). Or None to disregard it.
        wavelength: the real photon/neutron wavelength (Angstroem)
        distance: the distance from the sample to the detector
        res: pixel size. Either a vector (tuple) of two or a scalar. Must be
            expressed in the same units as the 'distance' parameter.
        bcx: the coordinate of the beam center along the first axis (row
            coordinates), starting from 0
        bcy: the coordinate of the beam center along the second axis (column
            coordiantes), starting from 0
        mask: the mask matrix (of the same size as 'data'). Nonzero is masked,
            zero is not masked. None to omit.
        q: the q (or pixel) points at which the integration is requested, in
            1/Angstroem (or pixel) units. If None, optimum range will be chosen
            automagically by taking the mask and the geometry into account.
        returnavgq: if True, returns the average q (or pixel) value for each
            bin, i.e. the average of the q (or pixel distance) values
            corresponding to the centers of the pixels which fell into each bin.
            False by default.
        returnmask: True if the effective mask matrix is to be returned
            (0 for pixels taken into account, 1 for all the others).
        errorpropagation: an integer number determining the type of error
            propagation. Can be:
            0: intensities falling to the same q-bin are assumed to be independent
                measurements of the same quantity, thus they will be weighted by
                the inverse squared of the error bars, and the error will be the
                inverse of the sum of these inverses.
            1: error bars are simply averaged, then divided by sqrt(# of pixels
                belonging to the bin).
            2: squared error propagation of independent quantities


    Outputs: q, Intensity, Error, Area, [effective mask]
    """
    cdef double xres,yres
    cdef Py_ssize_t N,M
    cdef np.ndarray[np.double_t, ndim=1] qout
    cdef np.ndarray[np.double_t, ndim=1] Intensity
    cdef np.ndarray[np.double_t, ndim=1] Error
    cdef np.ndarray[np.double_t, ndim=1] Area
    cdef Py_ssize_t ix,iy
    cdef Py_ssize_t l
    cdef double x,y,q1
    cdef double * qmax
    cdef double *weight
    cdef double w
    cdef double rho
    cdef Py_ssize_t Numq
    cdef np.ndarray[np.uint8_t,ndim=2] maskout
    cdef bint flagmask, flagerror

    #Process input data
    (xres,yres)=radint_getres(res)
    #array shapes
    M,N= radint_testarrays(data,dataerr,mask)
    if (M<=0) or (N<=0):
        raise ValueError('data, dataerr and mask should be of the same shape')
    flagerror=(dataerr is not None);
    flagmask=(mask is not None);

    if returnmask:
        maskout=np.ones((data.shape[0],data.shape[1]),dtype=np.uint8)
    # if the q-scale was not supplied, create one.
    if q is None:
        if not flagmask:
            mask=np.zeros((data.shape[0],data.shape[1]),dtype=np.uint8)
        q=autoqscale(wavelength, distance, xres, yres, bcx, bcy, mask);
        if not flagmask:
            mask=None
    Numq=len(q)
    # initialize the output vectors
    Intensity=np.zeros(Numq,dtype=np.double)
    Error=np.zeros(Numq,dtype=np.double)
    Area=np.zeros(Numq,dtype=np.double)
    qout=np.zeros(Numq,dtype=np.double)
    # set the upper bounds of the q-bins in qmax
    qmax=<double *>malloc(Numq*sizeof(double))
    weight=<double *>malloc(Numq*sizeof(double))
    for l from 0<=l<Numq:
        #initialize the weight and the qmax array.
        weight[l]=0
        if l==Numq-1:
            qmax[l]=q[Numq-1]
        else:
            qmax[l]=0.5*(q[l]+q[l+1])
    #loop through pixels
    for ix from 0<=ix<M: #rows
        for iy from 0<=iy<N: #columns
            if flagmask and (mask[ix,iy]): #if the pixel is masked, disregard it.
                continue
            if not isfinite(data[ix,iy]):
                #disregard nonfinite (NaN or inf) pixels.
                continue
            if flagerror and not isfinite(dataerr[ix,iy]):
                #disregard nonfinite (NaN or inf) pixels.
                continue
            if flagerror and errorpropagation==0 and dataerr[ix,iy]<=0:
                continue
            # coordinates of the pixel in length units (mm)
            x=((ix-bcx)*xres)
            y=((iy-bcy)*yres)
            #normalized distance of the pixel from the origin
            rho=sqrt(x*x+y*y)/distance
            q1=4*M_PI*sin(0.5*atan(rho))/wavelength
            if q1<q[0]: #q underflow
                continue
            if q1>q[Numq-1]: #q overflow
                continue
            #weight, corresponding to the Jacobian (integral with respect to q,
            # not on the detector plane)
            w=(2*M_PI/wavelength/distance)**2*(2+rho**2+2*sqrt(1+rho**2))/( (1+rho**2+sqrt(1+rho**2))**2*sqrt(1+rho**2) )
            for l from 0<=l<Numq: # Find the q-bin
                if (q1>qmax[l]):
                    #not there yet
                    continue
                #we reach this point only if q1 is in the l-th bin. Calculate
                # the contributions of this pixel to the weighted average.
                if flagerror:
                    if errorpropagation==2:
                        Error[l]+=dataerr[ix,iy]**2*w
                    elif errorpropagation==1:
                        Error[l]+=dataerr[ix,iy]*w
                    else:
                        Error[l]+=1/dataerr[ix,iy]**2
                        w/=dataerr[ix,iy]**2
                qout[l]+=q1*w
                Intensity[l]+=data[ix,iy]*w
                Area[l]+=1
                weight[l]+=w
                if returnmask:
                    maskout[ix,iy]=0
                break #avoid counting this pixel into higher q-bins.
    #normalize the results
    for l from 0<=l<Numq:
        if (weight[l]>0) and (Area[l]>0):
            qout[l]/=weight[l]
            Intensity[l]/=weight[l]
            if flagerror:
                if Error[l]==0:
                    pass
                elif errorpropagation==2:
                    Error[l]=sqrt(Error[l]/weight[l])
                elif errorpropagation==1:
                    Error[l]=Error[l]/weight[l]/Area[l]**0.5
                else:
                    Error[l]=sqrt(1/(Error[l]/weight[l]))
    #cleanup memory
    free(qmax)
    free(weight)
    #prepare return values
    if not returnavgq:
        qout=q
    output=[qout,Intensity]
    if flagerror:
        output.append(Error)
    output.append(Area)
    if returnmask:
        output.append(maskout)
    return tuple(output)


def azimint(np.ndarray[np.double_t, ndim=2] data not None,
            np.ndarray[np.double_t, ndim=2] dataerr, # error can be None
            double wavelength, double distance, res, double bcx, double bcy,
            np.ndarray[np.uint8_t, ndim=2] mask, Ntheta=100,
            double qmin=0, double qmax=INFINITY, bint returnmask=False,
            int errorpropagation =2):
    """Perform azimuthal integration of image, with respect to q values

    Inputs:
        data: matrix to average
        dataerr: error matrix. If not applicable, set it to None
        wavelength: wave length in Angstroem
        distance: sample-detector distance
        res: resolution (pixel size) of the detector (mm/pixel)
        bcx, bcy: beam center coordinates, starting from 0.
        mask: mask matrix; 1 means masked, 0 means non-masked
        Ntheta: number of desired points on the abscissa
        qmin: the lower bound of the circle stripe (expressed in q units)
        qmax: the upper bound of the circle stripe (expressed in q units)
        returnmask: if True, a mask is returned, only the pixels taken into
            account being unmasked (0).
        errorpropagation: an integer number determining the type of error
            propagation. Can be:
            0: intensities falling to the same q-bin are assumed to be independent
                measurements of the same quantity, thus they will be weighted by
                the inverse squared of the error bars, and the error will be the
                inverse of the sum of these inverses.
            1: error bars are simply averaged
            2: squared error propagation of independent quantities


    Outputs: theta,I,[E],A,[mask]

    Note: if any of 'wavelength', 'distance' or 'res' is nonpositive, q means pixel
        everywhere.
    """
    cdef np.ndarray[np.double_t, ndim=1] theta, I, E, A
    cdef Py_ssize_t ix,iy, M, N, index, Ntheta1, escaped
    cdef double d, x, y, phi
    cdef double q
    cdef int errorwasnone
    cdef double resx,resy
    cdef np.ndarray[np.uint8_t, ndim=2] maskout
    cdef bint flagerror, flagmask

    (resx,resy)=radint_getres(res);

    (M,N)=radint_testarrays(data,dataerr,mask)
    if (M<=0) or (N<=0):
        raise ValueError('data, dataerr and mask should be of the same shape')
    Ntheta1=<Py_ssize_t>floor(Ntheta)

    theta=np.linspace(0,2*np.pi,Ntheta1) # the abscissa of the results
    I=np.zeros(Ntheta1,dtype=np.double) # vector of intensities
    A=np.zeros(Ntheta1,dtype=np.double) # vector of effective areas
    E=np.zeros(Ntheta1,dtype=np.double)
    weight=np.zeros(Ntheta1, dtype=np.double)
    if returnmask:
        maskout=np.ones([data.shape[0],data.shape[1]],dtype=np.uint8)

    flagerror=(dataerr is not None)
    flagmask=(mask is not None)
    flagq=(distance>0 and wavelength>0 and resx>0 and resy>0);
    for ix from 0<=ix<M:
        for iy from 0<=iy<N:
            if flagmask and mask[ix,iy]:
                continue
            if flagerror and not isfinite(dataerr[ix,iy]):
                continue
            if flagerror and errorpropagation==0 and dataerr[ix,iy]<=0:
                continue
            x=(ix-bcx)*resx
            y=(iy-bcy)*resy
            d=sqrt(x**2+y**2)
            w=1
            if flagq:
                q=4*M_PI*sin(0.5*atan2(d,distance))/wavelength
            else:
                q=sqrt(x*x/resx/resx+y*y/resy/resy);
            if (q<qmin) or (q>qmax):
                continue
            phi=atan2(y,x)
            index=<Py_ssize_t>floor(phi/(2*M_PI)*Ntheta1)
            if index>=Ntheta1:
                continue
            if flagerror:
                if errorpropagation==2:
                    E[index]+=dataerr[ix,iy]**2
                elif errorpropagation==1:
                    E[index]+=dataerr[ix,iy]
                else:
                    E[index]+=1/dataerr[ix,iy]**2
                    w=1/dataerr[ix,iy]**2
            I[index]+=data[ix,iy]*w
            weight[index]+=w
            A[index]+=1
            if returnmask:
                maskout[ix,iy]=0
    #print "Escaped: ",escaped
    for index from 0<=index<Ntheta1:
        if A[index]>0:
            I[index]/=weight[index]
            if flagerror:
                if E[index]==0:
                    pass
                elif errorpropagation==2:
                    E[index]=sqrt(E[index]/weight[index])
                elif errorpropagation==1:
                    E[index]=E[index]/weight[index]/A[index]**0.5
                else:
                    E[index]=sqrt(1/(E[index]/weight[index]))
    ret=[theta,I]
    if flagerror:
        ret.append(E)
    ret.append(A)
    if returnmask:
        ret.append(maskout)
    return tuple(ret)

def bin2D(np.ndarray[np.double_t, ndim=2] M, Py_ssize_t xlen, Py_ssize_t ylen):
    """def bin2D(np.ndarray[np.double_t, ndim=2] M, Py_ssize_t xlen, Py_ssize_t ylen):

    Binning of a 2D matrix.

    Inputs:
        M: the matrix as a numpy array of type double
        xlen: this many pixels in the x (row) direction will be added up
        ylen: this many pixels in the y (column) direction will be added up.

    Output: the binned matrix

    Notes:
        each pixel of the returned matrix will be the sum of an xlen-times-ylen
            block in the original matrix.
    """
    cdef Py_ssize_t i,i1,j,j1
    cdef Py_ssize_t Nx,Ny
    cdef np.ndarray[np.double_t,ndim=2] N

    Nx=M.shape[0]/xlen
    Ny=M.shape[1]/ylen

    N=np.zeros((Nx,Ny),np.double)
    for i from 0<=i<Nx:
        for i1 from 0<=i1<xlen:
            for j from 0<=j<Ny:
                for j1 from 0<=j1<ylen:
                    N[i,j]+=M[i*xlen+i1,j*ylen+j1]
    return N/(xlen*ylen)

def calculateDmatrix(np.ndarray[np.uint8_t, ndim=2] mask, res, double bcx,
                     double bcy):
    """Calculate distances of pixels from the origin

    Inputs:
        mask: mask matrix (only its shape is used)
        res: pixel size in mm-s. Can be a vector of length 2 or a scalar
        bcx: Beam center in pixels, in the row direction, starting from 0
        bcy: Beam center in pixels, in the column direction, starting from 0

    Output:
        A matrix of the shape of <mask>. Each element contains the distance
        of the centers of the pixels from the origin (bcx,bcy), expressed in
        mm-s.
    """
    cdef np.ndarray[np.double_t, ndim=2] D
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef double res0
    cdef double res1
    cdef Py_ssize_t N
    cdef Py_ssize_t M

    if type(res)!=type([]) and type(res)!=type(tuple()):
        res0=res;
        res1=res;
    elif len(res)!=2:
        raise ValueError,'Argument <res> should either be a number, or a list of two.'
    else:
        res0=res[0]
        res1=res[1]
    M=mask.shape[0]
    N=mask.shape[1]
    D=np.zeros((mask.shape[0],mask.shape[1]))
    for i from 0<=i<M:
        for j from 0<=j<N:
            D[i,j]=sqrt((res0*(i-bcx))**2+(res1*(j-bcy))**2)
    return D

def twodimfromonedim(Py_ssize_t Nrows, Py_ssize_t Ncols, double pixelsize,
                     Py_ssize_t Nsubdiv, double bcx, double bcy, curvefunc):
    """def twodimfromonedim(Nrows, Ncols, pixelsize, Nsubdiv, bcx, bcy, curvefunc):

    Generate a two-dimensional matrix from a one-dimensional curve

    Inputs:
        Nrows: number of rows in the output matrix
        Ncols: number of columns in the output matrix
        pixelsize: the size of the pixel (i.e. mm)
        Nsubdiv: number of subpixels in each direction (to take the finite pixel
            size of real detectors into account)
        bcx: row coordinate of the beam center in pixels, counting starts with 0
        bcy: column coordinate of the beam center in pixels, counting starts
            with 0
        curvefunc: a Python function. Takes exactly one argument and should
            return a scalar (double).

    Output:
        the scattering matrix.
    """
    cdef Py_ssize_t i,j,k,l
    cdef np.ndarray[np.double_t, ndim=2] out
    cdef double tmp,r
    cdef double h, x, y
    out=np.zeros((Nrows,Ncols),np.double)
    h=pixelsize/Nsubdiv
    for i from 0<=i<Nrows:
        x=(i-bcx)*pixelsize
        for j from 0<=j<Ncols:
            tmp=0
            y=(j-bcy)*pixelsize
            for k from 0<=k<Nsubdiv:
                for l from 0<=l<Nsubdiv:
                    r=sqrt((x+(k+0.5)*h)**2+(y+(l+0.5)*h)**2)
                    tmp+=curvefunc(r)
            out[i,j]=tmp
    return out

def groupsum(np.ndarray[np.long_t, ndim=1] groupidx not None, np.ndarray[np.double_t, ndim=2] tosum not None):
    indices=np.unique(groupidx)
    outputdata=np.zeros((tosum.shape[1],len(indices)),dtype=np.double)
    for i from 0<=i<tosum.shape[0]:
        outputdata[:,groupidx[i]]+=tosum[i,:]
    return outputdata
