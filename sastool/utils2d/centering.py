import numpy as np
from integrate import radintpix, azimintpix
import scipy.optimize
from .. import misc

def findbeam_gravity(data,mask):
    """Find beam center with the "gravity" method

    Inputs:
        data: scattering image
        mask: mask matrix

    Output:
        a vector of length 2 with the x (row) and y (column) coordinates
         of the origin, starting from 1
    """
    # for each row and column find the center of gravity
    data1=data.copy() # take a copy, because elements will be tampered with
    data1[mask==0]=0 # set masked elements to zero
    # vector of x (row) coordinates
    x=np.arange(data1.shape[0])
    # vector of y (column) coordinates
    y=np.arange(data1.shape[1])
    # two column vectors, both containing ones. The length of onex and
    # oney corresponds to length of x and y, respectively.
    onex=np.ones((len(x),1))
    oney=np.ones((len(y),1))
    # Multiply the matrix with x. Each element of the resulting column
    # vector will contain the center of gravity of the corresponding row
    # in the matrix, multiplied by the "weight". Thus: nix_i=sum_j( A_ij
    # * x_j). If we divide this by spamx_i=sum_j(A_ij), then we get the
    # center of gravity. The length of this column vector is len(y).
    nix=np.dot(data1,x).flatten()
    spamx=np.dot(data1,onex).flatten()
    # indices where both nix and spamx is nonzero.
    goodx=((nix!=0) & (spamx!=0))
    # trim y, nix and spamx by goodx, eliminate invalid points.
    nix=nix[goodx]
    spamx=spamx[goodx]

    # now do the same for the column direction.
    niy=np.dot(data1.T,y).flatten()
    spamy=np.dot(data1.T,oney).flatten()
    goody=((niy!=0) & (spamy!=0))
    niy=niy[goody]
    spamy=spamy[goody]
    # column coordinate of the center in each row will be contained in
    # ycent, the row coordinate of the center in each column will be
    # in xcent.
    ycent=nix/spamx
    xcent=niy/spamy
    # return the mean values as the centers.
    return [xcent.mean(),ycent.mean()]

def findbeam_slices(data,orig_initial,mask=None,maxiter=0,epsfcn=0.001,
                      dmin=0, dmax=np.inf, callback=None):
    """Find beam center with the "slices" method
    
    Inputs:
        data: scattering matrix
        orig_initial: estimated value for x (row) and y (column)
            coordinates of the beam center, starting from 1.
        mask: mask matrix. If None, nothing will be masked. Otherwise it
            should be of the same size as data. Nonzero means non-masked.
        maxiter: maximum number of iterations for scipy.optimize.leastsq
        epsfcn: input for scipy.optimize.leastsq
        dmin: disregard pixels nearer to the origin than this
        dmax: disregard pixels farther from the origin than this
        callback: callback function (expects no arguments)
        
    Output:
        a vector of length 2 with the x (row) and y (column) coordinates
         of the origin.
    """
    sector_wid=np.pi/9.
    if mask is None:
        mask=np.ones(data.shape)
    def targetfunc(orig,data,mask,callback):
        #integrate four sectors
        p=[None]*4;
        I=[None]*4; A=[None]*4
        for i in range(4):
            p[i],I[i],A[i]=radintpix(data, None, orig[0], orig[1], mask=mask,
                                     phi0=(i*2+1)*np.pi/4-0.5*sector_wid,
                                     dphi=(i*2+1)*np.pi/4+0.5*sector_wid)
        minpix=max(max([x.min() for x in p]),dmin)
        maxpix=min(min([x.max() for x in p]),dmax)
        if (maxpix<minpix):
            raise ValueError('The four slices do not overlap! Please give a\
 better approximation for the origin or use another centering method.')
        for i in range(4):
            I[i]=I[i][(p[i]>=minpix)&(p[i]<=maxpix)];
        ret= ((I[0]-I[2])**2+(I[1]-I[3])**2)/(maxpix-minpix)
        if callback is not None:
            callback()
        return ret
    orig=scipy.optimize.leastsq(targetfunc,np.array(orig_initial),args=(data,1-mask.astype(np.uint8),callback),maxfev=maxiter,epsfcn=0.01)
    return orig[0]

def findbeam_azimuthal(data, orig_initial, mask=None, maxiter=100, Ntheta=50,
                         dmin=0, dmax=np.inf, callback=None):
    """Find beam center using azimuthal integration
    
    Inputs:
        data: scattering matrix
        orig_initial: estimated value for x (row) and y (column)
            coordinates of the beam center, starting from 1.
        mask: mask matrix. If None, nothing will be masked. Otherwise it
            should be of the same size as data. Nonzero means non-masked.
        maxiter: maximum number of iterations for scipy.optimize.fmin
        Ntheta: the number of theta points for the azimuthal integration
        dmin: pixels nearer to the origin than this will be excluded from
            the azimuthal integration
        dmax: pixels farther from the origin than this will be excluded from
            the azimuthal integration
        callback: callback function (expects no arguments)
    Output:
        a vector of length 2 with the x and y coordinates of the origin,
            starting from 1
    """
    if mask is None:
        mask=np.ones(data.shape)
    def targetfunc(orig,data,mask,callback):
        def sinfun(p,x,y):
            return (y-np.sin(x+p[1])*p[0]-p[2])/np.sqrt(len(x))
        t,I,a=azimintpix(data,None,orig[0],orig[1],mask.astype('uint8'),Ntheta,dmin,dmax)
        if len(a)>(a>0).sum():
            raise ValueError,'findbeam_azimuthal: non-complete azimuthal average, please consider changing dmin, dmax and/or orig_initial!'
        p=((I.max()-I.min())/2.0,t[I==I.max()][0],I.mean())
        p=scipy.optimize.leastsq(sinfun,p,(t,I))[0]
        #print "findbeam_azimuthal: orig=",orig,"amplitude=",abs(p[0])
        if callback is not None:
            callback()
        return abs(p[0])
    orig1=scipy.optimize.fmin(targetfunc,np.array(orig_initial),args=(data,1-mask,callback),maxiter=maxiter,disp=0)
    return orig1

def findbeam_azimuthal_fold(data, orig_initial, mask=None, maxiter=100,
                               Ntheta=50, dmin=0, dmax=np.inf, callback=None):
    """Find beam center using azimuthal integration and folding
    
    Inputs:
        data: scattering matrix
        orig_initial: estimated value for x (row) and y (column)
            coordinates of the beam center, starting from 1.
        mask: mask matrix. If None, nothing will be masked. Otherwise it
            should be of the same size as data. Nonzero means non-masked.
        maxiter: maximum number of iterations for scipy.optimize.fmin
        Ntheta: the number of theta points for the azimuthal integration.
            Should be even!
        dmin: pixels nearer to the origin than this will be excluded from
            the azimuthal integration
        dmax: pixels farther from the origin than this will be excluded from
            the azimuthal integration
        callback: callback function (expects no arguments)
    Output:
        a vector of length 2 with the x and y coordinates of the origin,
            starting from 1
    """
    if Ntheta%2:
        raise ValueError('Ntheta should be even!')
    if mask is None:
        mask=np.ones_like(data)
    
    #the function to minimize is the sum of squared difference of two halves of
    # the azimuthal integral.
    def targetfunc(orig,data,mask,callback):
        I=azimintpix(data,None,orig[0],orig[1],mask.astype(np.uint8),Ntheta,dmin,dmax)[1]
        if callback is not None:
            callback()
        return np.sum((I[:Ntheta/2]-I[Ntheta/2:])**2)/Ntheta
    orig1=scipy.optimize.fmin(targetfunc,np.array(orig_initial),args=(data,1-mask,callback),maxiter=maxiter,disp=0)
    return orig1

def findbeam_semitransparent(data,pri):
    """Find beam with 2D weighting of semitransparent beamstop area

    Inputs:
        data: scattering matrix
        pri: list of four: [xmin,xmax,ymin,ymax] for the borders of the beam
            area under the semitransparent beamstop. X corresponds to the column
            index (ie. A[Y,X] is the element of A from the Xth column and the 
            Yth row). You can get these by zooming on the figure and retrieving
            the result of axis() (like in Matlab)

    Outputs: bcx,bcy
        the x and y coordinates of the primary beam
    """
    threshold=0.05
    rowmin=np.floor(min(pri[2:]))
    rowmax=np.ceil(max(pri[2:]))
    colmin=np.floor(min(pri[:2]))
    colmax=np.ceil(max(pri[:2]))
    #beam area on the scattering image
    B=data[rowmin:rowmax,colmin:colmax];
    print B.shape
    #row and column indices
    Ri=np.arange(rowmin,rowmax)
    Ci=np.arange(colmin,colmax)
    print len(Ri)
    print len(Ci)
    Ravg=np.mean(B,1)    #average over column index, will be a concave curve
    Cavg=np.mean(B,0)    #average over row index, will be a concave curve
    #find the maxima im both directions and their positions
    maxR=max(Ravg);    maxRpos=Ri[Ravg==maxR][0]
    maxC=max(Cavg);    maxCpos=Ci[Cavg==maxC][0]
    #cut off pixels which are smaller than threshold*peak_height
    Rmin=Ri[(Ravg>=(Ravg[0]+(maxR-Ravg[0])*threshold))&(Ri<maxRpos)][0]
    Rmax=Ri[(Ravg>=(Ravg[-1]+(maxR-Ravg[-1])*threshold))&(Ri>maxRpos)][-1]
    Cmin=Ci[(Cavg>=(Cavg[0]+(maxC-Cavg[0])*threshold))&(Ci<maxCpos)][0]
    Cmax=Ci[(Cavg>=(Cavg[-1]+(maxC-Cavg[-1])*threshold))&(Ci>maxCpos)][-1]
    d=data[Rmin:Rmax+1,Cmin:Cmax+1]
    x=np.arange(Rmin,Rmax+1)
    y=np.arange(Cmin,Cmax+1)
    bcx=np.sum(np.sum(d,1)*x)/np.sum(d)
    bcy=np.sum(np.sum(d,0)*y)/np.sum(d)
    return bcx,bcy

def findbeam_radialpeak(data, orig_initial, mask, rmin, rmax, maxiter=100,
                          drive_by='amplitude', extent=10, callback=None):
    """Find the beam by minimizing the width of a peak in the radial average.
    
    Inputs:
        data: scattering matrix
        orig_initial: first guess for the origin
        mask: mask matrix. Nonzero is non-masked.
        rmin,rmax: distance from the origin (in pixels) of the peak range.
        drive_by: 'hwhm' to minimize the hwhm of the peak or 'amplitude' to
            maximize the peak amplitude
        extent: approximate distance of the current and the real origin. Too
            high a value makes the fitting procedure unstable. Too low a value
            does not permit to move avay the current origin. 
        callback: callback function (expects no arguments)
    Outputs:
        the beam coordinates
    
    Notes:
        A Gaussian will be fitted.
    """
    orig_initial=np.array(orig_initial)
    mask=1-mask.astype(np.uint8)
    pix=np.arange(rmin*1.0,rmax*1.0,1)
    if drive_by.lower()=='hwhm':
        def targetfunc(orig,data,mask,rmin,rmax,orig_orig, callback):
            I=radintpix(data,None,orig[0]+orig_orig[0],orig[1]+orig_orig[1],mask,pix)[1]
            p=misc.findpeak(pix,I)[2]
            print orig[0]+orig_orig[0],orig[1]+orig_orig[1],p
            if callback is not None:
                callback()
            return abs(p)
    elif drive_by.lower()=='amplitude':
        def targetfunc(orig,data,mask,rmin,rmax,orig_orig, callback):
            I=radintpix(data,None,orig[0]+orig_orig[0],orig[1]+orig_orig[1],mask,pix)[1]
            fp=misc.findpeak(pix,I)
            p=-(fp[6]+fp[4])
            print orig[0]+orig_orig[0],orig[1]+orig_orig[1],p
            if callback is not None:
                callback()
            return p
    else:
        raise ValueError('Invalid argument for drive_by %s'%drive_by)
    orig1=scipy.optimize.fmin(targetfunc,np.array([extent,extent]),
                              args=(data,mask,1.0*rmin,1.0*rmax,orig_initial-extent,callback),
                              maxiter=maxiter,disp=0)
    return np.array(orig_initial)-extent+orig1
