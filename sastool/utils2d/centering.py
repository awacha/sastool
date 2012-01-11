import numpy as np
from integrate import radintpix, azimintpix
def findbeam_gravity(data,mask):
    """Find beam center with the "gravity" method

    Inputs:
        data: scattering image
        mask: mask matrix

    Output:
        a vector of length 2 with the x (row) and y (column) coordinates
         of the origin, starting from 1
    """
    print "Finding beam (gravity), please be patient..."
    # for each row and column find the center of gravity
    data1=data.copy() # take a copy, because elements will be tampered with
    data1[mask==0]=0 # set masked elements to zero

    #pylab.imshow(data1) # show the matrix
    #pylab.gcf().show() #
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
    #y1=y[goodx]
    nix=nix[goodx]
    spamx=spamx[goodx]

    # now do the same for the column direction.
    niy=np.dot(data1.T,y).flatten()
    spamy=np.dot(data1.T,oney).flatten()
    goody=((niy!=0) & (spamy!=0))
#    x1=x[goody]
    niy=niy[goody]
    spamy=spamy[goody]
    # column coordinate of the center in each row will be contained in
    # ycent, the row coordinate of the center in each column will be
    # in xcent.
    ycent=nix/spamx
    xcent=niy/spamy
    #pylab.figure()
    #pylab.plot(x1,xcent,'.',label='xcent')
    #pylab.plot(y1,ycent,'.',label='ycent')
    #pylab.gcf().show()
    # return the mean values as the centers.
    return [xcent.mean()+1,ycent.mean()+1]

def findbeam_slices(data,orig_initial,mask=None,maxiter=0):
    """Find beam center with the "slices" method
    
    Inputs:
        data: scattering matrix
        orig_initial: estimated value for x (row) and y (column)
            coordinates of the beam center, starting from 1.
        mask: mask matrix. If None, nothing will be masked. Otherwise it
            should be of the same size as data. Nonzero means non-masked.
        maxiter: maximum number of iterations for scipy.optimize.fmin
    Output:
        a vector of length 2 with the x (row) and y (column) coordinates
         of the origin.
    """
    print "Finding beam (slices), please be patient..."
    orig=np.array(orig_initial)
    if mask is None:
        mask=np.ones(data.shape)
    def targetfunc(orig,data,mask):
        #integrate four sectors
        print "integrating... (for finding beam)"
        print "orig (before integration):",orig[0],orig[1]
        c1,nc1=radintpix(data,orig,mask,35,20)
        c2,nc2=radintpix(data,orig,mask,35+90,20)
        c3,nc3=radintpix(data,orig,mask,35+180,20)
        c4,nc4=radintpix(data,orig,mask,35+270,20)
        # the common length is the lowest of the lengths
        last=min(len(c1),len(c2),len(c3),len(c4))
        # first will be the first common point: the largest of the first
        # nonzero points of the integrated data
        first=np.array([pylab.find(nc1!=0).min(),
                           pylab.find(nc2!=0).min(),
                           pylab.find(nc3!=0).min(),
                           pylab.find(nc4!=0).min()]).max()
        ret= np.array(((c1[first:last]-c3[first:last])**2+(c2[first:last]-c4[first:last])**2)/(last-first))
        print "orig (after integration):",orig[0],orig[1]
        print "last-first:",last-first
        print "sum(ret):",ret.sum()
        return ret
    orig=scipy.optimize.leastsq(targetfunc,np.array(orig_initial),args=(data,1-mask),maxfev=maxiter,epsfcn=0.0001)
    return orig[0]

def findbeam_azimuthal(data,orig_initial,mask=None,maxiter=100,Ntheta=50,dmin=0,dmax=np.inf):
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
    Output:
        a vector of length 2 with the x and y coordinates of the origin,
            starting from 1
    """
    print "Finding beam (azimuthal), please be patient..."
    if mask is None:
        mask=np.ones(data.shape)
    def targetfunc(orig,data,mask):
        def sinfun(p,x,y):
            return (y-np.sin(x+p[1])*p[0]-p[2])/np.sqrt(len(x))
        t,I,a=azimintpix(data,None,orig,mask.astype('uint8'),Ntheta,dmin,dmax)
        if len(a)>(a>0).sum():
            raise ValueError,'findbeam_azimuthal: non-complete azimuthal average, please consider changing dmin, dmax and/or orig_initial!'
        p=((I.max()-I.min())/2.0,t[I==I.max()][0],I.mean())
        p=scipy.optimize.leastsq(sinfun,p,(t,I))[0]
        #print "findbeam_azimuthal: orig=",orig,"amplitude=",abs(p[0])
        return abs(p[0])
    orig1=scipy.optimize.fmin(targetfunc,np.array(orig_initial),args=(data,1-mask),maxiter=maxiter)
    return orig1

def findbeam_azimuthal_fold(data,orig_initial,mask=None,maxiter=100,Ntheta=50,dmin=0,dmax=np.inf):
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
    Output:
        a vector of length 2 with the x and y coordinates of the origin,
            starting from 1
    """
    print "Finding beam (azimuthal_fold), please be patient..."
    if Ntheta%2:
        raise ValueError('Ntheta should be even in function findbeam_azimuthal_fold()!')
    if mask is None:
        mask=np.ones(data.shape)
    def targetfunc(orig,data,mask):
        t,I,a=azimintpix(data,None,orig,mask.astype('uint8'),Ntheta,dmin,dmax)
        return np.sum((I[:Ntheta/2]-I[Ntheta/2:])**2)/Ntheta
    orig1=scipy.optimize.fmin(targetfunc,np.array(orig_initial),args=(data,1-mask),maxiter=maxiter)
    return orig1

def findbeam_semitransparent(data,pri):
    """Find beam with 2D weighting of semitransparent beamstop area

    Inputs:
        data: scattering matrix
        pri: list of four: [xmin,xmax,ymin,ymax] for the borders of the beam
            area under the semitransparent beamstop. X corresponds to the column
            index (ie. A[Y,X] is the element of A from the Xth column and the 
            Yth row)

    Outputs: bcx,bcy
        the x and y coordinates of the primary beam
    """
    threshold=0.05
    print "Finding beam (semitransparent), please be patient..."
#    xmin=min([pri[0],pri[1]])
#    ymin=min([pri[2],pri[3]])
#    xmax=max([pri[0],pri[1]])
#    ymax=max([pri[2],pri[3]])
    C,R=np.meshgrid(np.arange(data.shape[1]),
                       np.arange(data.shape[0]))
    B=data[pri[2]:pri[3],pri[0]:pri[1]];
    Ri=range(pri[2],pri[3])
    Ci=range(pri[0],pri[1])
    Ravg=np.sum(B,1)/len(Ri)
    Cavg=np.sum(B,0)/len(Ci)
    maxR=max(Ravg)
    maxRpos=[i for i,avg in zip(Ri,Ravg) if avg==maxR][0]
    Rmin=[i for i,avg in zip(Ri,Ravg) if (avg<Ravg[0]+(maxR-Ravg[0])*threshold) and (i<maxRpos)][-1]    
    Rmax=[i for i,avg in zip(Ri,Ravg) if (avg<Ravg[-1]+(maxR-Ravg[-1])*threshold) and (i>maxRpos)][0]
    maxC=max(Cavg)
    maxCpos=[i for i,avg in zip(Ci,Cavg) if avg==maxC][0]
    Cmin=[i for i,avg in zip(Ci,Cavg) if (avg<Cavg[0]+(maxC-Cavg[0])*threshold) and (i<maxCpos)][-1]    
    Cmax=[i for i,avg in zip(Ci,Cavg) if (avg<Cavg[-1]+(maxC-Cavg[-1])*threshold) and (i>maxCpos)][0]
    d=data[Rmin:Rmax,Cmin:Cmax]
    x=np.arange(Rmin,Rmax)
    y=np.arange(Cmin,Cmax)
    bcx=np.sum(np.sum(d,1)*x)/np.sum(d)+1
    bcy=np.sum(np.sum(d,0)*y)/np.sum(d)+1
    return bcx,bcy
