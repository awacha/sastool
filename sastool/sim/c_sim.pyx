# pylint: disable-msg-cat=WCREFI
#cython: boundscheck=False
#cython: embedsignature=True
cimport numpy as np
import numpy as np
from libc.math cimport *
from libc.stdlib cimport *

np.import_array()

#cdef extern from "math.h":
#    int isfinite(double)
#    double INFINITY
#    double fabs(double)

#-------------------- Auxiliary functions -------------------------------------
#
#  Scattering form factors of cylinders and spheres

cdef inline float bessj1(double x):
    """Returns the Bessel function J1 (x) for any real x.
    
    Lovingly ripped off from Numerical Recipes
    """
    cdef float ax,z
    cdef double xx,y,ans,ans1,ans2
    ax=fabs(x)
    if (ax < 8.0):
        y=x*x;
        ans1=x*(72362614232.0+y*(-7895059235.0+y*(242396853.1
            +y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))));
        ans2=144725228442.0+y*(2300535178.0+y*(18583304.74
            +y*(99447.43394+y*(376.9991397+y*1.0))));
        ans=ans1/ans2;
    else:
        z=8.0/ax;
        y=z*z;
        xx=ax-2.356194491;
        ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
            +y*(0.2457520174e-5+y*(-0.240337019e-6))));
        ans2=0.04687499995+y*(-0.2002690873e-3
            +y*(0.8449199096e-5+y*(-0.88228987e-6
            +y*0.105787412e-6)));
        ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
        if (x < 0.0):
            ans = -ans;
    return ans

cdef inline double fcylinder(double qx, double qy, double qz,
                             double vx, double vy, double vz, double L, double R):
    """Scattering factor of a cylinder with unit scattering length density
    
    Inputs:
        qx, qy, qz: x,y and z component of the q vector
        vx, vy, vz: components of the director of the cylinder
        L: length of the cylinder
        R: radius of the cylinder
    
    Output:
        the value of the scattering factor ( F(0) = volume of the cylinder)
    """
    cdef double qax # axial component of the q vector
    cdef double qrad # radial component of q (perpendicular to the axial
    cdef double h
    cdef double term1,term2
    qax=fabs(qx*vx+qy*vy+qz*vz)
    qrad=sqrt(qx*qx+qy*qy+qz*qz-qax**2)
    h=L*0.5
    if (qax*h>0):
        term1=h*sin(qax*h)/(qax*h)
    else:
        term1=h
    if (R*qrad>0):
        term2=R/qrad*bessj1(R*qrad)
    else:
        term2=R*0.5
    return 4*M_PI*term1*term2
    
cdef inline double fsphere(double q, double R):
    """Scattering factor of a sphere
    
    Inputs:
        q: q value(s) (scalar or an array of arbitrary size and shape)
        R: radius (scalar)
        
    Output:
        the values of the scattering factor in an array of the same shape as q
    """
    if q==0:
        return 4*M_PI*R**3/3
    else:
        return 4*M_PI/q**3*(sin(q*R)-q*R*cos(q*R))

cdef inline double sinxdivx(double x):
    if x==0:
        return 1
    else:
        return sin(x)/x

# ----------------- Public functions ------------------------------------


def SAStrans2D(double pixsizedivdist, double wavelength, Py_ssize_t N, 
               Py_ssize_t M, double centerx=INFINITY, double centery=INFINITY,
               np.ndarray[np.double_t, ndim=2] spheres=None,
               np.ndarray[np.double_t, ndim=2] cylinders=None):
    """Calculate scattering intensity in transmission setup
    
    Inputs:
        pixsizedivdist: the ratio of the pixel size and the sample-to-detector
            distance
        wavelength: monochromatic wavelength in the desired length units.
        N: number of rows of the detector (the row index is x)
        M: number of columns of the detector (the column index is y)
        centerx: row coordinate of the beam center (defaults to the middle row)
        centery: column coordiante of the beam center (defaults to the middle
            column)
        spheres: Parameters of spherical units. Each row in the array
            corresponds to a distinct sphere. The columns are laid out as
            follows:
            
                <x> <y> <z> <R> <breal> <bimag>
            
            where <x>, <y> and <z> are the coordinates of the center, <R> is the
            radius and <breal> and <bimag> are the scattering length densities.
        
        cylinders: This is for cylindrical units what is 'spheres' for spherical
            ones. The layout of each row is:
                <x> <y> <z> <vx> <vy> <vz> <R> <L> <breal> <bimag>
            
            where <x> <y> and <z> are the coordinates of the center of gravity,
            <R> is the radius, <L> is the full length of the cylinder, <vx> <vy>
            and <vz> is its director (vector parallel to its axis) and <breal>
            and <bimag> are the scattering length densities.
    Outputs:
        the scattering image: an N-by-M matrix.
    
    Notes:
        the detector flatness is taken into account.
    """
    cdef np.ndarray[np.double_t, ndim=2] Intensity
    cdef Py_ssize_t row,column,i
    cdef double Areal,Aimag
    cdef double qx,qy,qz,q,qR
    cdef double F,sqR,cqR
    if spheres is None:
        spheres=np.zeros((0,6),np.double)
    if cylinders is None:
        cylinders=np.zeros((0,9),np.double)
    if spheres.shape[1]<6:
        raise ValueError('sphere array should have at least 6 columns')
    if cylinders.shape[1]<9:
        raise ValueError('cylinder array should have at least 9 columns')
    if not isfinite(centerx):
        centerx=N/2.
    if not isfinite(centery):
        centery=N/2.0
    Intensity=np.zeros((N,M),np.double)
    for row from 0<=row<N:
        for column from 0<=column<M:
            qx=(row-centerx)*pixsizedivdist # x component of the scattered wave vector
            qy=(column-centery)*pixsizedivdist # y component of the scattered wave vector
            q=sqrt(qx**2+qy**2+1)
            qx=qx/q*2*M_PI/wavelength
            qy=qy/q*2*M_PI/wavelength
            qz=(1-1/q)*2*M_PI/wavelength
            q=sqrt(qx**2+qy**2+qz**2)
            Areal=0;
            Aimag=0;
            for i from 0<=i<spheres.shape[0]:
                qR=qx*spheres[i,0]+qy*spheres[i,1]+qz*spheres[i,2]
                F=fsphere(q,spheres[i,3])
                sqR=sin(qR);  cqR=cos(qR)
                Areal+=F*cqR*spheres[i,4]-F*sqR*spheres[i,5]
                Aimag+=F*cqR*spheres[i,5]+F*sqR*spheres[i,4]
            for i from 0<=i<cylinders.shape[0]:
                qR=qx*cylinders[i,0]+qy*cylinders[i,1]+qz*cylinders[i,2]
                F=fcylinder(qx,qy,qz,cylinders[i,3],cylinders[i,4],cylinders[i,5],
                            cylinders[i,7],cylinders[i,6])
                sqR=sin(qR);  cqR=cos(qR)
                Areal+=F*cqR*cylinders[i,8]-F*sqR*cylinders[i,9]
                Aimag+=F*cqR*cylinders[i,9]+F*sqR*cylinders[i,8]
            Intensity[row,column]=Areal*Areal+Aimag*Aimag
    return Intensity

def SAS1D(np.ndarray[np.double_t, ndim=2] spheres, 
          np.ndarray[np.double_t, ndim=1] q):
    """Calculate spherically averaged small-angle scattering intensity of an
    ensemble of spheres.
    
    Inputs:
        spheres: array of sphere data. Each row corresponds to a single sphere.
        Rows are laid out as:
            <x> <y> <z> <R> <breal> <bimag>
        
        where <x>, <y> and <z> are the coordinates of the center, <R> is the
        radius and <breal> and <bimag> are the real and imaginary part of the
        scattering length densities.
    
    Outputs:
        the result of the Debye-formula, i.e.
        
        I(q)=sum_j sum_k F_j(q)*F_k(q)*sin(q*R_jk)/(q*R_jk)
    """
    cdef np.ndarray[np.double_t, ndim=1] Intensity
    cdef Py_ssize_t i,j,iq
    cdef double I,Fi,Fj,qR
    
    if spheres.shape[1]<6:
        raise ValueError('sphere array should have at least 6 columns')
    Intensity=np.zeros(len(q),np.double)
    for iq from 0<=iq<len(q):
        I=0
        for i from 0<=i<spheres.shape[0]:
            Fi=fsphere(q[iq],spheres[i,3])
            I+=Fi*Fi*(spheres[i,4]**2+spheres[i,5]**2)
            for j from i<=j<spheres.shape[0]:
                Fj=fsphere(q[iq],spheres[j,3])
                qR=q*sqrt((spheres[i,0]-spheres[j,0])**2+
                          (spheres[i,1]-spheres[j,1])**2+
                          (spheres[i,2]-spheres[j,2])**2)
                I+=2*Fi*Fj*(spheres[i,4]*spheres[j,4]+spheres[i,5]*spheres[j,5])*sinxdivx(qR)
        Intensity[iq]=I
    return Intensity
