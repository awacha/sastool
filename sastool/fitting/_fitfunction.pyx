# pylint: disable-msg-cat=WCREFI
#cython: boundscheck=False
#cython: embedsignature=True

from libc.stdlib cimport *
from libc.math cimport *
import numpy as np
cimport numpy as np

ctypedef struct Coordtype:
    double x
    double y
    double z

cdef double HC=12398.419 #Planck's constant times speed of light, in eV*Angstrom units

cdef inline double randn():
    """Standard normal distribution
    """
    cdef double x
    cdef double y
    cdef int notready
    notready=1
    while(notready):
        x=-log(rand()/<double>RAND_MAX)
        y=exp(-0.5*(x-1)*(x-1))
        if (rand()/<double>RAND_MAX <y):
            notready=0
            if (rand()/<double>RAND_MAX<0.5):
                x=-x
    return x


cdef inline float bessj1(double x):
    """Returns the Bessel function J1 (x) for any real x.
    
    Taken from Numerical Recipes
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

cdef inline double fcylinder3d(double qx, double qy, double qz, double L, double R):
    """Scattering factor of a cylinder
    
    Inputs:
        qx, qy, qz: x,y and z component of the q vector
        L: length of the cylinder
        R: radius of the cylinder
    
    Output:
        the value of the scattering factor. The axis is assumed to lie in z.
        It is normalized to F(q=0)=V_cylinder
    """
    cdef double qax # axial component of the q vector
    cdef double qrad # radial component of q (perpendicular to the axial
    cdef double h
    cdef double term1,term2
    qax=fabs(qz)
    qrad=sqrt(qx*qx+qy*qy)
    h=L*0.5
    if (qax*h>0):
        term1=h*sin(qax*h)/(qax*h)
    else:
        term1=h
    if (R*qrad>0):
        term2=R/qrad*bessj1(R*qrad)
    else:
        term2=R*R*0.5 # bessj1(x)/x -> 0.5 while x -> 0
    return 4*M_PI*term1*term2
    

cdef inline Coordtype unidirC(): #uniform distribution of points on the surface of a sphere
    cdef Coordtype ret
    cdef double phi
    cdef double rho
    phi=rand()/<double>RAND_MAX*2*M_PI
    ret.z=rand()/<double>RAND_MAX*2-1
    rho=sqrt(1-(ret.z)**2)
    ret.x=rho*cos(phi)
    ret.y=rho*sin(phi)
    return ret


       
    
cdef inline double Fcylinder_scalar(double q, double R, double L, Py_ssize_t Nstep):
    cdef double r
    cdef Py_ssize_t i
    cdef double lim1,lim2,lim
    cdef Py_ssize_t ok
    
    lim1=(bessj1(q*R)/(q*R)*0.5)**2
    lim2=(0.5*sin(q*L*0.5)/(q*L))**2
    lim=max(lim1,lim2)
    ok=0
    for i from 0<=i<Nstep:
        x=(1.0*rand())/RAND_MAX
        y=(lim*rand())/RAND_MAX
        ok+=(y<(bessj1(q*R*sqrt(1-x**2))/(q*R*sqrt(1-x**2))*sin(q*L*x*0.5)/(q*L*x))**2)
    return ok/(1.0*Nstep)        

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


def Fcylinder(np.ndarray[np.double_t, ndim=1] q, double R, double L, Py_ssize_t Nstep):
    cdef double factor
    cdef np.ndarray[np.double_t, ndim=1] F2
    cdef Py_ssize_t i
    
    F2=np.zeros_like(q)
    factor=16*(R*R*L*M_PI)**2
    for 0<=i<len(q):
        F2[i]=factor*Fcylinder_scalar(q[i],R,L,Nstep)
    return F2
