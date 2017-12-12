# cython: boundscheck=False
# cython: cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport round
from cython.parallel import prange
cimport
numpy as np
import numpy as np
from cython.parallel import prange
from libc.math cimport

round

cdef double* _pixelintegrate(double[:,:] matrix,
                  unsigned char[:,:] mask, Py_ssize_t Nrow, Py_ssize_t Ncol,
                  double origrow, double origcol, Py_ssize_t rmin, Py_ssize_t rmax) nogil:
    cdef:
        double *I
        double *A
        Py_ssize_t irow, icol,  ibin
        double r

    I=<double*>calloc(sizeof(double), rmax-rmin+1)
    A=<double*>calloc(sizeof(double),  rmax-rmin+1)

    for irow in range(Nrow):
        for icol in prange(Ncol, nogil=False):
            if not mask[irow, icol]:
                continue
            r = round(((irow-origrow)**2+(icol-origcol)**2)**0.5)
            if r<rmin or r>rmax:
                continue
            ibin = <Py_ssize_t>(r-rmin)
            I[ibin]+=matrix[irow,icol]
            A[ibin]+=1

    for ibin in range(rmax-rmin+1):
        if A[ibin]:
            I[ibin]/=A[ibin]
    free(A)
    return I

def pixelintegrate(np.ndarray[np.double_t, ndim=2] matrix not None,
                  np.ndarray[np.uint8_t, ndim=2] mask not None,
                  double origrow, double origcol, Py_ssize_t rmin, Py_ssize_t rmax):
    cdef:
        double *I
        np.ndarray[np.double_t, ndim=1] out
        Py_ssize_t i
    I=_pixelintegrate(matrix, mask, matrix.shape[0], matrix.shape[1], origrow, origcol, rmin, rmax)
    out=np.empty(rmax-rmin+1,np.double)
    for i in range(rmax-rmin+1):
        out[i]=I[i]
    free(I)
    return out
