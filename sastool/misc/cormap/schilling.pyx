#cython: boundscheck=False
#cython: cdivision=True
#cython: embedsignature=True
#cython: nonecheck=False

"""Utilities for the Schilling distribution (longest runs in an n-long coin-toss sequence)

See: M. F. Schilling: The Longest Run of Heads. Coll. Math. J 21(3) p196-207 (1990)
"""
cimport numpy as np
import cython
import numpy as np
from libc.float cimport DBL_MAX_EXP
from libc.stdlib cimport calloc, free

np.import_array()

cdef Py_ssize_t A_(Py_ssize_t n, Py_ssize_t x):
    """Calculate A_n(x) as per Schilling's original paper"""
    cdef Py_ssize_t j
    cdef Py_ssize_t val = 0
    if n <= x:
        return 2 ** n
    else:
        for j in range(0, x + 1):
            val += A_(n - 1 - j, x)
        return val

def A(n, x):
    """Python interface for A_(n, x)"""
    return A_(n, x)

def Amatrix(Py_ssize_t N):
    """Calculate an NxN matrix of the Schilling distribution

    The elements A[n, x] are the number of possible outcomes of an n-sequence of
    independent coin-tosses where the maximum length of consecutive heads is
    _not_larger_than x."""
    cdef np.ndarray[np.uint64_t, ndim=2] result
    cdef Py_ssize_t n, x
    result = np.empty((N, N), np.uint64)
    for x in range(N):
        for n in range(0, x + 1):
            result[n, x] = (2 ** n)
        for n in range(x + 1, N):
            result[n, x] = result[n - 1 - x:n, x].sum()
    return result

def amatrix(Py_ssize_t N):
    """Calculate an NxN matrix of the Schilling distribution

    The elements a[n, x] are the number of possible outcomes of an n-sequence of
    independent coin-tosses where the maximum length of consecutive heads is
    _exactly_ x. Thus a[n, x] = A[n, x] - A[n, x-1]"""
    cdef np.ndarray[np.uint64_t, ndim=2] result
    cdef Py_ssize_t n, x
    cdef Py_ssize_t val
    result = np.zeros((N, N), np.uint64)
    result[:, 0] = 1  # a_n(x=0) = 1
    for n in range(N):
        result[n, n] = 1  #a_n(x=n) = 1
        # a_n(x>n) = 0
        for x in range(1, n):
            result[n, x] = result[n - 1 - x:n, x].sum() + result[n - 1 - x, :x].sum()
    return result

def pmatrix(Py_ssize_t N):
    """Calculate an NxN matrix of the Schilling distribution.

    The elements p[n, x] of the resulting matrix are the probabilities that
    the length of the longest head-run in a sequence of n independent tosses
    of a fair coin is exactly x.

    It holds that p[n, x] = a[n, x] / 2 ** n = (A[n, x] - A[n, x-1]) / 2 ** n

    Note that the probability that the length of the longest run
    (no matter if head or tail) in a sequence of n independent
    tosses of a fair coin is _exactly_ x is p[n-1, x-1].
    """
    cdef np.ndarray[np.double_t, ndim=2] result
    cdef Py_ssize_t n, x, j
    cdef double val
    result = np.zeros((N, N), np.double)
    for n in range(N):
        result[n, 0] = 2.0 ** (-n)
        result[n, n] = 2.0 ** (-n)  #p_n(x=n) = 1/2**n
        # p_n(x>n) = 0
        for x in range(1, n):
            val = 0
            for j in range(n - 1 - x, n):
                val += 2.0 ** (j - n) * result[j, x]
            for j in range(0, x):
                val += 2.0 ** (-x - 1) * result[n - 1 - x, j]
            result[n, x] = val
    return result

@cython.cdivision
def cormap_pval(Py_ssize_t n, Py_ssize_t x):
    """Calculate the p-value for the cormap test, i.e. the probability that the
    longest run of consecutive heads or tails is not shorter than x.
    """
    cdef double half_pow_x = 2.0 ** (-x)
    cdef double *P = NULL
    cdef double result
    cdef Py_ssize_t i_x = 0, i = 0, im1_x = 0
    if x <= 1:
        return 1.0
    elif x > n:
        return 0.0
    elif x > DBL_MAX_EXP:
        return 0.0
    else:
        P = <double *> calloc(<size_t> DBL_MAX_EXP, sizeof(double))
        P[0] = 2 * half_pow_x
        for i in range(x + 1, n + 1):
            im1_x = i_x
            i_x = i % x
            P[i_x] = P[im1_x] + half_pow_x * (1 - P[i_x])
        result = P[i_x]
        free(P)
        return result

def longest_edge(np.ndarray[np.double_t, ndim=1] diag):
    """Calculate the longest edge length in a correlation map, based
    on the diagonal of the cormap matrix, supplied in `diag`.
    """
    cdef:
        Py_ssize_t l = 1, i = 0, lmax = 0
        double sgn = diag[0]
    for i in range(1, len(diag)):
        if sgn * diag[i] >= 0:
            l += 1
        else:
            if l > lmax:
                lmax = l
            l = 1
            sgn = diag[i]
    if l > lmax:
        lmax = l
    return lmax
