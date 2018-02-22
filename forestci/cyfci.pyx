# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
from numpy cimport ndarray
cimport numpy
import numpy
cimport cython
from cython.parallel cimport prange
cimport scipy.linalg.cython_blas as blas

def _cycore_computation(
        ndarray[numpy.float64_t, ndim=2] inbag not None,
        ndarray[numpy.float64_t, ndim=2] pred_centered not None):
    """
    Helper function performs core computation using cython
    and avoids storing intermediate matrices in-memory.

    Parameters
    ----------
    inbag: ndarray
        The inbag matrix that fit the data. If set to `None` (default) it
        will be inferred from the forest. However, this only works for trees
        for which bootstrapping was set to `True`. That is, if sampling was
        done with replacement. Otherwise, users need to provide their own
        inbag matrix.

    pred_centered : ndarray
        Centered predictions that are an intermediate result in the
        computation.
    """
    result = numpy.zeros(pred_centered.shape[0], dtype=numpy.float64)
    inbag = inbag-1
    _matmul_colsum(inbag, pred_centered, result)
    return result

cdef _matmul_colsum(double[:,:] a, double[:,:] b, double[:] c):
    """
    Matrix multiply `a` and `b` and then sum over columns without
    storing the intermediate matrix (a dot b) in memory.
    Result is stored in `c`.
    Equivalent to `np.sum(np.dot(a,b), axis=0)`

    Parameters
    ----------
    a: ndarray
       `(n,p)` 2d input array

    b: ndarray
        `(p,m)` 2d input array

    c: ndarray
        `(m)` 1d output array (data overwritten with result)

    Returns
    -------
    None
    """
    cdef int i, j
    cdef int n=a.shape[0], m=b.shape[0], B=a.shape[1]
    cdef int ONE=1
    cdef double x=0.0;
    for i in prange(m, nogil=True, num_threads=32, schedule='static'):
        x=0.0
        for j in range(n):
            x = x + blas.ddot(&B, &a[j][0], &ONE, &b[i][0], &ONE) ** 2
        c[i] = x / B**2
