#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3, embedsignature=True

import numpy as np

cimport numpy as np
cimport openmp

from cython.parallel import prange, parallel, threadid
from libc.stdlib cimport abort, malloc, free
from libc.math cimport isnan, sqrt, acos, fabs, exp, log
from .utils import get_max_threads

ctypedef np.int16_t int16_t
ctypedef np.uint16_t uint16_t
ctypedef np.float32_t floating
ctypedef np.float32_t float32_t
ctypedef np.float64_t float64_t
ctypedef np.npy_bool bool

MAXITERS = 10000
EPS = 1e-4


def __emad(const floating [:, :, :, :] X, const floating [:, :, :] gm,
           floating [:,:,:] result, const int num_threads=1):
    cdef int number_of_threads = num_threads
    cdef int m = X.shape[0]
    cdef int q = X.shape[1]
    cdef int p = X.shape[2]
    cdef int n = X.shape[3]
    cdef int j, t, row, col
    cdef float64_t total, value

    with nogil, parallel(num_threads=number_of_threads):
        for row in prange(m, schedule='dynamic'):
            for col in range(q):
                for t in range(n):
                    # euclidean distance
                    total = 0.
                    for j in range(p):
                        value = X[row, col, j, t] - gm[row, col, j]
                        total = total + value*value
                    result[row, col, t] = sqrt(total)


def __emad_uint16(const uint16_t [:, :, :, :] X, const floating [:, :, :] gm,
                  floating [:,:,:] result, const int num_threads=1,
                  const uint16_t nodata=0, const floating scale=1e-4,
                  const floating offset = 0.):

    cdef int number_of_threads = num_threads
    cdef int m = X.shape[0]
    cdef int q = X.shape[1]
    cdef int p = X.shape[2]
    cdef int n = X.shape[3]
    cdef int j, t, row, col
    cdef float64_t total, value
    cdef uint16_t int_value

    with nogil, parallel(num_threads=number_of_threads):
        for row in prange(m, schedule='dynamic'):
            for col in range(q):
                for t in range(n):
                    # euclidean distance
                    total = 0.
                    for j in range(p):
                        int_value = X[row, col, j, t]
                        if int_value != nodata:
                            value = int_value * scale + offset - gm[row, col, j]
                            total = total + value*value
                    result[row, col, t] = sqrt(total)


def __smad(const floating[:, :, :, :] X, const floating[:, :, :] gm,
           floating[:,:,:] result, const int num_threads=1):
    """ """
    cdef int number_of_threads = num_threads
    cdef int m = X.shape[0]
    cdef int q = X.shape[1]
    cdef int p = X.shape[2]
    cdef int n = X.shape[3]
    cdef int j, t, row, col
    cdef float64_t numer, norma, normb, normb_sqrt, value

    with nogil, parallel(num_threads=number_of_threads):
        for row in prange(m, schedule='dynamic'):
            for col in range(q):
                normb = 0.
                for j in range(p):
                    normb = normb + gm[row, col, j]*gm[row, col, j]
                normb_sqrt = sqrt(normb)

                for t in range(n):
                    numer = 0.
                    norma = 0.

                    for j in range(p):
                        value = X[row, col, j, t]*gm[row, col, j]
                        numer = numer + value
                        norma = norma + X[row, col, j, t]*X[row, col, j, t]

                    result[row, col, t] = 1. - numer/(sqrt(norma)*normb_sqrt)


def __smad_uint16(const uint16_t [:, :, :, :] X, const floating[:, :, :] gm,
                  floating[:,:,:] result, const int num_threads=1,
                  const uint16_t nodata=0, const floating scale=1e-4,
                  const floating offset=0.):
    """ """
    cdef int number_of_threads = num_threads
    cdef int m = X.shape[0]
    cdef int q = X.shape[1]
    cdef int p = X.shape[2]
    cdef int n = X.shape[3]
    cdef int j, t, row, col
    cdef float64_t numer, norma, normb, normb_sqrt, value
    cdef float64_t scaled

    with nogil, parallel(num_threads=number_of_threads):
        for row in prange(m, schedule='dynamic'):
            for col in range(q):
                normb = 0.
                for j in range(p):
                    normb = normb + gm[row, col, j]*gm[row, col, j]
                normb_sqrt = sqrt(normb)

                for t in range(n):
                    numer = 0.
                    norma = 0.

                    for j in range(p):
                        scaled = X[row, col, j, t] * scale + offset
                        value = scaled * gm[row, col, j]
                        numer = numer + value
                        norma = norma + scaled*scaled

                    result[row, col, t] = 1. - numer/(sqrt(norma)*normb_sqrt)


def __bcmad(const floating[:, :, :, :] X, const floating[:, :, :] gm,
            floating[:,:,:] result, int num_threads=1):
    """ """
    cdef int number_of_threads = num_threads
    cdef int m = X.shape[0]
    cdef int q = X.shape[1]
    cdef int p = X.shape[2]
    cdef int n = X.shape[3]
    cdef int j, t, row, col
    cdef float64_t numer, denom, value

    with nogil, parallel(num_threads=number_of_threads):
        for row in prange(m, schedule='dynamic'):
            for col in range(q):
                for t in range(n):

                    numer = 0.
                    denom = 0.

                    for j in range(p):
                        numer = numer + fabs(X[row, col, j, t] - gm[row, col, j])
                        denom = denom + fabs(X[row, col, j, t] + gm[row, col, j])

                    result[row, col, t] = numer / denom


def __bcmad_uint16(const uint16_t [:, :, :, :] X, const floating[:, :, :] gm,
                  floating[:,:,:] result, const int num_threads=1,
                  const uint16_t nodata=0, const floating scale=1e-4,
                  const floating offset=0.):
    """ """
    cdef int number_of_threads = num_threads
    cdef int m = X.shape[0]
    cdef int q = X.shape[1]
    cdef int p = X.shape[2]
    cdef int n = X.shape[3]
    cdef int j, t, row, col
    cdef float64_t numer, denom, value
    cdef float64_t scaled

    with nogil, parallel(num_threads=number_of_threads):
        for row in prange(m, schedule='dynamic'):
            for col in range(q):
                for t in range(n):

                    numer = 0.
                    denom = 0.

                    for j in range(p):
                        scaled = X[row, col, j, t] * scale + offset
                        numer = numer + fabs(scaled - gm[row, col, j])
                        denom = denom + fabs(scaled + gm[row, col, j])

                    result[row, col, t] = numer / denom


def emad(X, np.ndarray[floating, ndim=3] gm,
         num_threads=None, nocheck=False, nodata=None, **kw):
    """
    Generate a Euclidean geometric median absolute deviation pixel
    composite mosaic by reducing along the last axis.

    Parameters
    ----------
    X : array_like of dtype float32 or float64
        The array has dimensions (m, q, p, n).
    gm : array_like of dtype float32 or float64
	The geometric median of the stack of dimension (m, q, p)
    num_threads : int
        The number of processing threads to use for the computation.

    Returns
    -------
    m : ndarray
        The array has dimensions (m, q, p)
    """
    cdef int m = X.shape[0]
    cdef int q = X.shape[1]
    cdef int p = X.shape[2]
    cdef int n = X.shape[3]

    if num_threads is None:
        num_threads = get_max_threads()

    old = np.seterr(all='ignore')

    dtype = np.float32

    result = np.empty((m, q, n), dtype=dtype)

    if nodata is None:
        nodata = 0

    if X.dtype == np.uint16:
        __emad_uint16(X, gm, result, num_threads=num_threads, nodata=nodata, **kw)
    else:
        __emad(X, gm, result, num_threads=num_threads)

    np.seterr(**old)

    if not nocheck:
        return np.nanmedian(result, axis=2)
    else:
        return np.median(result, axis=2)


def smad(X, np.ndarray[floating, ndim=3] gm,
         num_threads=None, nocheck=False, nodata=None, **kw):
    """
    Generate a spectral geometric median absolute deviation pixel
    composite mosaic by reducing along the last axis.

    Distance is calculated using spectral distance, aka. cosine
    distance. The distance is invariant under parallel shifts.

    Parameters
    ----------
    X : array_like of dtype float32 or float64
        The array has dimensions (m, q, p, n).
    gm : array_like of dtype float32 or float64
	The geometric median of the stack of dimension (m, q, p)
    num_threads : int
        The number of processing threads to use for the computation.

    Returns
    -------
    m : ndarray
        The array has dimensions (m, q, p)
    """
    cdef int m = X.shape[0]
    cdef int q = X.shape[1]
    cdef int p = X.shape[2]
    cdef int n = X.shape[3]

    if num_threads is None:
        num_threads = get_max_threads()

    old = np.seterr(all='ignore')

    dtype = np.float32

    result = np.empty((m, q, n), dtype=dtype)

    if nodata is None:
        nodata = 0

    if X.dtype == np.uint16:
        __smad_uint16(X, gm, result, num_threads=num_threads, nodata=nodata, **kw)
    else:
        __smad(X, gm, result, num_threads=num_threads)

    np.seterr(**old)

    if not nocheck:
        return np.nanmedian(result, axis=2)
    else:
        return np.median(result, axis=2)


def bcmad(X, np.ndarray[floating, ndim=3] gm,
          num_threads=None, nocheck=False, nodata=None, **kw):
    """
    Generate a bray-curtis geometric median absolute deviation pixel
    composite mosaic by reducing along the last axis.

    Parameters
    ----------
    X : array_like of dtype float32 or float64
        The array has dimensions (m, q, p, n).
    gm : array_like of dtype float32 or float64
	The geometric median of the stack of dimension (m, q, p)
    num_threads : int
        The number of processing threads to use for the computation.

    Returns
    -------
    m : ndarray
        The array has dimensions (m, q, p)
    """
    cdef int m = X.shape[0]
    cdef int q = X.shape[1]
    cdef int p = X.shape[2]
    cdef int n = X.shape[3]

    if num_threads is None:
        num_threads = get_max_threads()

    old = np.seterr(all='ignore')

    dtype = np.float32

    result = np.empty((m, q, n), dtype=dtype)

    if nodata is None:
        nodata = 0

    if X.dtype == np.uint16:
        __bcmad_uint16(X, gm, result, num_threads=num_threads, nodata=nodata, **kw)
    else:
        __bcmad(X, gm, result, num_threads=num_threads)

    np.seterr(**old)

    if not nocheck:
        return np.nanmedian(result, axis=2)
    else:
        return np.median(result, axis=2)
