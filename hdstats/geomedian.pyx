# cython: cdivision=True, boundscheck=False, nonecheck=False, wraparound=False, language_level=3

import numpy as np

cimport numpy as np

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

#cdef float32_t dot(const float32_t[:] x, const float32_t[:] y) nogil:
#    cdef size_t n = x.shape[0]
#    cdef size_t i = 0
#    cdef float64_t result = 0.
#    for i in range(n):
#        result += x[i] * y[i]
#    return <float32_t>result

cdef float32_t sum(const float32_t[:] x) nogil:
    cdef size_t n = x.shape[0]
    cdef float64_t total = 0.
    for i in range(n):
        total += x[i]
    return <float32_t>total

cdef float32_t nansum(const float32_t[:] x) nogil:
    cdef size_t n = x.shape[0]
    cdef float64_t total = 0.
    for i in range(n):
        if not isnan(x[i]):
            total += x[i]
    return <float32_t>total

#cdef float32_t dist_naneuclidean(const float32_t[:] x, const float32_t[:] y) nogil:
#    cdef size_t n = x.shape[0]
#    cdef float64_t d = 0.
#    cdef float64_t tmp
#    for i in range(n):
#        if (not isnan(x[i])) and (not isnan(y[i])):
#            tmp = x[i] - y[i]
#            d += tmp * tmp
#    return <float32_t>sqrt(d)

cdef float32_t dist_euclidean(const float32_t[:] x, const float32_t[:] y) nogil:
    cdef size_t n = x.shape[0]
    cdef float64_t d = 0.
    cdef float64_t tmp
    for i in range(n):
        tmp = x[i] - y[i]
        d += tmp * tmp
    return <float32_t>sqrt(d)

cdef float32_t norm_euclidean(const float32_t[:] x) nogil:
    cdef size_t n = x.shape[0]
    cdef float64_t d = 0.
    for i in range(n):
        d += x[i] * x[i]
    return <float32_t>sqrt(d)

cdef geomedian_axis_zero(const float32_t [:, :] X, float32_t [:] result, float32_t eps=1e-7, size_t maxiters=500):
    cdef size_t p = X.shape[0]
    cdef size_t n = X.shape[1]

    cdef float32_t[:] y = np.mean(X, axis=0)

    if p == 0:
        return y

    cdef float32_t[:] D = np.empty(p, dtype=np.float32)
    cdef float32_t[:] Dinv = np.empty(p, dtype=np.float32)
    cdef float32_t[:] W = np.empty(p, dtype=np.float32)
    cdef float32_t[:] T = np.empty(n, dtype=np.float32)
    cdef float32_t[:] y1 = np.empty(n, dtype=np.float32)
    cdef float32_t[:] R = np.empty(n, dtype=np.float32)

    cdef float32_t dist, Dinvs, total, r, rinv, tmp, Di
    cdef size_t nzeros = p
    cdef size_t iteration

    with nogil:
        iteration = 0
        while iteration < maxiters:

            for i in range(p):
                Di = dist_euclidean(X[i, :], y)
                if fabs(Di) > eps:
                    Dinv[i] = 1. / Di
                else:
                    Dinv[i] = 0.
                D[i] = Di

            Dinvs = sum(Dinv)

            for i in range(p):
                W[i] = Dinv[i] / Dinvs

            for j in range(n):
                total = 0.
                for i in range(p):
                    if fabs(D[i]) > eps:
                        total += W[i] * X[i, j]
                T[j] = total

            nzeros = p
            for i in range(p):
                if fabs(D[i]) > eps:
                    nzeros -= 1

            if nzeros == 0:
                y1 = T
            elif nzeros == p:
                break
            else:
                for j in range(n):
                    R[j] = (T[j] - y[j]) * Dinvs
                r = norm_euclidean(R)
                if r > eps:
                    rinv = nzeros/r
                else:
                    rinv = 0.
                for j in range(n):
                    y1[j] = max(0, 1-rinv)*T[j] + min(1, rinv)*y[j]

            dist = dist_euclidean(y, y1)
            if dist < eps:
               break

            y[:] = y1
            iteration = iteration + 1
    
    return y

cdef geomedian_axis_one(const float32_t[:, :] X, float32_t[:] result, float32_t eps=1e-7, size_t maxiters=500):
    cdef size_t p = X.shape[0]
    cdef size_t n = X.shape[1]

    cdef float32_t[:] y = np.mean(X, axis=1)

    if n == 1:
        return y

    cdef float32_t[:] D = np.empty(n, dtype=np.float32)
    cdef float32_t[:] Dinv = np.empty(n, dtype=np.float32)
    cdef float32_t[:] W = np.empty(n, dtype=np.float32)
    cdef float32_t[:] T = np.empty(p, dtype=np.float32)
    cdef float32_t[:] y1 = np.empty(p, dtype=np.float32)
    cdef float32_t[:] R = np.empty(p, dtype=np.float32)

    cdef float32_t dist, Dinvs, total, r, rinv, tmp, Di
    cdef size_t nzeros = n
    cdef size_t iteration

    with nogil:
        iteration = 0
        while iteration < maxiters:

            for i in range(n):
                Di = dist_euclidean(X[:, i], y)
                D[i] = Di
                if fabs(Di) > eps:
                    Dinv[i] = 1. / Di
                else:
                    Dinv[i] = 0.

            Dinvs = sum(Dinv)

            for i in range(n):
                W[i] = Dinv[i] / Dinvs

            for j in range(p):
                total = 0.
                for i in range(n):
                    if fabs(D[i]) > eps:
                        total += W[i] * X[j, i]
                T[j] = total

            nzeros = n
            for i in range(n):
                if fabs(D[i]) > eps:
                    nzeros -= 1

            if nzeros == 0:
                y1 = T
            elif nzeros == n:
                break
            else:
                for j in range(p):
                    R[j] = (T[j] - y[j]) * Dinvs
                r = norm_euclidean(R)
                if r > eps:
                    rinv = nzeros/r
                else:
                    rinv = 0.
                for j in range(p):
                    y1[j] = max(0, 1-rinv)*T[j] + min(1, rinv)*y[j]

            dist = dist_euclidean(y, y1)
            if dist < eps:
               break

            y[:] = y1
            iteration = iteration + 1
            
    return y


cdef nangeomedian_axis_zero(const float32_t[:, :] X, float32_t[:] result, float32_t eps=1e-7, size_t maxiters=500):
    cdef size_t p = X.shape[0]
    cdef size_t n = X.shape[1]

    cdef float32_t nan = <float32_t>float('NaN')

    cdef float32_t[:] y = np.nanmean(X, axis=0)

    cdef float32_t[:] D = np.empty(p, dtype=np.float32)
    cdef float32_t[:] Dinv = np.empty(p, dtype=np.float32)
    cdef float32_t[:] W = np.empty(p, dtype=np.float32)
    cdef float32_t[:] T = np.empty(n, dtype=np.float32)
    cdef float32_t[:] y1 = np.empty(n, dtype=np.float32)
    cdef float32_t[:] R = np.empty(n, dtype=np.float32)

    cdef float32_t dist, Dinvs, total, r, rinv, tmp, Di
    cdef size_t nzeros = p
    cdef size_t iteration

    with nogil:
        iteration = 0
        while iteration < maxiters:

            for i in range(p):
                Di = dist_euclidean(X[i, :], y)
                if fabs(Di) > 0.:
                    Dinv[i] = 1. / Di
                else:
                    Dinv[i] = nan
                D[i] = Di

            Dinvs = nansum(Dinv)

            for i in range(p):
                W[i] = Dinv[i] / Dinvs

            for j in range(n):
                total = 0.
                for i in range(p):
                   tmp = W[i] * X[i, j]
                   if not isnan(tmp):
                       total += tmp
                T[j] = total

            nzeros = p
            for i in range(p):
                if isnan(D[i]):
                    nzeros -= 1
                elif fabs(D[i]) > 0.:
                    nzeros -= 1

            if nzeros == 0:
                y1 = T
            elif nzeros == p:
                break
            else:
                for j in range(n):
                    R[j] = (T[j] - y[j]) * Dinvs
                r = norm_euclidean(R)
                if r > 0.:
                    rinv = nzeros/r
                else:
                    rinv = 0.
                for j in range(n):
                    y1[j] = max(0, 1-rinv)*T[j] + min(1, rinv)*y[j]

            dist = dist_euclidean(y, y1)
            if dist < eps:
               break

            y[:] = y1
            iteration = iteration + 1
            
        for j in range(p):
            result[j] = y1[j]

    return result


cdef nangeomedian_axis_one(const float32_t[:, :] X, float32_t[:] result, float32_t eps=1e-7, size_t maxiters=500):
    cdef size_t p = X.shape[0]
    cdef size_t n = X.shape[1]

    cdef float32_t nan = <float32_t>float('NaN')

    cdef float32_t[:] y = np.nanmean(X, axis=1)

    cdef float32_t[:] D = np.empty(n, dtype=np.float32)
    cdef float32_t[:] Dinv = np.empty(n, dtype=np.float32)
    cdef float32_t[:] W = np.empty(n, dtype=np.float32)
    cdef float32_t[:] T = np.empty(p, dtype=np.float32)
    cdef float32_t[:] y1 = np.empty(p, dtype=np.float32)
    cdef float32_t[:] R = np.empty(p, dtype=np.float32)

    cdef float32_t dist, Dinvs, total, r, rinv, tmp, Di
    cdef size_t nzeros = n
    cdef size_t iteration

    with nogil:
        iteration = 0
        while iteration < maxiters:

            for i in range(n):
                Di = dist_euclidean(X[:, i], y)
                if fabs(Di) > 0.:
                    Dinv[i] = 1. / Di
                else:
                    Dinv[i] = nan
                D[i] = Di

            Dinvs = nansum(Dinv)

            for i in range(n):
                W[i] = Dinv[i] / Dinvs

            for j in range(p):
                total = 0.
                for i in range(n):
                   tmp = W[i] * X[j, i]
                   if not isnan(tmp):
                       total += tmp
                T[j] = total

            nzeros = n
            for i in range(n):
                if isnan(D[i]):
                    nzeros -= 1
                elif fabs(D[i]) > 0.:
                    nzeros -= 1

            if nzeros == 0:
                y1 = T
            elif nzeros == n:
                break
            else:
                for j in range(p):
                    R[j] = (T[j] - y[j]) * Dinvs
                r = norm_euclidean(R)
                if r > 0.:
                    rinv = nzeros/r
                else:
                    rinv = 0.
                for j in range(p):
                    y1[j] = max(0, 1-rinv)*T[j] + min(1, rinv)*y[j]

            dist = dist_euclidean(y, y1)
            if dist < eps:
               break

            y[:] = y1
            iteration = iteration + 1

        for j in range(p):
            result[j] = y1[j]

    return result


cpdef geomedian(float32_t[:, :] X, size_t axis=1, float32_t eps=1e-8, size_t maxiters=1000):
    """Calculates a Geometric Median for an array `X` of
    shape (p,n).

    If the median is calculated across axis=1 (default)
    (the axis of size n) an array of size (p,1) is
    returned.
    """
    if axis == 0:
        result = np.zeros(X.shape[1], dtype=np.float32)
        return geomedian_axis_zero(X, result, eps, maxiters)

    if axis == 1:
        result = np.zeros(X.shape[0], dtype=np.float32)
        return geomedian_axis_one(X, result, eps, maxiters)
        
    raise IndexError("axis {} out of bounds".format(axis)) 


cpdef nangeomedian(float32_t[:, :] X, size_t axis=1, float32_t eps=1e-7, size_t maxiters=1000):
    """Calculates a Geometric Median for an array `X` of
    shape (p,n).

    If the median is calculated across axis=1 (default)
    (the axis of size n) an array of size (p,1) is
    returned.
    
    Missing values should be assigned as `np.nan`.
    """
    if axis == 0:
        ngood = np.count_nonzero(~np.isnan(X).any(axis=1))
        if ngood == 0:
            raise ValueError("All-NaN slice encountered")
        elif ngood < 3:
            return np.nanmedian(X, axis=axis)
        else:
            result = np.zeros(X.shape[1], dtype=np.float32)
            return nangeomedian_axis_zero(X, result, eps, maxiters)

    if axis == 1:
        ngood = np.count_nonzero(~np.isnan(X).any(axis=0))
        if ngood == 0:
            raise ValueError("All-NaN slice encountered")
        elif ngood < 3:
            return np.nanmedian(X, axis=axis)
        else:
            result = np.zeros(X.shape[0], dtype=np.float32)
            return nangeomedian_axis_one(X, result, eps, maxiters)

    raise IndexError("axis {} out of bounds".format(axis)) 


def __bad_mask(np.ndarray[floating, ndim=4] X):
    """ Input is bad if all observation for a given X,Y location are nan.
        Individual observation is nan if any band value is nan.

        Returns
        ========
        2-d boolean array with the shape equal to X.shape[:2]
        True  -- all observations were nan for this column
        False -- at least one valid observation for this column
    """
    return np.isnan(X.sum(axis=2)).all(axis=2)


def __nangeomedian_pcm_float32(const float32_t [:,:,:,:] X, float32_t [:,:,:] result, float32_t eps, size_t maxiters):
    cdef int m = X.shape[0]
    cdef int q = X.shape[1]
    cdef int p = X.shape[2]
    cdef int n = X.shape[3]
    cdef int row, col

    for row in range(m):
        for col in range(q):
            nangeomedian_axis_one(X[row, col, :, :], result[row, col,:], eps, maxiters)


def __nangeomedian_pcm_int16(const int16_t [:,:,:,:] X, int16_t [:,:,:] result, float32_t eps, size_t maxiters):
    cdef int m = X.shape[0]
    cdef int q = X.shape[1]
    cdef int p = X.shape[2]
    cdef int n = X.shape[3]
    cdef int row, col

    for row in range(m):
        for col in range(q):
            vs = np.array(X[row, col, :, :], dtype=np.float32)
            rs = np.empty(p, dtype=np.float32)
            nangeomedian_axis_one(vs, rs)
            for i in range(p):
                result[row, col, i] = rs[i]


def nangeomedian_pcm(X, eps=1e-7, maxiters=1000, num_threads=1, nodata=None, nocheck=False):
    """
    Generate a geometric median pixel composite mosaic by reducing along 
    the last axis.

    Parameters
    ----------
    X : array_like of dtype float32 or float64
        The array has dimensions (m, q, p, n).
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

    if X.dtype == np.int16:
        if nodata is None:
            nodata = 0
        result = np.empty((m, q, p), dtype=np.int16)
        __nangeomedian_pcm_int16(X, result, eps, maxiters)
    else:
        if nodata is None:
            nodata = np.nan

        result = np.empty((m, q, p), dtype=np.float32)
        __nangeomedian_pcm_float32(X, result, eps, maxiters)

        if not nocheck:
            bad = __bad_mask(X)
            result[bad] = np.nan

    np.seterr(**old)

    return result
