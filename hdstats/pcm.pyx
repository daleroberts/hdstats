#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3, embedsignature=True

import numpy as np

cimport numpy as np
cimport openmp

from cython.parallel import prange, parallel, threadid
from libc.stdlib cimport abort, malloc, free
from libc.math cimport isnan, sqrt, acos, fabs, exp, log
from .utils import get_max_threads

ctypedef np.int16_t int16_t
ctypedef np.int16_t uint16_t
ctypedef np.float32_t floating
ctypedef np.float32_t float32_t
ctypedef np.float64_t float64_t
ctypedef np.npy_bool bool

MAXITERS = 10000
EPS = 1e-4


def __gm(const floating [:, :, :, :] X, floating [:, :, :] mX,
         const floating [:] w, int maxiters, const floating eps,
         int num_threads):
    """ """
    cdef int number_of_threads = num_threads
    cdef int m = X.shape[0]
    cdef int q = X.shape[1]
    cdef int p = X.shape[2]
    cdef int n = X.shape[3]
    cdef int i, j, k, row, col
    cdef int nzeros, iteration
    cdef int reseed = 0
    cdef bool allnan = True
    cdef float64_t dist, Dinvs, total, r, rinv, tmp, Di, d, value
    cdef float64_t nan = <float64_t> np.nan
    cdef floating *D
    cdef floating *Dinv
    cdef floating *T
    cdef floating *W
    cdef floating *y
    cdef floating *y1
    cdef floating *R

    with nogil, parallel(num_threads=number_of_threads):
        Dinv = <floating *> malloc(sizeof(floating) * n)
        y1 = <floating *> malloc(sizeof(floating) * p)
        y = <floating *> malloc(sizeof(floating) * p)
        D = <floating *> malloc(sizeof(floating) * n)
        W = <floating *> malloc(sizeof(floating) * n)
        T = <floating *> malloc(sizeof(floating) * p)
        R = <floating *> malloc(sizeof(floating) * p)

        for row in prange(m, schedule='dynamic'):

            reseed = 1

            for col in range(q):

                # zero everything just to be careful for now...

                for j in range(p):
                    y1[j] = 0.0
                    y[j] = 0.0
                    T[j] = 0.0
                    R[j] = 0.0

                for i in range(n):
                    Dinv[i] = 0.0
                    D[i] = 0.0
                    W[i] = 0.0

                dist = 0.0
                Dinvs = 0.0
                total = 0.0
                r = 0.0
                rinv = 0.0
                Di = 0.0
                d = 0.0
                value = 0.0

                nzeros = 0

                if reseed == 1:

                    for j in range(p):
                        # nanmean
                        total = 0.
                        k = 0
                        for i in range(n):
                            value = X[row, col, j, i]
                            if not isnan(value):
                                total = total + value
                                k = k + 1
                        y[j] = total / k

                iteration = 0
                while iteration < maxiters:

                    for i in range(n):

                        # euclidean distance
                        total = 0.
                        for j in range(p):
                            value = X[row, col, j, i]
                            if not isnan(value) and not isnan(y[j]):
                                value = value - y[j]
                                total = total + value*value
                            else:
                                total = nan
                                break
                        Di = sqrt(total)

                        D[i] = Di
                        if not isnan(Di):
                            if fabs(Di) > 0.:
                                Dinv[i] = w[i] / Di
                            else:
                                Dinv[i] = w[i]
                        else:
                            Dinv[i] = nan

                    # nansum
                    Dinvs = 0.
                    for i in range(n):
                        if not isnan(Dinv[i]):
                            Dinvs = Dinvs + Dinv[i]

                    for i in range(n):
                        W[i] = Dinv[i] / Dinvs

                    for j in range(p):
                        allnan = True
                        total = 0.
                        for i in range(n):
                            tmp = W[i] * X[row, col, j, i]
                            if not isnan(tmp):
                                total = total + tmp
                                allnan = False

                        if allnan:
                            T[j] = nan
                        else:
                            T[j] = total

                    nzeros = n
                    for i in range(n):
                        if isnan(D[i]) or fabs(D[i]) > 0.:
                            nzeros = nzeros - 1

                    if nzeros == 0:
                        for j in range(p):
                            y1[j] = T[j]
                    elif nzeros == n:
                        for j in range(p):
                            y1[j] = y[j]
                        break
                    else:
                        for j in range(p):
                            R[j] = (T[j] - y[j]) * Dinvs

                        r = 0.
                        for j in range(p):
                            r = r + R[j]*R[j]
                        r = sqrt(r)

                        if r > 0.:
                            rinv = nzeros/r
                        else:
                            rinv = 0.

                        for j in range(p):
                            y1[j] = max(0, 1-rinv)*T[j] + \
                                min(1, rinv)*y[j]

                    total = 0.
                    for j in range(p):
                        value = y[j] - y1[j]
                        total = total + value*value
                    dist = sqrt(total)

                    for j in range(p):
                        y[j] = y1[j]

                    iteration = iteration + 1

                    if isnan(dist):
                        reseed = 1
                        break
                    else:
                        reseed = 0

                    if dist < eps:
                        break

                for j in range(p):
                    mX[row, col, j] = y1[j]

        free(Dinv)
        free(y1)
        free(D)
        free(W)
        free(T)
        free(R)

def __gm_int16(const int16 [:, :, :, :] X, floating [:, :, :] mX,
               const floating [:] w, int maxiters, const floating eps,
               int num_threads, int16 nodata=-999):
    """ """
    cdef int number_of_threads = num_threads
    cdef int m = X.shape[0]
    cdef int q = X.shape[1]
    cdef int p = X.shape[2]
    cdef int n = X.shape[3]
    cdef int i, j, k, row, col
    cdef int nzeros, iteration
    cdef int reseed = 0
    cdef bool allnan = True
    cdef int16 fixedvalue
    cdef float64_t dist, Dinvs, total, r, rinv, tmp, Di, d, value
    cdef float64_t nan = <float64_t> np.nan
    cdef floating *D
    cdef floating *Dinv
    cdef floating *T
    cdef floating *W
    cdef floating *y
    cdef floating *y1
    cdef floating *R

    with nogil, parallel(num_threads=number_of_threads):
        Dinv = <floating *> malloc(sizeof(floating) * n)
        y1 = <floating *> malloc(sizeof(floating) * p)
        y = <floating *> malloc(sizeof(floating) * p)
        D = <floating *> malloc(sizeof(floating) * n)
        W = <floating *> malloc(sizeof(floating) * n)
        T = <floating *> malloc(sizeof(floating) * p)
        R = <floating *> malloc(sizeof(floating) * p)

        for row in prange(m, schedule='dynamic'):

            reseed = 1

            for col in range(q):

                # zero everything just to be careful for now...

                for j in range(p):
                    y1[j] = 0.0
                    y[j] = 0.0
                    T[j] = 0.0
                    R[j] = 0.0

                for i in range(n):
                    Dinv[i] = 0.0
                    D[i] = 0.0
                    W[i] = 0.0

                dist = 0.0
                Dinvs = 0.0
                total = 0.0
                r = 0.0
                rinv = 0.0
                Di = 0.0
                d = 0.0
                value = 0.0

                nzeros = 0

                if reseed == 1:

                    for j in range(p):
                        # nanmean
                        total = 0.
                        k = 0
                        for i in range(n):
                            fixedvalue = X[row, col, j, i]
                            value = fixedvalue / 10000.
                            if fixedvalue != nodata:
                                total = total + value
                                k = k + 1
                        y[j] = total / k

                iteration = 0
                while iteration < maxiters:

                    for i in range(n):

                        # euclidean distance
                        total = 0.
                        for j in range(p):
                            fixedvalue = X[row, col, j, i]
                            value = fixedvalue / 10000.
                            if fixedvalue != nodata and not isnan(y[j]):
                                value = value - y[j]
                                total = total + value*value
                            else:
                                total = nan
                                break
                        Di = sqrt(total)

                        D[i] = Di
                        if not isnan(Di):
                            if fabs(Di) > 0.:
                                Dinv[i] = w[i] / Di
                            else:
                                Dinv[i] = w[i]
                        else:
                            Dinv[i] = nan

                    # nansum
                    Dinvs = 0.
                    for i in range(n):
                        if not isnan(Dinv[i]):
                            Dinvs = Dinvs + Dinv[i]

                    for i in range(n):
                        W[i] = Dinv[i] / Dinvs

                    for j in range(p):
                        allnan = True
                        total = 0.
                        for i in range(n):
                            fixedvalue = X[row, col, j, i]
                            tmp = W[i] * X[row, col, j, i] / 10000.
                            if not isnan(tmp) and fixedvalue != nodata:
                                total = total + tmp
                                allnan = False

                        if allnan:
                            T[j] = nan
                        else:
                            T[j] = total

                    nzeros = n
                    for i in range(n):
                        if isnan(D[i]) or fabs(D[i]) > 0.:
                            nzeros = nzeros - 1

                    if nzeros == 0:
                        for j in range(p):
                            y1[j] = T[j]
                    elif nzeros == n:
                        for j in range(p):
                            y1[j] = y[j]
                        break
                    else:
                        for j in range(p):
                            R[j] = (T[j] - y[j]) * Dinvs

                        r = 0.
                        for j in range(p):
                            r = r + R[j]*R[j]
                        r = sqrt(r)

                        if r > 0.:
                            rinv = nzeros/r
                        else:
                            rinv = 0.

                        for j in range(p):
                            y1[j] = max(0, 1-rinv)*T[j] + \
                                min(1, rinv)*y[j]

                    total = 0.
                    for j in range(p):
                        value = y[j] - y1[j]
                        total = total + value*value
                    dist = sqrt(total)

                    for j in range(p):
                        y[j] = y1[j]

                    iteration = iteration + 1

                    if isnan(dist):
                        reseed = 1
                        break
                    else:
                        reseed = 0

                    if dist < eps:
                        break

                for j in range(p):
                    mX[row, col, j] = y1[j]

        free(Dinv)
        free(y1)
        free(D)
        free(W)
        free(T)
        free(R)

def __gm_uint16(const uint16 [:, :, :, :] X, floating [:, :, :] mX,
               const floating [:] w, int maxiters, const floating eps,
               int num_threads, uint16 nodata=0):
    """ """
    cdef int number_of_threads = num_threads
    cdef int m = X.shape[0]
    cdef int q = X.shape[1]
    cdef int p = X.shape[2]
    cdef int n = X.shape[3]
    cdef int i, j, k, row, col
    cdef int nzeros, iteration
    cdef int reseed = 0
    cdef bool allnan = True
    cdef uint16 fixedvalue
    cdef float64_t dist, Dinvs, total, r, rinv, tmp, Di, d, value
    cdef float64_t nan = <float64_t> np.nan
    cdef floating *D
    cdef floating *Dinv
    cdef floating *T
    cdef floating *W
    cdef floating *y
    cdef floating *y1
    cdef floating *R

    with nogil, parallel(num_threads=number_of_threads):
        Dinv = <floating *> malloc(sizeof(floating) * n)
        y1 = <floating *> malloc(sizeof(floating) * p)
        y = <floating *> malloc(sizeof(floating) * p)
        D = <floating *> malloc(sizeof(floating) * n)
        W = <floating *> malloc(sizeof(floating) * n)
        T = <floating *> malloc(sizeof(floating) * p)
        R = <floating *> malloc(sizeof(floating) * p)

        for row in prange(m, schedule='dynamic'):

            reseed = 1

            for col in range(q):

                # zero everything just to be careful for now...

                for j in range(p):
                    y1[j] = 0.0
                    y[j] = 0.0
                    T[j] = 0.0
                    R[j] = 0.0

                for i in range(n):
                    Dinv[i] = 0.0
                    D[i] = 0.0
                    W[i] = 0.0

                dist = 0.0
                Dinvs = 0.0
                total = 0.0
                r = 0.0
                rinv = 0.0
                Di = 0.0
                d = 0.0
                value = 0.0

                nzeros = 0

                if reseed == 1:

                    for j in range(p):
                        # nanmean
                        total = 0.
                        k = 0
                        for i in range(n):
                            fixedvalue = X[row, col, j, i]
                            value = fixedvalue / 10000.
                            if fixedvalue != nodata:
                                total = total + value
                                k = k + 1
                        y[j] = total / k

                iteration = 0
                while iteration < maxiters:

                    for i in range(n):

                        # euclidean distance
                        total = 0.
                        for j in range(p):
                            fixedvalue = X[row, col, j, i]
                            value = fixedvalue / 10000.
                            if fixedvalue != nodata and not isnan(y[j]):
                                value = value - y[j]
                                total = total + value*value
                            else:
                                total = nan
                                break
                        Di = sqrt(total)

                        D[i] = Di
                        if not isnan(Di):
                            if fabs(Di) > 0.:
                                Dinv[i] = w[i] / Di
                            else:
                                Dinv[i] = w[i]
                        else:
                            Dinv[i] = nan

                    # nansum
                    Dinvs = 0.
                    for i in range(n):
                        if not isnan(Dinv[i]):
                            Dinvs = Dinvs + Dinv[i]

                    for i in range(n):
                        W[i] = Dinv[i] / Dinvs

                    for j in range(p):
                        allnan = True
                        total = 0.
                        for i in range(n):
                            fixedvalue = X[row, col, j, i]
                            tmp = W[i] * X[row, col, j, i] / 10000.
                            if not isnan(tmp) and fixedvalue != nodata:
                                total = total + tmp
                                allnan = False

                        if allnan:
                            T[j] = nan
                        else:
                            T[j] = total

                    nzeros = n
                    for i in range(n):
                        if isnan(D[i]) or fabs(D[i]) > 0.:
                            nzeros = nzeros - 1

                    if nzeros == 0:
                        for j in range(p):
                            y1[j] = T[j]
                    elif nzeros == n:
                        for j in range(p):
                            y1[j] = y[j]
                        break
                    else:
                        for j in range(p):
                            R[j] = (T[j] - y[j]) * Dinvs

                        r = 0.
                        for j in range(p):
                            r = r + R[j]*R[j]
                        r = sqrt(r)

                        if r > 0.:
                            rinv = nzeros/r
                        else:
                            rinv = 0.

                        for j in range(p):
                            y1[j] = max(0, 1-rinv)*T[j] + \
                                min(1, rinv)*y[j]

                    total = 0.
                    for j in range(p):
                        value = y[j] - y1[j]
                        total = total + value*value
                    dist = sqrt(total)

                    for j in range(p):
                        y[j] = y1[j]

                    iteration = iteration + 1

                    if isnan(dist):
                        reseed = 1
                        break
                    else:
                        reseed = 0

                    if dist < eps:
                        break

                for j in range(p):
                    mX[row, col, j] = y1[j]

        free(Dinv)
        free(y1)
        free(D)
        free(W)
        free(T)
        free(R)

def __wgm(const floating [:, :, :, :] X, floating [:, :, :] mX,
          int bi, int bj, float64_t rho, float64_t delta,
          float64_t xi, const floating [:] alpha,
          const floating [:] gamma,
          const floating [:] beta, const floating [:] sigma, int maxiters,
          float64_t eps, int num_threads):
    """ """
    cdef int number_of_threads = num_threads
    cdef int m = X.shape[0]
    cdef int q = X.shape[1]
    cdef int p = X.shape[2]
    cdef int n = X.shape[3]
    cdef int i, j, k, l, row, col
    cdef int nzeros, iteration
    cdef int reseed = 0
    cdef float64_t dist, Dinvs, total, r, rinv, tmp, Di, d, value
    cdef float64_t numer, denom, min_weight, max_weight
    cdef float64_t nan = <float64_t> np.nan
    cdef floating *D
    cdef floating *Dinv
    cdef floating *W
    cdef floating *T
    cdef floating *y
    cdef floating *y1
    cdef floating *R
    cdef floating *w

    with nogil, parallel(num_threads=number_of_threads):
        Dinv = <floating *> malloc(sizeof(floating) * n)
        y1 = <floating *> malloc(sizeof(floating) * p)
        y = <floating *> malloc(sizeof(floating) * p)
        D = <floating *> malloc(sizeof(floating) * n)
        W = <floating *> malloc(sizeof(floating) * n)
        T = <floating *> malloc(sizeof(floating) * p)
        R = <floating *> malloc(sizeof(floating) * p)
        w = <floating *> malloc(sizeof(floating) * n)

        for row in prange(m, schedule='dynamic'):

            reseed = 1

            for col in range(q):

                # zero everything just to be careful for now...

                for j in range(p):
                    y1[j] = 0.0
                    y[j] = 0.0
                    T[j] = 0.0
                    R[j] = 0.0

                for i in range(n):
                    Dinv[i] = 0.0
                    D[i] = 0.0
                    W[i] = 0.0

                dist = 0.0
                Dinvs = 0.0
                total = 0.0
                r = 0.0
                rinv = 0.0
                Di = 0.0
                d = 0.0
                value = 0.0
                numer = 0.0
                denom = 0.0

                nzeros = 0

                # calculate weights
                for i in range(n):
                    value = (X[row, col, bi, i] - X[row, col, bj, i])/(X[row, col, bi, i] + X[row, col, bj, i] + xi)
                    numer = rho
                    denom = delta
                    for j in range(p):
                        numer = numer + alpha[j]*(X[row, col, j, i] - gamma[j])
                        denom = denom + beta[j]*(X[row, col, j, i] - sigma[j])
                    value = value*(numer/denom)
                    if isnan(value):
                        w[i] = 0.0
                    else:
                        w[i] = value

                # weighted softmax
                value = 0.
                for i in range(n):
                    w[i] = exp(w[i])
                    if isnan(w[i]):
                        w[i] = 0.
                    else:
                        value = value + w[i]
                for i in range(n):
                    w[i] = w[i] / value

                if reseed == 1:

                    for j in range(p):
                        # nanmean
                        total = 0.
                        k = 0
                        for i in range(n):
                            value = X[row, col, j, i]
                            if not isnan(value):
                                total = total + value
                                k = k + 1
                        y[j] = total / k

                iteration = 0
                while iteration < maxiters:

                    for i in range(n):

                        # euclidean distance
                        total = 0.
                        for j in range(p):
                            value = X[row, col, j, i] - y[j]
                            total = total + value*value
                        Di = sqrt(total)

                        D[i] = Di
                        if not isnan(Di) and fabs(Di) > 0.:
                            Dinv[i] = w[i] / Di
                        else:
                            Dinv[i] = nan

                    # nansum
                    Dinvs = 0.
                    for i in range(n):
                        if not isnan(Dinv[i]):
                            Dinvs = Dinvs + Dinv[i]

                    for i in range(n):
                        W[i] = Dinv[i] / Dinvs

                    for j in range(p):
                        total = 0.
                        for i in range(n):
                            tmp = W[i] * X[row, col, j, i]
                            if not isnan(tmp):
                                total = total + tmp
                        T[j] = total

                    nzeros = n
                    for i in range(n):
                        if isnan(D[i]) or fabs(D[i]) > 0.:
                            nzeros = nzeros - 1

                    if nzeros == 0:
                        for j in range(p):
                            y1[j] = T[j]
                    elif nzeros == n:
                        break
                    else:
                        for j in range(p):
                            R[j] = (T[j] - y[j]) * Dinvs

                        r = 0.
                        for j in range(p):
                            r = r + R[j]*R[j]
                        r = sqrt(r)

                        if r > 0.:
                            rinv = nzeros/r
                        else:
                            rinv = 0.

                        for j in range(p):
                            y1[j] = max(0, 1-rinv)*T[j] + min(1, rinv)*y[j]

                    total = 0.
                    for j in range(p):
                        value = y[j] - y1[j]
                        total = total + value*value
                    dist = sqrt(total)

                    for j in range(p):
                        y[j] = y1[j]

                    iteration = iteration + 1

                    if isnan(dist):
                        reseed = 1
                        break
                    else:
                        reseed = 0

                    if dist < eps:
                        break

                for j in range(p):
                    mX[row, col, j] = y1[j]

        free(Dinv)
        free(y1)
        free(D)
        free(W)
        free(T)
        free(R)


def __emad(const floating [:, :, :, :] X, const floating [:, :, :] gm, floating [:,:,:] result, int num_threads):
    cdef int number_of_threads = num_threads
    cdef int m = X.shape[0]
    cdef int q = X.shape[1]
    cdef int p = X.shape[2]
    cdef int n = X.shape[3]
    cdef int j, t, row, col
    cdef float64_t total, value

    with nogil, parallel(num_threads=number_of_threads):
        for row in prange(m, schedule='static'):
            for col in range(q):
                for t in range(n):
                    # euclidean distance
                    total = 0.
                    for j in range(p):
                        value = X[row, col, j, t] - gm[row, col, j]
                        total = total + value*value
                    result[row, col, t] = sqrt(total)



def __smad(const floating[:, :, :, :] X, const floating[:, :, :] gm, floating[:,:,:] result, int num_threads):
    """ """
    cdef int number_of_threads = num_threads
    cdef int m = X.shape[0]
    cdef int q = X.shape[1]
    cdef int p = X.shape[2]
    cdef int n = X.shape[3]
    cdef int j, t, row, col
    cdef float64_t numer, norma, normb, normb_sqrt, value

    with nogil, parallel(num_threads=number_of_threads):
        for row in prange(m, schedule='static'):
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


def __bcmad(const floating[:, :, :, :] X, const floating[:, :, :] gm, floating[:,:,:] result, int num_threads):
    """ """
    cdef int number_of_threads = num_threads
    cdef int m = X.shape[0]
    cdef int q = X.shape[1]
    cdef int p = X.shape[2]
    cdef int n = X.shape[3]
    cdef int j, t, row, col
    cdef float64_t numer, denom, value

    with nogil, parallel(num_threads=number_of_threads):
        for row in prange(m, schedule='static'):
            for col in range(q):
                for t in range(n):

                    numer = 0.
                    denom = 0.

                    for j in range(p):
                        numer = numer + fabs(X[row, col, j, t] - gm[row, col, j])
                        denom = denom + fabs(X[row, col, j, t] + gm[row, col, j])

                    result[row, col, t] = numer / denom

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

def gm(X, weight=None, maxiters=MAXITERS, floating eps=EPS, num_threads=None, nocheck=False,
       nodata=None):
    """
    Generate a geometric median pixel composite mosaic by reducing along the last axis.

    Parameters
    ----------
    X : array_like of dtype float32 or float64
        The array has dimensions (m, q, p, n).
    weight : array_like
        The optional array of weights of dimension (n,).
    maxiters : int
        The maximum number of iterations to use to find the solution.
    eps : float
        The tolerance for stopping the algorithm.
    num_threads : int
        The number of processing threads to use for the computation.
    nocheck : bool
        Do not perform data checks.
    nodata : uint16 or int16
        If the dtype is (u)int16 use this value to indicate missing data.
        Default is -999 for int16 and 0 for uint16
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

    dtype = np.float32

    if weight is None:
        w = np.ones((n,), dtype=dtype)
    else:
        w = np.array(weight, dtype=dtype)

    result = np.empty((m, q, p), dtype=dtype)

    if X.dtype == np.int16:
        if nodata is None:
            nodata = -999
        __gm_int16(X, result, w, maxiters, eps, num_threads, nodata=nodata)
        return result

    if X.dtype == np.uint16:
        if nodata is None:
            nodata = 0
        __gm_uint16(X, result, w, maxiters, eps, num_threads, nodata=nodata)
        return result

    if not nocheck:
        bad = __bad_mask(X)

    __gm(X, result, w, maxiters, eps, num_threads)

    if not nocheck:
        result[bad] = np.nan

    return result

def wgm(np.ndarray[floating, ndim=4] X, int bi, int bj, float64_t rho=1.0,
        float64_t delta=1.0, float64_t xi=0.0, alpha=None, gamma=None, beta=None, sigma=None,
        int maxiters=MAXITERS, float64_t eps=EPS,
        num_threads=None, nocheck=False):
    """
    Generate a weighted geometric median pixel composite mosaic by reducing along
    the last axis. Weighting is performed on each multivariate time series based
    on a spectral feature model that give a relative ranking of observations at
    the particular pixel.

    Generate simple weight features based on the spectral information in the
    pixel $\mathbb{x} = (x_1, \ldots, x_p)^T$. These features have the form
    $$
        f(\mathbb{x}) = \\frac{x_i-x_j}{x_i+x_j+\\xi}\left(\\frac{\\alpha_1(x_1-\gamma_1)+
        \cdots + \\alpha_p(x_p - \gamma_p) + \\rho}{\\beta_1(x_1-\sigma_1)+
        \cdots + \\beta_p (x_p - \sigma_p) + \delta}\\right)
    $$
    for some choice of $i,j$ and for vectors $\\alpha = (\\alpha_1, \ldots, \\alpha_p)^T$,
    $\gamma = (\gamma_1, \ldots, \gamma_p)^T$, $\\beta = (\\beta_1, \ldots ,\\beta_p)^T$,
    $\sigma = (\sigma_1, \ldots, \sigma_p)^T$ and constants $\\rho$, $\delta$.

    Parameters
    ----------
    X : array_like of dtype float32 or float64
        The array has dimensions (m, q, p, n).
    bi : int
        First index for the normalised difference band ratio.
    bj : int
        Second index for the normalised difference band ratio.
    rho : float
        Constant rho in the model.
    delta : float
        Constant delta in the model.
    alpha : array_like
        Array of dimension (p,) giving alpha vector in model.
    gamma : array_like
        (Optional) array of dimension (p,) giving gamma vector in model.
    beta : array_like
        Array of dimension (p,) giving beta vector in model.
    sigma : array_like
        Array of dimension (p,) giving sigma vector in model.
    maxiters : int
        The maximum number of iterations to use to find the solution.
    eps : float
        The tolerance for stopping the algorithm.
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

    # if X.dtype is np.float32:
    #     dtype = np.float32
    # else:
    #     dtype = np.float64

    dtype = np.float32

    result = np.empty((m, q, p), dtype=dtype)

    if alpha is None:
        alpha = np.zeros(p, dtype=dtype)
    else:
        alpha = np.ascontiguousarray(alpha, dtype=dtype)

    if beta is None:
        beta = np.zeros(p, dtype=dtype)
    else:
        beta = np.ascontiguousarray(beta, dtype=dtype)

    if gamma is None:
        gamma = np.zeros(p, dtype=dtype)
    else:
        gamma = np.ascontiguousarray(gamma, dtype=dtype)

    if sigma is None:
        sigma = np.zeros(p, dtype=dtype)
    else:
        sigma = np.ascontiguousarray(sigma, dtype=dtype)

    if not nocheck:
        bad = __bad_mask(X)

    __wgm(X, result, bi, bj, rho, delta, xi, alpha, gamma, beta, sigma, maxiters, eps, num_threads)

    if not nocheck:
        result[bad] = np.nan

    return result

def emad(np.ndarray[floating, ndim=4] X, np.ndarray[floating, ndim=3] gm, num_threads=None, nocheck=False):
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

    __emad(X, gm, result, num_threads)

    np.seterr(**old)

    if not nocheck:
        return np.nanmedian(result, axis=2)
    else:
        return np.median(result, axis=2)


def smad(np.ndarray[floating, ndim=4] X, np.ndarray[floating, ndim=3] gm, num_threads=None, nocheck=False):
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

    __smad(X, gm, result, num_threads)

    np.seterr(**old)

    if not nocheck:
        return np.nanmedian(result, axis=2)
    else:
        return np.median(result, axis=2)


def bcmad(np.ndarray[floating, ndim=4] X, np.ndarray[floating, ndim=3] gm, num_threads=None, nocheck=False):
    """
    Generate a bray-curtis geometric median absolute deviation pixel
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

    __bcmad(X, gm, result, num_threads)

    np.seterr(**old)

    if not nocheck:
        return np.nanmedian(result, axis=2)
    else:
        return np.median(result, axis=2)
