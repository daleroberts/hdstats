# cython: cdivision=True, boundscheck=False, nonecheck=False, wraparound=False, language_level=3

import numpy as np
import hdstats

from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from scipy.signal import cwt, find_peaks_cwt, ricker, welch, wiener

cimport numpy as np
cimport openmp

from cython.parallel import prange, parallel, threadid
from libc.stdlib cimport abort, malloc, free
from libc.math cimport isnan, sqrt, acos, fabs, exp, log
from .utils import get_max_threads
from .dtw import dtw, local_dtw, dtw_dist

ctypedef np.float32_t floating
ctypedef np.float32_t float32_t
ctypedef np.float64_t float64_t


def __cosdist(floating [:, :, :, :] X, floating [:, :, :] gm, floating [:,:,:] result, num_threads):
    """ """
    cdef int m = X.shape[0]
    cdef int q = X.shape[1]
    cdef int p = X.shape[2]
    cdef int n = X.shape[3]
    
    cdef float64_t numer, norma, normb, value
    cdef int j, t, row, col

    cdef int number_of_threads = num_threads

    with nogil, parallel(num_threads=number_of_threads):
        for row in prange(m, schedule='static'):
            for col in range(q):
                for t in range(n):
                    
                    numer = 0.
                    norma = 0.
                    normb = 0.
                    
                    for j in range(p):
                        value = X[row, col, j, t]*gm[row, col, j]
                        numer = numer + value
                        norma = norma + X[row, col, j, t]*X[row, col, j, t]
                        normb = normb + gm[row, col, j]*gm[row, col, j]

                    result[row, col, t] = 1. - numer/(sqrt(norma)*sqrt(normb))


def cosdist(np.ndarray[floating, ndim=4] X, np.ndarray[floating, ndim=3] gm, num_threads=None):
    cdef int m = X.shape[0]
    cdef int q = X.shape[1]
    cdef int p = X.shape[2]
    cdef int n = X.shape[3]
    
    dtype = np.float32

    if num_threads is None:
        num_threads = get_max_threads()

    result = np.empty((m, q, n), dtype=dtype)

    __cosdist(X, gm, result, num_threads)
    
    return result


def __eucdist(floating [:, :, :, :] X, floating [:, :, :] gm, floating [:,:,:] result, num_threads):
    cdef int m = X.shape[0]
    cdef int q = X.shape[1]
    cdef int p = X.shape[2]
    cdef int n = X.shape[3]
    
    cdef float64_t total, value
    cdef int j, t, row, col

    cdef int number_of_threads = num_threads

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


def eucdist(np.ndarray[floating, ndim=4] X, np.ndarray[floating, ndim=3] gm, num_threads=None):
    cdef int m = X.shape[0]
    cdef int q = X.shape[1]
    cdef int p = X.shape[2]
    cdef int n = X.shape[3]
    
    dtype = np.float32

    if num_threads is None:
        num_threads = get_max_threads()

    result = np.empty((m, q, n), dtype=dtype)
    
    __eucdist(X, gm, result, num_threads)
    
    return result


def completion(data, s=1.0):
    kernel = Gaussian2DKernel(x_stddev=s, y_stddev=s)
    data = np.transpose(data, [0,2,1,3])
    for i in range(data.shape[0]):
        for b in range(data.shape[1]):
            data[i,b,:,:] = interpolate_replace_nans(data[i,b,:,:], kernel)
    data = np.transpose(data, [0,2,1,3])
    return data


def fast_completion(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[-1]), 0)
    np.maximum.accumulate(idx, axis=-1, out=idx)
    i, j = np.meshgrid(
        np.arange(idx.shape[0]), np.arange(idx.shape[1]), indexing="ij"
    )
    dat = arr[i[:, :, np.newaxis], j[:, :, np.newaxis], idx]
    if np.isnan(np.sum(dat[:, :, 0])):
        fill = np.nanmean(dat, axis=-1)
        for t in range(dat.shape[-1]):
            mask = np.isnan(dat[:, :, t])
            if mask.any():
                dat[mask, t] = fill[mask]
            else:
                break
    return dat


def smooth(arr, k=3):
    return wiener(arr, (1, 1, k))


def discordance(x, n=10):
    X = x.copy()

    mX = np.mean(X, axis=(0,1))
    Y = np.fft.fft(mX)
    np.put(Y, range(n, mX.shape[0]), 0.0)
    mX = np.abs(np.fft.ifft(Y)).astype(np.float32)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y = np.fft.fft(X[i,j,:])
            np.put(Y, range(n, mX.shape[0]), 0.0)
            X[i,j,:] = np.real(np.fft.ifft(Y))

    X -= mX[np.newaxis, np.newaxis, :]

    return np.mean(X, axis=2)


def fourier_mean(x, n=3, step=5):
    result = np.empty((x.shape[0], x.shape[1], n), dtype=np.float32)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y = np.fft.fft(x[i,j,:])
            for k in range(n):
                result[i,j,k] = np.mean(np.abs(y[1+k*step:((k+1)*step+1) or None]))

    return result


def fourier_std(x, n=3, step=5):
    result = np.empty((x.shape[0], x.shape[1], n), dtype=np.float32)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y = np.fft.fft(x[i,j,:])
            for k in range(n):
                result[i,j,k] = np.std(np.abs(y[1+k*step:((k+1)*step+1) or None]))

    return result


def fourier_median(x, n=3, step=5):
    result = np.empty((x.shape[0], x.shape[1], n), dtype=np.float32)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y = np.fft.fft(x[i,j,:])
            for k in range(n):
                result[i,j,k] = np.median(np.abs(y[1+k*step:((k+1)*step+1) or None]))

    return result


def mean_change(x):
    return np.mean(np.diff(x), axis=-1)


def median_change(x):
    return np.median(np.diff(x), axis=-1)


def mean_abs_change(x):
    return np.mean(np.abs(np.diff(x)), axis=-1)


def mean_central_diff(x):
    diff = (np.roll(x, 1, axis=2) - 2 * x + np.roll(x, -1, axis=2))/2.0
    return np.mean(diff[:,:,1:-1], axis=2)


def complexity(x, normalize=True):
    if normalize:
        s = np.std(x, axis=2)
        x = (x-np.mean(x, axis=2)[:,:,np.newaxis]) / s[:,:,np.newaxis]

    z = np.diff(x)

    return np.einsum('ijk,ijk->ij', z, z)


def number_peaks(x, n=10):
    npeaks = np.empty(x.shape[:2], dtype=np.int8)
    for i in range(npeaks.shape[0]):
        for j in range(npeaks.shape[1]):
            npeaks[i,j] = len(find_peaks_cwt(vector=x[i,j,:], widths=np.array(list(range(1, n + 1))), wavelet=ricker))
    
    return npeaks


def symmetry(x, gm=None, num_threads=None):
    if num_threads is None:
        num_threads = get_max_threads()

    if gm is None:
        gm = hdstats.nangeomedian_pcm(x, num_threads=num_threads)

    mm = np.nanmean(x, axis=3)

    cd = cosdist(mm[:,:,:,np.newaxis], gm, num_threads=num_threads)
    cd = cd.reshape(cd.shape[:2])

    return cd


def area_warp_similarity(x, areats=None):
    if areats is None:
        areats = np.median(x, axis=(0,1))

    similarity = np.empty(x.shape[:2], dtype=np.float32)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            similarity[i,j] = dtw_dist(areats.reshape(1,-1), x[i,j,:].reshape(1,-1))

    return similarity

