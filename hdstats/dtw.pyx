#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3, embedsignature=True

import numpy as np
cimport numpy as cnp

cimport openmp

from cython.parallel import prange, parallel, threadid
from libc.stdlib cimport abort, malloc, free
from libc.math cimport isnan, sqrt, acos, fabs, exp, log

ctypedef fused floating:
    cnp.float32_t
    cnp.float64_t
    
ctypedef cnp.float32_t float32_t
ctypedef cnp.float64_t float64_t

def max_threads():
    return openmp.omp_get_max_threads()

cdef inline float32_t min3(float32_t a, float32_t b, float32_t c):
    cdef float32_t m = a
    if b < m:
        m = b
    if c < m:
        m = c
    return m


def local_dtw(cnp.ndarray s, cnp.ndarray t, window=4):
    cdef int r = s.shape[0]
    cdef int c = t.shape[0]
    cdef cnp.ndarray[float32_t, ndim=2] D = np.inf * np.ones((r+1, c+1), dtype=np.float32)

    window = np.max((window, abs(r - c)))

    D[0,0] = 0.0
    
    cdef size_t i,j
    cdef float64_t cost
    
    for i in range(r):
        for j in range(max(0, i - window), min(c, i + window + 1)):
            cost = np.linalg.norm(s[i] - t[j], ord=1)
            D[i+1,j+1] = cost + min3(D[i,j+1], D[i+1,j], D[i,j])

    path = optimalpath(D)
    return D[r,c]/(r+c), D[1:,1:], path

def _dtw_dist(floating[:] x, floating [:] y, floating[:,:] D):
    cdef int r = x.shape[0]
    cdef int c = y.shape[0]
    cdef size_t i, j
    cdef float64_t cost = 0
    
    for i in range(r):
        for j in range(c):
            cost = np.linalg.norm(x[i] - y[j], ord=1)
            D[i+1,j+1] = cost + min3(D[i,j+1], D[i+1,j], D[i,j])

    return D[r,c]/(r+c)


def dtw_dist(cnp.ndarray x, cnp.ndarray y):
    cdef int r = x.shape[0]
    cdef int c = y.shape[0]

    cdef cnp.ndarray[float32_t, ndim=2] D = np.zeros((r+1, c+1), dtype=np.float32)

    D[1:,0] = np.inf
    D[0,1:] = np.inf
    
    cdef size_t i, j
    cdef float64_t cost
    
    for i in range(r):
        for j in range(c):
            cost = np.linalg.norm(x[i] - y[j], ord=1)
            D[i+1,j+1] = cost + min3(D[i,j+1], D[i+1,j], D[i,j])

    return D[r,c]/(r+c)


def dtw(cnp.ndarray x, cnp.ndarray y):
    cdef int r = x.shape[0]
    cdef int c = y.shape[0]

    cdef cnp.ndarray[float32_t, ndim=2] D = np.zeros((r+1, c+1), dtype=np.float32)

    D[1:,0] = np.inf
    D[0,1:] = np.inf
    
    cdef size_t i, j
    cdef float64_t cost
    
    for i in range(r):
        for j in range(c):
            cost = np.linalg.norm(x[i] - y[j], ord=1)
            D[i+1,j+1] = cost + min3(D[i,j+1], D[i+1,j], D[i,j])

    path = optimalpath(D)
    return D[r,c]/(r+c), D[1:,1:], path

def optimalpath(D):
    i, j = np.array(D.shape) - 2
    path = [(i,j)]
    while (i > 0) or (j > 0):
        if D[i,j] <= D[i, j + 1] and D[i, j] <= D[i + 1, j]:
            i -= 1
            j -= 1
        elif D[i, j + 1] <= D[i + 1, j] and D[i, j + 1] <= D[i, j]:
            i -= 1
        else:
            j -= 1
        path.append((i,j))
    return np.array(path)[::-1].T