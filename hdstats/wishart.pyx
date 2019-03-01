# cython: cdivision=True, boundscheck=False, nonecheck=False, wraparound=False, language_level=3

import numpy as np
import warnings

from libc.math cimport isnan, sqrt, acos, fabs
from scipy.linalg.cython_lapack cimport dpotrf
from scipy.linalg.cython_blas cimport dtrmm, dsyrk
from libc.stdlib cimport abort, malloc, free, calloc
from libc.string cimport memcpy, memset

cimport numpy as cnp

ctypedef fused floating:
    cnp.float32_t
    cnp.float64_t

ctypedef cnp.float32_t float32_t
ctypedef cnp.float64_t float64_t

cdef extern from "randomkit.h":
    ctypedef struct rk_state:
        pass
    unsigned long rk_random(rk_state * state)
    void rk_seed(unsigned long seed, rk_state * state)
    extern double rk_double(rk_state *state)

cdef extern from "dists.c":
    cdef double rk_normal(rk_state *state, double loc, double scale)
    cdef double rk_standard_exponential(rk_state *state)
    cdef double rk_exponential(rk_state *state, double scale)
    cdef double rk_uniform(rk_state *state, double loc, double scale)
    cdef double rk_standard_gamma(rk_state *state, double shape)
    cdef double rk_gamma(rk_state *state, double shape, double scale)
    cdef double rk_chisquare(rk_state *state, double df)
    cdef double rk_noncentral_chisquare(rk_state *state, double df, double nonc)
    cdef double rk_f(rk_state *state, double dfnum, double dfden)
    cdef double rk_noncentral_f(rk_state *state, double dfnum, double dfden, double nonc)
    cdef double rk_noncentral_f(rk_state *state, double dfnum, double dfden, double nonc)

cdef sample_cholesky_factor(int df, int p, double *ans, unsigned long seed):
    cdef int i, j, uind, lind

    cdef rk_state state
    rk_seed(seed, &state)

    memset(&ans[0], 0, p * p * sizeof(double))

    for j in range(p):
        ans[j*(p+1)] = sqrt(rk_chisquare(&state, df - <double>j))
        for i in range(i, j):
            ans[i+j*p] = rk_normal(&state, 0.0, 1.0)
            ans[j+i*p] = 0.0

def _sample_wishart(int n, int df, float64_t[:,:] S, float64_t[:,:,:] X, unsigned long seed):
    cdef int i, j
    cdef int p = S.shape[1]
    cdef int psq = p*p
    cdef int info = 0

    dpotrf('U', &p, &S[0,0], &p, &info)

    cdef double one = 1
    cdef double zero = 0
    cdef double *ansj

    cdef double *tmp = <double *>malloc(p * p * sizeof(double))

    for j in range(n):
        ansj = &X[0,0,0] + j * p * p 
        sample_cholesky_factor(df, p, tmp, seed)
        dtrmm('R', 'U', 'N', 'N', &p, &p, &one, &S[0,0], &p, tmp, &p)
        dsyrk('U', 'T', &p, &p, &one, tmp, &p, &zero, ansj, &p)

    for i in range(1, p):
        for k in range(i):
            ansj[i + k * p] = ansj[k + i * p]

    free(tmp)

    return info


def sample_wishart(n, df, Sigma, seed = 1):
    p = Sigma.shape[1]

    if Sigma.shape[0] != p:
        raise ValueError('Sigma must be a square matrix')

    X = np.empty((p, p, n), dtype=np.float64, order='F')
    S = np.asfortranarray(Sigma, dtype=np.float64)

    info = _sample_wishart(n, df, S, X, seed)
    assert info == 0

    return X

def foobar(unsigned long seed=1):
    cdef rk_state state
    rk_seed(seed, &state)
    cdef double loc = 0.
    cdef double scale = 1.
    cdef double random_value
    random_value = rk_normal(&state, loc, scale)
    return random_value
