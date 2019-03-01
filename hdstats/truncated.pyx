# cython: cdivision=True, boundscheck=False, nonecheck=False, wraparound=False, language_level=3

import numpy as np
import warnings

from libc.math cimport isnan, sqrt, acos, fabs, exp, log, floor
from scipy.linalg.cython_lapack cimport dpotrf
from scipy.linalg.cython_blas cimport dtrmm, dsyrk
from libc.stdlib cimport abort, malloc, free, calloc
from libc.string cimport memcpy, memset

cimport numpy as cnp
cimport scipy.special.cython_special as special

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

def erfinv(y):
    return special.ndtri((y+1)/2.0)/sqrt(2)

def pnorm(double x):
    return 0.5 * (1 + special.erf(x/sqrt(2)))

def qnorm(double x):
    return sqrt(2) * erfinv(2*x - 1)

def sample_truncated_normal(int n, floating[:] X, floating[:] U, floating[:,:] directions,
                            floating[:,:] alphas, floating[:,:] result, int burnin=2000,
                            unsigned long seed=1):

    cdef int ndirections = directions.shape[0] # 2*p
    cdef int which_direction
    cdef int c = U.shape[0]
    cdef int p = X.shape[0]
    cdef int i
    cdef int j

    cdef double lower_bound = -1e12
    cdef double upper_bound = 1e12
    cdef double bound_val = 0
    cdef double value = 0
    cdef double tol = 1e-7
    cdef double tnorm
    cdef double cdfU
    cdef double cdfL
    cdef double unif
    cdef double delta

    cdef floating[:] direction
    cdef floating[:] alpha

    cdef rk_state state
    rk_seed(seed, &state)

    for i in range(burnin + n):

        which_direction = <int> floor(rk_double(&state) * ndirections) # *(ndirections + 1)?
        direction = directions[which_direction,:]
        alpha = alphas[which_direction,:]

        value = 0.
        for j in range(p):
            value = value + direction[j] * X[j]

        for j in range(c):
            bound_val = -U[j] / alpha[j] + value

            if (alpha[j] > tol) and (bound_val < upper_bound):
                upper_bound = bound_val

            elif (alpha[j] < -tol) and (bound_val > lower_bound):
                lower_bound = bound_val

        if (lower_bound > value):
            lower_bound = value - tol

        elif (upper_bound < value):
            upper_bound = value + tol

        # step

        if upper_bound < -10:
            unif = rk_double(&state) * (1-exp(-fabs((lower_bound - upper_bound) * upper_bound)))
            tnorm = (upper_bound + log(1-unif) / fabs(lower_bound))

        elif lower_bound > 10:
            unif = rk_double(&state) * (1-exp(-fabs((upper_bound - lower_bound) * lower_bound)))
            tnorm = (lower_bound + log(1-unif) / lower_bound)

        elif lower_bound < 0:
            cdfL = pnorm(lower_bound)
            cdfU = pnorm(upper_bound)
            unif = rk_double(&state) * (cdfU - cdfL) + cdfL

            if unif < 0.5:
                tnorm =  qnorm(unif)
            else:
                tnorm = -qnorm(1-unif)

        else:
            cdfL = pnorm(-lower_bound)
            cdfU = pnorm(-upper_bound)
            unif = rk_double(&state) * (cdfL - cdfU) + cdfU

            if unif < 0.5:
                tnorm = -qnorm(unif)
            else:
                tnorm = qnorm(1-unif)

        delta = tnorm - value

        for j in range(p):
            X[j] = X[j] + delta * direction[j]

        for j in range(c):
            U[j] = U[j] + delta * alpha[j]

        # save result

        if i >= burnin:
            for j in range(p):
                result[j, i] = X[j]


