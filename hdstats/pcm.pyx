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

def __gm(floating [:, :, :, :] X, floating [:, :, :] mX, floating [:] w, 
               size_t maxiters=10000, floating eps=1e-6, num_threads=None):
    """ """
    cdef size_t m = X.shape[0]
    cdef size_t q = X.shape[1]
    cdef size_t p = X.shape[2]
    cdef size_t n = X.shape[3]
    
    cdef size_t i, j, k, l, row, col
    cdef size_t nzeros, iteration
    cdef size_t reseed = 0
    cdef float64_t dist, Dinvs, total, r, rinv, tmp, Di, d, value
    cdef float64_t nan = <float64_t> np.nan
    cdef floating *D
    cdef floating *Dinv
    cdef floating *W
    cdef floating *T
    cdef floating *y
    cdef floating *y1
    cdef floating *R

    cdef int number_of_threads

    if num_threads is None:
        number_of_threads = openmp.omp_get_max_threads()
    else:
        number_of_threads = num_threads
    
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

def gm(floating [:, :, :, :] X, weight=None, maxiters=1000, floating eps=1e-4, num_threads=None):
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

    Returns
    -------
    m : ndarray
        The array has dimensions (m, q, p)
    """
    cdef size_t m = X.shape[0]
    cdef size_t q = X.shape[1]
    cdef size_t p = X.shape[2]
    cdef size_t n = X.shape[3]
    
    if floating is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64
        
    if weight is None:
        w = np.ones((n,), dtype=dtype)
    else:
        w = np.array(weight, dtype=dtype)

    result = np.empty((m, q, p), dtype=dtype)
    
    __gm(X, result, w, maxiters=maxiters, eps=eps, num_threads=num_threads)
    
    return result

def __wgm(floating [:, :, :, :] X, floating [:, :, :] mX,
              size_t bi, size_t bj, floating rho, floating delta,
              floating [:] alpha, floating [:] gamma, 
              floating [:] beta, floating [:] sigma, size_t maxiters=10000, 
              floating eps=1e-6, num_threads=None):
    """ """
    
    cdef size_t m = X.shape[0]
    cdef size_t q = X.shape[1]
    cdef size_t p = X.shape[2]
    cdef size_t n = X.shape[3]
    
    cdef size_t i, j, k, l, row, col
    cdef size_t nzeros, iteration
    cdef size_t reseed = 0
    cdef float64_t dist, Dinvs, total, r, rinv, tmp, Di, d, value, numer, denom, min_weight, max_weight
    cdef float64_t nan = <float64_t> np.nan
    cdef floating *D
    cdef floating *Dinv
    cdef floating *W
    cdef floating *T
    cdef floating *y
    cdef floating *y1
    cdef floating *R
    cdef floating *w
    
    cdef int number_of_threads

    if num_threads is None:
        number_of_threads = openmp.omp_get_max_threads()
    else:
        number_of_threads = num_threads
    
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
                    value = (X[row, col, bi, i] - X[row, col, bj, i])/(X[row, col, bi, i] + X[row, col, bj, i])
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

def wgm(floating [:, :, :, :] X, bi, bj, rho=1.0, delta=1.0, alpha=None, 
        gamma=None, beta=None, sigma=None, maxiters=1000, floating eps=1e-4,
        num_threads=None):
    """
    Generate a weighted geometric median pixel composite mosaic by reducing along 
    the last axis. Weighting is performed on each multivariate time series based
    on a spectral feature model that give a relative ranking of observations at 
    the particular pixel.

    Generate simple weight features based on the spectral information in the 
    pixel $\mathbb{x} = (x_1, \ldots, x_p)^T$. These features have the form
    $$
        f(\mathbb{x}) = \\frac{x_i-x_j}{x_i+x_j}\left(\\frac{\\alpha_1(x_1-\gamma_1)+ 
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
    cdef size_t m = X.shape[0]
    cdef size_t q = X.shape[1]
    cdef size_t p = X.shape[2]
    cdef size_t n = X.shape[3]
    
    if floating is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64
        
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
        
    __wgm(X, result, bi, bj, rho, delta, alpha, gamma, beta, sigma, maxiters=maxiters, eps=eps, num_threads=num_threads)
    
    return result

def __emad(floating [:, :, :, :] X, floating [:, :, :] gm, floating [:,:,:] result, num_threads=None):
    cdef size_t m = X.shape[0]
    cdef size_t q = X.shape[1]
    cdef size_t p = X.shape[2]
    cdef size_t n = X.shape[3]
    
    cdef float64_t total, value
    cdef size_t j, t, row, col

    cdef int number_of_threads

    if num_threads is None:
        number_of_threads = openmp.omp_get_max_threads()
    else:
        number_of_threads = num_threads
    
    with nogil, parallel(num_threads=number_of_threads):
        for row in prange(m, schedule='static'):
            for col in range(q):
                for t in range(n):

                    # euclidean distance
                    total = 0.
                    for j in range(p):
                        value = X[row, col, j, t] - gm[row, col, j]
                        if not isnan(value):
                            total = total + value*value

                    result[row, col, t] = sqrt(total)
            

def emad(floating [:, :, :, :] X, floating [:,:,:] gm, num_threads=None):
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
    cdef size_t m = X.shape[0]
    cdef size_t q = X.shape[1]
    cdef size_t p = X.shape[2]
    cdef size_t n = X.shape[3]
    
    if floating is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    result = np.empty((m, q, n), dtype=dtype)
    
    __emad(X, gm, result, num_threads=num_threads)
    
    return np.median(result, axis=2)


def __smad(floating [:, :, :, :] X, floating [:, :, :] gm, floating [:,:,:] result, num_threads=None):
    """ """
    cdef size_t m = X.shape[0]
    cdef size_t q = X.shape[1]
    cdef size_t p = X.shape[2]
    cdef size_t n = X.shape[3]
    
    cdef float64_t numer, norma, normb, value
    cdef size_t j, t, row, col

    cdef int number_of_threads

    if num_threads is None:
        number_of_threads = openmp.omp_get_max_threads()
    else:
        number_of_threads = num_threads
    
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
                    

def smad(floating [:, :, :, :] X, floating [:,:,:] gm, num_threads=None):
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
    cdef size_t m = X.shape[0]
    cdef size_t q = X.shape[1]
    cdef size_t p = X.shape[2]
    cdef size_t n = X.shape[3]
    
    if floating is cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64

    result = np.empty((m, q, n), dtype=dtype)
    
    __smad(X, gm, result, num_threads=num_threads)
    
    return np.nanmedian(result, axis=2)
