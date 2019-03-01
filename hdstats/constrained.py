import numpy as np

from numpy.linalg import svd, norm
from numpy.random import randn as rnorm, rand as runif

from .truncated import sample_truncated_normal, pnorm, qnorm

def factor_covariance(S, rank=None) -> tuple:
    if rank is None:
        rank = S.shape[0]

    u, d, v = svd(S, full_matrices=False)

    # Q st. S = Q.T @ Q
    sqrt_cov = (np.sqrt(d[:rank]) * v[:rank].T).T
    sqrt_inv = (1.0 / np.sqrt(d[:rank]) * v[:rank].T).T

    return (sqrt_cov, sqrt_inv)

def whiten_constraint(A, b, mean, covariance):
    sqrt_cov, sqrt_inv = factor_covariance(covariance)

    new_A = A @ sqrt_cov
    new_b = b - A @ mean

    scaling = np.sum(np.square(new_A), axis=1)
    new_A /= scaling[:,np.newaxis]
    new_b /= scaling[:,np.newaxis]

    def inverse_map(Z):
        return sqrt_cov @ Z + mean

    def forward_map(W):
        return sqrt_inv @ (W - mean)

    return (new_A, new_b, inverse_map, forward_map)

def sample_constrained_gaussian(n, A, b, mean, covariance, initial_point=None, burnin=2000, seed=1):
    r, p = A.shape

    if initial_point is None:
        initial_point = np.minimum(A @ mean, b) # good guess? project mean onto constraint

    assert b.shape[0] == r
    assert mean.shape[0] == p
    assert covariance.shape == (p, p)
    assert initial_point.shape[0] == p

    white_A, white_b, inverse_map, forward_map = whiten_constraint(A, b, mean, covariance)

    print(white_A.shape)
    print(white_b.shape)

    white_initial = forward_map(initial_point)
    assert white_initial.shape[0] == p

    good_rows = np.isfinite(white_b)
    white_A = white_A[good_rows]
    white_b = white_b[good_rows]

    c = white_A.shape[0]

    if c > 0:

        if p > 1:
            # multivariate

            c = white_A.shape[0]
            directions = np.vstack(np.identity(p), rnorm(p, p))
            assert directions.shape == (2 * p, p)

            scaling = norm(directions, 2, axis=1)
            directions /= scaling[:,np.newaxis]
            
            ndirections = directions.shape[0]
            assert ndirections == 2 * p

            alphas = directions @ white_A.T
            assert alphas.shape == (2*p, c)

            U = white_A @ white_initial - white_b
            assert U.shape[0] == c

            result = np.zeros((p, n))

            sample_truncated_normal(n, white_initial, U, directions, alphas, result, burnin, seed)

        else: # univariate

            pos = white_A > 0
            neg = white_A < 0

            if np.sum(pos) > 0:
                U = np.min((white_b / white_A)[pos])
            else:
                U = np.inf

            if np.sum(neg) > 0:
                L = np.max((white_b / white_A)[neg])
            else:
                L = -np.inf

            result = qnorm((pnorm(U) - pnorm(L)) * runif(n) + pnorm(L)).reshape((1, n))

    else:
        result = rnorm(p, n)

    return inverse_map(result).T

def thresholds_to_constraints(p, lower=None, upper=None):
    if lower is None:
        lower = -np.inf * np.ones(p, dtype=np.float)

    if upper is None:
        upper = np.inf * np.ones(p, dtype=np.float)

    lower = np.asarray(lower)
    upper = np.asarray(upper)

    A = np.empty((0, p), dtype=np.float)
    b = []

    lower_constraints = np.argwhere(lower > -np.inf)
    for ell in lower_constraints:
        newrow = np.zeros((1, p))
        newrow[0, ell] = -1
        A = np.vstack([A, newrow])
        b.append(-lower[ell])

    upper_constraints = np.argwhere(upper > -np.inf)
    for ell in upper_constraints:
        newrow = np.zeros((1, p))
        newrow[0, ell] = -1
        A = np.vstack([A, newrow])
        b.append(-upper[ell])

    b = np.array(b)

    return (A,b)


