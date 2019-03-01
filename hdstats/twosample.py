import numpy as np
import numpy.linalg as la
import scipy.linalg as sla

from scipy.stats import f as fisher_f
from scipy.stats import norm
from scipy.stats import chi2

from .geomedian import geomedian, nangeomedian

def _hotelling_test_statistic(x, y):
    nx = x.shape[0]
    ny = y.shape[0]
    px = x.shape[1]
    py = y.shape[1]
    p = px
    assert nx + ny > px - 1
    mx = np.mean(x, axis=0)
    my = np.mean(y, axis=0)
    sx = np.cov(x, rowvar=False)
    sy = np.cov(y, rowvar=False)
    sp = ((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2)  # pooled
    spi = la.inv(sp)
    T2 = np.dot(np.dot(mx - my, spi), mx - my) * nx * ny / (nx + ny)
    m = (nx + ny - p - 1) / (p * (nx + ny - 2))
    return {
        "statistic": T2,
        "m": m,
        "df": (p, nx + ny - p - 1),
        "nx": nx,
        "ny": ny,
        "p": p,
    }


def _test_sample_means_hotelling(x, y):
    s = _hotelling_test_statistic(x, y)
    d1, d2 = s["df"]
    dist = fisher_f(d1, d2, 0)
    statistic = s["m"] * s["statistic"]
    pvalue = 1.0 - dist.cdf(statistic)
    return (statistic, pvalue)


def _test_sample_means_bai_sarandasa(x, y):
    n1, n2 = x.shape[0], y.shape[0]
    tau = (n1 * n2) / (n1 + n2)
    n = n1 + n2 - 2
    p = x.shape[1]

    # sum of square distance of means
    sqdiff = np.sum(np.square(np.mean(x, axis=0) - np.mean(y, axis=0)))

    S = (
        (n1 - 1) * np.cov(x, rowvar=False) + (n2 - 1) * np.cov(y, rowvar=False)
    ) / n

    trS = np.trace(S)

    trS2 = n ** 2 / ((n + 2) * (n - 1)) * (np.sum(np.square(S)) - np.square(trS) / n)

    test_stat = (tau * sqdiff - trS) / np.sqrt(2 * (n + 1) / n * trS2)

    pval = 1.0 - norm.cdf(test_stat)

    return (test_stat, pval)


def _test_sample_means_chen_qin(x, y):
    n1, n2 = x.shape[0], y.shape[0]
    tau = (n1 * n2) / (n1 + n2)
    n = n1 + n2 - 2
    p = x.shape[1]

    S = (
        (n1 - 1) * np.cov(x, rowvar=False) + (n2 - 1) * np.cov(y, rowvar=False)
    ) / n

    trS = np.trace(S)

    trS2 = n ** 2 / ((n + 2) * (n - 1)) * (np.sum(np.square(S)) - np.square(trS) / n)

    T1 = np.dot(x, x.T)
    T2 = np.dot(y, y.T)

    P1 = (np.sum(T1) - np.trace(T1)) / (n1 * (n1 - 1))
    P2 = (np.sum(T2) - np.trace(T2)) / (n2 * (n2 - 1))
    P3 = -2 * np.sum(np.dot(x, y.T)) / (n1 * n2)

    T = P1 + P2 + P3

    test_stat = T / np.sqrt(
        (2.0 / (n1 * (n1 - 1)) + 2.0 / (n2 * (n2 - 1)) + 4 / (n1 * n2)) * trS2
    )

    pval = 1.0 - norm.cdf(test_stat)

    return (test_stat, pval)


MEAN_METHODS = {
        'Hotelling': _test_sample_means_hotelling,
        'BaiSarandasa': _test_sample_means_bai_sarandasa,
        'ChenQin': _test_sample_means_chen_qin
}

def test_sample_means(x, y, method='Hotelling'):
    avail = list(MEAN_METHODS.keys())
    if method not in avail:
        raise ValueError(f'Incorrect method, available methods are: {avail}')
    return MEAN_METHODS[method](x,y)


def _test_sample_geomedian_rublik_somorcik(x,y):
    def pairdiff(X):
        n, p = X.shape
        D = np.empty((int(n*(n-1)/2), p))
        k = 0
        for i in range(n-1):
            for j in range(i+1, n):
                D[k,:] = X[i,:]-X[j,:]
                k = k + 1
        return D
    
    def SSCov(X):
        D = pairdiff(X)
        Ds = D/la.norm(D, axis=1).reshape((-1,1))
        return np.einsum('ij,ik->jk', Ds, Ds) / (X.shape[0]*(X.shape[0]-1)/2)
    
    def shape_det(a):
        return a/(la.det(a)**(1/a.shape[1]))
    
    def duembgen_shape(X, maxiters=1000, init=None):
        n,p = X.shape
        if init is None:
            init = np.cov(X, rowvar=False)
        init = shape_det(init)
        it = 0
        V = init.copy()
        while it < maxiters:
            sqrtV = sla.sqrtm(V)
            V1 = sqrtV @ SSCov(X @ la.inv(sqrtV)) @ sqrtV
            V1 = shape_det(V1)
            if la.norm(V1-V) < 0.0001:
                break
            V = V1.copy()
            it = it + 1
        #print('iters: {}'.format(it))
        return V1
    
    def U(x):
        return x/la.norm(x)
    
    def P(x):
        nrm = la.norm(x)
        I = np.identity(len(x))
        return (I - np.outer(x, x)/(nrm**2))/nrm
    
    
    X = np.vstack([x, y])
    n = np.array([x.shape[0], y.shape[0]])
    W = duembgen_shape(X)
    Wsqrt = sla.sqrtm(W)
    Y = X @ la.inv(Wsqrt)
    eta = []
    cns = np.cumsum(n)
    for j in range(len(n)):
        si, ei = cns[j] - n[j], cns[j]
        gm = np.array(geomedian(Y[si:ei], axis=0))
        eta.append(Wsqrt @ gm)
    eta = np.stack(eta)
    Ygm = np.array(geomedian(Y, axis=0))
    
    def U(x):
        return x/la.norm(x)
    
    def P(x):
        nrm = la.norm(x)
        I = np.identity(len(x))
        return (I - np.outer(x, x)/(nrm**2))/nrm
    
    D1 = np.mean(np.apply_along_axis(P, 1, Y-Ygm), axis=0)
    D2 = np.cov(np.apply_along_axis(U, 1, Y-Ygm), rowvar=False, bias=True)
    invB = la.inv(Wsqrt) @ D1 @ la.inv(D2) @ D1 @ la.inv(Wsqrt)
    A1 = n @ eta / np.sum(n)
    A2 = Wsqrt @ Ygm
    
    dist = chi2((n.shape[0]-1)*X.shape[1])
    loc = A2
    A = 0
    for j in range(n.shape[0]):
        A = A + n[j]*(eta[j]-loc).T @ invB @ (eta[j]-loc)

    pvalue = 1. - dist.cdf(A)

    return (A, pvalue)


MEDIAN_METHODS = {
        'RublikSomorcik': _test_sample_geomedian_rublik_somorcik,
}

def test_sample_geomedians(x, y, method='RublikSomorcik'):
    avail = list(MEDIAN_METHODS.keys())
    if method not in avail:
        raise ValueError(f'Incorrect method, available methods are: {avail}')
    return MEDIAN_METHODS[method](x,y)
