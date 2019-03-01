import numpy as np
import numpy.linalg as la

from scipy.stats import f as fisher_f
from scipy.stats import norm

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


METHODS = {
        'Hotelling': _test_sample_means_hotelling,
        'BaiSarandasa': _test_sample_means_bai_sarandasa,
        'ChenQin': _test_sample_means_chen_qin
}

def test_sample_means(x, y, method='Hotelling'):
    avail = list(METHODS.keys())
    if method not in avail:
        raise ValueError(f'Incorrect method, available methods are: {avail}')
    return METHODS[method](x,y)

