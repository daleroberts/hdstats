import numpy as np

from scipy.stats import rv_continuous

from scipy.stats import wishart as Wishart
from scipy.stats import chi2 as ChiSquared
from scipy.stats import multivariate_normal as MultivariateNormal

# class Wishart(wishart_gen):
#
#    def __init__(self, df=None, scale=None, seed=None):
#        super().__init__(df, scale, seed)
#
#    def rvs(self, df, scale, size=1, random_state=None):
#        result = super().rvs(df, scale, size, random_state)
#        return result.T


class MarchenkoPastur_gen(rv_continuous):
    """ Construct a Marchenko-Pastur distribution. """

    def _pdf(self, x, y, sigma):
        err = np.seterr(all="ignore")
        a = np.power(sigma * (1 - np.sqrt(y)), 2)
        b = np.power(sigma * (1 + np.sqrt(y)), 2)
        result = np.where(
            np.logical_and(a < x, x < b),
            np.sqrt((b - x) * (x - a)) / (2 * np.pi * sigma * sigma * x * y),
            np.zeros_like(x),
        )
        np.seterr(**err)
        return result

    def _cdf(self, x, y, sigma):
        a = np.power(sigma * (1 - np.sqrt(y)), 2)
        b = np.power(sigma * (1 + np.sqrt(y)), 2)
        d = np.sqrt(-(-1 + x / sigma ** 2 - y) ** 2 + 4 * y)

        if 0 < y < 1:
            result = np.where(
                np.logical_and(a < x, x < b),
                (1 / (2 * np.pi * y)) * (np.pi * y + d + 4 * y)
                - (1 + y) * np.arctan((1 - x / (sigma ** 2) + y) / d)
                - (1 - y)
                * np.arctan(
                    (-(-1 + y) ** 2 + (x * (1 + y)) / sigma ** 2) / ((1 - y) * d)
                ),
                np.zeros_like(x),
            )
            result[x > b] = 1.0
            return result

        elif y == 1:
            e = np.sqrt((x * (4 - x / sigma ** 2)) / sigma ** 2)
            result = np.where(
                x < 4 * sigma ** 2,
                (np.pi + e - 2 * np.arctan((2 - x / sigma ** 2) / e)) / (2 * np.pi),
                np.ones_like(x),
            )
            result[x <= 0] = 0
            return result

        elif y > 1:
            result = np.where(
                np.logical_and(a <= x, x <= b),
                1
                + (1 / (2 * np.pi * y))(
                    -np.pi
                    + d
                    - (1 + y) * np.arctan((1 - x / sigma ** 2 + y) / d)
                    - (-1 + y)
                    * np.arctan(
                        (-(-1 + y) ** 2 + (x * (1 + y)) / sigma ** 2) / ((-1 + y) * d)
                    )
                ),
                np.ones_like(x),
            )
            result[np.logical_and(0 <= x, x <= a)] = 1 - 1 / y
            result[x <= 0] = 0.0
            result[x >= b] = 1.0
            return result

        return np.zeros_like(x)


MarchenkoPastur = MarchenkoPastur_gen(name="mp", longname="Marchenko-Pastur")

