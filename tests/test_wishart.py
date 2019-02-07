import numpy.testing as npt
import numpy as np
import pytest

#from hdstats.wishart import sample_wishart

#class TestWishartSampling:
#
#    @pytest.mark.parametrize(
#        "n,p,expected", [
#            (1, 3, (3,3,1)),
#            (10, 3, (3,3,10))
#        ])
#    def test_shape(self, n, p, expected):
#        Sigma = np.eye(p)
#        X = sample_wishart(n, 3, Sigma)
#        assert X.shape == expected
#
#    def test_sigma_intact(self):
#        Sigma0 = np.eye(3)
#        Sigma1 = Sigma0.copy()
#        X = sample_wishart(3, 3, Sigma1)
#        npt.assert_array_equal(Sigma0.flat, Sigma1.flat)
#
#    def test_large(self, n=10000, p=1000):
#        Sigma = np.eye(p)
#        X = sample_wishart(1, n, Sigma)
#        assert X.shape == (p, p, 1)
