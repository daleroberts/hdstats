import numpy.testing as npt
import numpy as np
import pytest

from hdstats import MarchenkoPastur


class TestMarchenkoPastur:
    @pytest.mark.parametrize(
        "y,sigma,expected", [
            (0.2, 1.0, np.array([0., 0., 0., 0., 0.795775, 0.886137, 0.879762, 0.843089, \
                                 0.795775, 0.745035, 0.69374, 0.643, 0.593135, 0.544077, 0.495529, \
                                 0.447021, 0.397887, 0.347154, 0.293254, 0.233194, 0.159155, 0., 0., \
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])),
            (0.4, 1.0, np.array([0., 0., 0.795775, 0.828269, 0.770506, 0.7073, 0.649747, \
                                 0.598858, 0.553836, 0.51367, 0.477465, 0.444484, 0.414134, 0.385936, \
                                0.359494, 0.334478, 0.310601, 0.287607, 0.265258, 0.243318, 0.221534, \
                                0.199619, 0.177204, 0.153761, 0.128418, 0.0993922, 0.0612134, 0., 0., \
                                0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
        ])
    def test_pdf(self, y, sigma, expected):
        mp = MarchenkoPastur(y, sigma)
        x = np.arange(0, 4.1, 0.1)
        result = mp.pdf(x)
        npt.assert_array_almost_equal(result, expected, decimal=6)
