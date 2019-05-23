import numpy.testing as npt
import numpy as np
import joblib
import pytest

from hdstats import nangeomedian_pcm

class TestPixelCompositeMosaic:

    data = joblib.load('tests/landchar-small.pkl')

    def test_data(self):
        assert self.data.shape == (200, 200, 8, 18)

    def test_nangeomedian(self):
        gm = nangeomedian_pcm(self.data)
        assert gm.shape == (200, 200, 8)