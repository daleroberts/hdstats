import numpy.testing as npt
import numpy as np
import hdstats
import joblib
import pytest


class TestPixelCompositeMosaic:

    data = joblib.load('tests/landchar-small.pkl')

    def test_data(self):
        assert self.data.shape == (200, 200, 8, 18)
        assert self.data.dtype == np.float32

    def test_nangeomedian(self):
        gm = hdstats.nangeomedian_pcm(self.data)
        assert gm.shape == (200, 200, 8)


class TestMedianAbsoluteDeviation:

    data = joblib.load('tests/landchar-small.pkl')
    gm = hdstats.nangeomedian_pcm(data)

#    def test_data(self):
#        assert self.data.shape == (200, 200, 8, 18)
#        assert self.gm.shape == (200, 200, 8)

    def test_emad(self):
        emad = hdstats.emad_pcm(self.data, self.gm)
        assert emad.shape == (200, 200)

    def test_smad(self):
        smad = hdstats.smad_pcm(self.data, self.gm)
        assert smad.shape == (200, 200)
