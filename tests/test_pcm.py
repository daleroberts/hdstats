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

    def test_nangeomedian_baddata(self):
        baddata = self.data[:3,:3,:,:].copy()
        baddata[1,1,0,:] = np.nan
        gm = hdstats.nangeomedian_pcm(baddata)
        assert np.isnan(gm[1,1,0])

class TestMedianAbsoluteDeviation:

    data = joblib.load('tests/landchar-small.pkl')
    gm = hdstats.nangeomedian_pcm(data)

    def test_emad(self):
        emad = hdstats.emad_pcm(self.data, self.gm)
        assert emad.shape == (200, 200)

    def test_emad_baddata(self):
        baddata = self.data[:3,:3,:,:].copy()
        baddata[1,1,0,:] = np.nan
        emad = hdstats.emad_pcm(baddata, self.gm)
        print(emad.shape)
        assert np.isnan(emad[1,1])

    def test_smad(self):
        smad = hdstats.smad_pcm(self.data, self.gm)
        assert smad.shape == (200, 200)

    def test_smad_baddata(self):
        baddata = self.data[:3,:3,:,:].copy()
        baddata[1,1,0,:] = np.nan
        smad = hdstats.smad_pcm(baddata, self.gm)
        assert np.isnan(smad[1,1])

    def test_bcmad(self):
        bcmad = hdstats.smad_pcm(self.data, self.gm)
        assert bcmad.shape == (200, 200)

    def test_bcmad_baddata(self):
        baddata = self.data[:3,:3,:,:].copy()
        baddata[1,1,0,:] = np.nan
        bcmad = hdstats.smad_pcm(baddata, self.gm)
        assert np.isnan(bcmad[1,1])