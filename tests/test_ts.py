import numpy.testing as npt
import numpy as np
import hdstats
import joblib
import pytest

class TestTimeSeriesFeatures:

    data = joblib.load('tests/landchar-small.pkl').astype(np.float32)
    ndvi = (data[:,:,4,:]-data[:,:,3,:])/(data[:,:,4,:]+data[:,:,3,:])
    gm = hdstats.nangeomedian_pcm(data)

    def test_data(self):
        assert self.data.shape == (200, 200, 8, 18)

    def test_cosdist(self):
        cd = hdstats.cosdist(self.data, self.gm)
        assert cd.shape == (200, 200, 18)

    def test_eucdist(self):
        ed = hdstats.eucdist(self.data, self.gm)
        assert ed.shape == (200, 200, 18)

    def test_completion(self):
        completed = hdstats.completion(self.data, 1.0)
        assert np.count_nonzero(np.isnan(completed)) == 0

    def test_discordance(self):
        dc = hdstats.discordance(self.ndvi)
        assert dc.shape == (200, 200)

    def test_fourier_mean(self):
        fs = hdstats.fourier_mean(self.ndvi)
        assert fs.shape == (200, 200, 3)

    def test_fourier_std(self):
        fs = hdstats.fourier_std(self.ndvi)
        assert fs.shape == (200, 200, 3)

    def test_fourier_median(self):
        fs = hdstats.fourier_median(self.ndvi)
        assert fs.shape == (200, 200, 3)

    def test_mean_change(self):
        mc = hdstats.mean_change(self.ndvi)
        assert mc.shape == (200, 200)
    
    def test_median_change(self):
        mc = hdstats.median_change(self.ndvi)
        assert mc.shape == (200, 200)

    def test_mean_central_diff(self):
        mcd = hdstats.mean_central_diff(self.ndvi)
        assert mcd.shape == (200, 200)

    def test_complexity(self):
        cplx = hdstats.complexity(self.ndvi)
        assert cplx.shape == (200, 200)

    # def test_number_peaks(self):
    #     mndvi = np.mean(self.ndvi, axis=(0,1))
    #     peaks = hdstats.number_peaks(mndvi, 20)
    #     assert len(peaks) == 1

    def test_symmetry(self):
        sym = hdstats.symmetry(self.data)
        assert sym.shape == (200, 200)

    def test_area_warp_similarity(self):
        sim = hdstats.area_warp_similarity(self.ndvi)
        assert sim.shape == (200, 200)
