# hdstats

A library of multivariate, high-dimensional statistics, and time series algorithms for spatial-temporal stacks.

----

### Algorithms and methods

#### Geometric median PCM

Generation of geometric median pixel composite mosaics from a stack of data; see [example](https://github.com/daleroberts/hdstats/blob/master/docs/geomedian.ipynb).

If you are using this algorithm in your research or products, please cite:

Roberts, D., Mueller, N., & McIntyre, A. (2017). High-dimensional pixel composites from earth observation time series. IEEE Transactions on Geoscience and Remote Sensing, 55(11), 6254-6264.

#### Geometric Median Absolute Deviation (MAD) PCM

Accelerated generation of geometric median absolute deviation pixel composite mosaics from a stack of data; see [example](https://github.com/daleroberts/hdstats/blob/master/docs/mad.ipynb).

If you are using this algorithm in your research or products, please cite:

Roberts, D., Dunn, B., & Mueller, N. (2018, July). Open data cube products using high-dimensional statistics of time series. In IGARSS 2018-2018 IEEE International Geoscience and Remote Sensing Symposium (pp. 8647-8650). IEEE.

#### Feature generation for spatial-temporal time series stacks.

see [example](https://github.com/daleroberts/hdstats/blob/master/docs/temporal.ipynb).

---

### Assumptions

We assume that the data stack dimensions are ordered so that the spatial dimensions are first (*y*,*x*), followed by the spectral dimension of size *p*, finishing with the temporal dimension. Algorithms reduce in the last dimension (typically, the temporal dimension).

### Research and Development / Advanced Implementations

All advanced implementations and cutting-edge research codes are now found in [github.com/daleroberts/hdstats-private](https://github.com/daleroberts/hdstats-private). These are only available to research collaborators.
