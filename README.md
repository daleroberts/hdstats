# hdstats

A library of multivariate, high-dimensional statistics, and time series algorithms for spatial-temporal stacks.

----

### Algorithms and methods

#### Geometric median PCM

Generation of geometric median pixel composite mosaics from a stack of data; see [example](https://github.com/daleroberts/hdstats/blob/master/docs/geomedian.ipynb).

#### Spectral (Geometric) Median Absolute Deviation (MAD)

Accelerated generation of spectral (geometric) median absolute deviation pixel composite mosaics from a stack of data; see [example](https://github.com/daleroberts/hdstats/blob/master/docs/mad.ipynb).

#### Euclidean (Geometric) Median Absolute Deviation (MAD)

Accelerated generation of Euclidean (geometric) median absolute deviation pixel composite mosaics from a stack of data; see [example](https://github.com/daleroberts/hdstats/blob/master/docs/mad.ipynb).

#### Bray-Curtis (Geometric) Median Absolute Deviation (MAD)

Accelerated generation of Bray-Curtis (geometric) median absolute deviation pixel composite mosaics from a stack of data; see [example](https://github.com/daleroberts/hdstats/blob/master/docs/mad.ipynb).

#### Feature generation for spatial-temporal time series stacks.

see [example](https://github.com/daleroberts/hdstats/blob/master/docs/temporal.ipynb).

---

### Assumptions

We assume that the data stack dimensions are ordered so that the spatial dimensions are first (*y*,*x*), followed by the spectral dimension of size *p*, finishing with the temporal dimension. Algorithms reduce in the last dimension (typically, the temporal dimension).

### Research and Development / Advanced Implementations

The advanced implementations and new research codes can now be found at [github.com/daleroberts/hdstats-private](https://github.com/daleroberts/hdstats-private).
