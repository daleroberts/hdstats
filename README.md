# hdstats

A library of multivariate and high-dimensional statistics algorithms.

The mission of this library is to provide algorithms for multivariate data where: (1) the dimensionality *p* is relatively large (with respect to the sample size *n*), (2) there might be missing data, (3) the data might be spatially or temporally stacked and can we use this knowledge to accelerate or enhance the algorithm. This library is aimed to completement what is available in numpy, scipy, scikit-learn, and scikit-image.

### Assumptions

We assume that the data stack dimensions are ordered so that the spatial dimensions are first (*y*,*x*), followed by the spectral dimension of size *p*, finishing with the temporal dimension. Algorithms reduce in the last dimension (typically, the temporal dimension).

### Algorithms and methods

#### Geometric median PCM

Accelerated generation of geometric median pixel composite mosaics from a stack of data; see [example](https://github.com/daleroberts/hdstats/blob/master/docs/geomedian.ipynb).

#### Weighted geometric median PCM

Accelerated generation of weighted geometric median pixel composite mosaics from a stack of data; see [example](https://github.com/daleroberts/hdstats/blob/master/docs/wgeomedian.ipynb).

#### Spectral (Geometric) Median Absolute Deviation (MAD)

Accelerated generation of spectral (geometric) median absolute deviation pixel composite mosaics from a stack of data; see [example](https://github.com/daleroberts/hdstats/blob/master/docs/mad.ipynb).

#### Euclidean (Geometric) Median Absolute Deviation (MAD)

Accelerated generation of Euclidean (geometric) median absolute deviation pixel composite mosaics from a stack of data; see [example](https://github.com/daleroberts/hdstats/blob/master/docs/mad.ipynb).

#### Generalised Variance

Generalised variance of multivariate observations; see [example](https://github.com/daleroberts/hdstats/blob/master/docs/mvn.ipynb).

#### Multivariate Gaussian distribution

Multivariate Gaussian distribution; see [example](https://github.com/daleroberts/hdstats/blob/master/docs/mvn.ipynb).

#### Wishart distribution

The Wishart distribution; see [example](https://github.com/daleroberts/hdstats/blob/master/docs/wishart.ipynb).

#### Empirical spectral distribution

The empirical spectral distribution; see [example](https://github.com/daleroberts/hdstats/blob/master/docs/mp.ipynb).

#### Marchenko-Pastur density

The Marchenko-Pastur distribution; see [example](https://github.com/daleroberts/hdstats/blob/master/docs/mp.ipynb).

#### Generalised Marchenko-Pastur density

The generalised Marchenko-Pastur distribution; see [example](https://github.com/daleroberts/hdstats/blob/master/docs/mp.ipynb).

#### Two-sample tests of mean

Various tests of mean between two samples; see [example](https://github.com/daleroberts/hdstats/blob/master/docs/twosample.ipynb).

#### Two-sample test of geometric median

A robust test of geometric median between two samples; see [example](https://github.com/daleroberts/hdstats/blob/master/docs/twosample.ipynb).

#### Two-sample tests of covariance

Various tests of covariances between two samples; see [example](https://github.com/daleroberts/hdstats/blob/master/docs/twosample.ipynb).

#### Constrained Gaussian distribution

Constrained multivariate Gaussian distribution; see [example](https://github.com/daleroberts/hdstats/blob/master/docs/constrained.ipynb).
