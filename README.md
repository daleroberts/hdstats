# hdstats

A library of multivariate and high-dimensional statistics algorithms.

The mission of this library is to provide algorithms for multivariate data where: (1) the dimensionality *p* is relatively large (with respect to the sample size *n*), (2) there might be missing data, (3) the data might be spatially or temporally stacked and can we use this knowledge to accelerate or enhance the algorithm. This library is aimed to completement what is available in numpy, scipy, scikit-learn, and scikit-image.

### Assumptions

We assume that the data stack dimensions are ordered so that the spatial dimensions are first (*y*,*x*), followed by the spectral dimension of size *p*, finishing with the temporal dimension. Algorithms reduce in the last dimension. 

### Algorithms

#### Geometric median PCM

Accelerated generation of geometric median pixel composite mosaics from a stack of data; see [example](https://github.com/daleroberts/hdstats/blob/master/docs/geomedian.ipynb).

#### Weighted geometric median

#### Spectral (Geometric) Median Absolute Deviation (MAD)

#### Euclidean (Geometric) Median Absolute Deviation (MAD)


#### Generalised Variance



#### Empirical spectral distribution


#### Two-sample tests of mean

#### Two-sample test of geometric median

#### Two-sample tests of covariance


#### Multivariate Gaussian distribution

#### Wishart distribution

#### Marchenko-Pastur density

#### Generalised Marchenko-Pastur density

#### Constrained Gaussian distribution

