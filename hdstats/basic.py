def nancov(m, y=None, rowvar=1, ddof=1, fweights=None, aweights=None, pairwise=False):

    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be integer")

    m = np.asarray(m)
    if m.ndim > 2:
        raise ValueError("m has more than 2 dimensions")

    if y is None:
        dtype = np.result_type(m, np.float64)
    else:
        y = np.asarray(y)
        if y.ndim > 2:
            raise ValueError("y has more than 2 dimensions")
        dtype = np.result_type(m, y, np.float64)
    X = np.array(m, ndmin=2, dtype=dtype)

    if X.shape[0] == 1:
        rowvar = 1
    if rowvar:
        axis = 0
    else:
        axis = 1

    if y is not None:
        y = np.array(y, copy=False, ndmin=2, dtype=dtype)
        X = np.concatenate((X, y), axis)

    if axis:
        n = X.shape[0]
    else:
        n = X.shape[1]

    # Get the product of frequencies and weights
    if fweights is not None:
        fweights = np.asarray(fweights, dtype=np.float64)
        if not np.all(fweights == np.around(fweights)):
            raise TypeError("fweights must be integer")
        if fweights.ndim > 1:
            raise RuntimeError("cannot handle multidimensional fweights")
        if fweights.shape[0] != n:
            raise RuntimeError("incompatible numbers of samples and fweights")
        if any(fweights < 0):
            raise ValueError("fweights cannot be negative")

    if aweights is not None:
        aweights = np.asarray(aweights, dtype=np.float64)
        if aweights.ndim > 1:
            raise RuntimeError("cannot handle multidimensional aweights")
        if aweights.shape[0] != n:
            raise RuntimeError("incompatible numbers of samples and aweights")
        if any(aweights < 0):
            raise ValueError("aweights cannot be negative")

    if pairwise:
        w = None
        if fweights is not None:
            w = fweights
        if aweights is not None:
            if w is None:
                w = aweights
            else:
                w *= aweights

        nan_vals = np.isnan(X)
        if axis:
            one_array = np.ones(X.T.shape)
            one_array[nan_vals.T] = np.nan
        else:
            one_array = np.ones(X.shape)
            one_array[nan_vals] = np.nan
        # each pair may cause a unique mean for the variable, so we create
        # pair_array which has the correctly nan-ed values.
        if axis:
            pair_array = X[:, None, :].swapaxes(0, 2) * one_array[None, :, :]
        else:
            pair_array = X[:, None, :] * one_array[None, :, :]

        if w is None:
            pair_means = nanmean(pair_array, axis=2)
            pair_w_sum = nansum(~np.isnan(pair_array), axis=2)
        else:
            pair_w = w[None, None, :] * ~np.isnan(pair_array)
            pair_w_sum = np.nansum(pair_w, axis=2)
            pair_means = nansum(w[None, None, :] * pair_array, axis=2) / pair_w_sum
        pair_array -= pair_means[:, :, None]

        if w is None:
            dotted = pair_array * pair_array.swapaxes(0, 1).conj()
        else:
            dotted = pair_array * (w[None, None, :] * pair_array).swapaxes(0, 1).conj()

        c = nansum(dotted, axis=2)

        if aweights is None:
            fact = pair_w_sum - ddof
        else:
            ddof_weight = nansum(aweights[None, None, :] * pair_w, axis=2)
            fact = pair_w_sum - ddof * ddof_weight / pair_w_sum

        if np.any(fact <= 0):
            warnings.warn("Degrees of freedom <= 0 for a slice", RuntimeWarning, stacklevel=2)
            fact[fact <= 0] = 0.0

        c *= 1.0 / fact.astype(np.float64)
        return c

    else:
        # "Complete" version for handling nans where a nan value in any
        # variable causes the observation to be removed from all variables.

        # Find observations with nans
        nan_obvs = np.any(np.isnan(X), axis=axis)

        if fweights is not None:
            fweights = fweights[~nan_obvs]
        if aweights is not None:
            aweights = aweights[~nan_obvs]

        if axis:
            X_nonan = X[~nan_obvs, :]
        else:
            X_nonan = X[:, ~nan_obvs]

        return np.cov(X_nonan, rowvar=rowvar, ddof=ddof, fweights=fweights, aweights=aweights)
