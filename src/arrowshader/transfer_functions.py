# Transfer functions ported from datashader (https://github.com/holoviz/datashader)
# Copyright (c) 2015, Continuum Analytics, Inc. and contributors. Licensed under BSD-3-Clause.

import numpy as np


def eq_hist(data, mask=None, nbins=256 * 256):
    if mask is not None and np.all(mask):
        return np.full_like(data, np.nan), 0

    data2 = data if mask is None else data[~mask]
    if data2.dtype == bool or (np.issubdtype(data2.dtype, np.integer) and np.ptp(data2) < nbins):
        values, counts = np.unique(data2, return_counts=True)
        vmin, vmax = values[0].item(), values[-1].item()  # Convert from arrays to scalars.
        interval = vmax - vmin
        bin_centers = np.arange(vmin, vmax + 1)
        hist = np.zeros(interval + 1, dtype="uint64")
        hist[values - vmin] = counts
        discrete_levels = len(values)
    else:
        hist, bin_edges = np.histogram(data2, bins=nbins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        keep_mask = hist > 0
        discrete_levels = np.count_nonzero(keep_mask)
        if discrete_levels != len(hist):
            hist = hist[keep_mask]
            bin_centers = bin_centers[keep_mask]
    cdf = hist.cumsum()
    cdf = cdf / float(cdf[-1])
    out = np.interp(data, bin_centers, cdf).reshape(data.shape)
    return out if mask is None else (np.where(mask, np.nan, out), discrete_levels)


_interpolate_lookup = {
    "log": lambda d, m: np.log1p(np.where(m, np.nan, d)),
    "cbrt": lambda d, m: np.where(m, np.nan, d) ** (1 / 3.0),
    "linear": lambda d, m: np.where(m, np.nan, d),
    "eq_hist": eq_hist,
}


def _normalize_interpolate_how(how):
    if callable(how):
        return how
    elif how in _interpolate_lookup:
        return _interpolate_lookup[how]
    raise ValueError(f"Unknown interpolation method: {how}")
