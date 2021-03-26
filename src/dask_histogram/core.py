# stdlib
from typing import Any, Optional

# boost-histogram and dask
import boost_histogram as bh
import dask.array as da
from dask.delayed import delayed, Delayed

# additional third party
import numpy as np


def blocked_fill(
    data: np.ndarray, hist: bh.Histogram, weight: Optional[np.ndarray] = None
) -> bh.Histogram:
    hist_for_block = bh.Histogram(*hist.axes, storage=hist._storage_type())
    hist_for_block.fill(data, weight=weight)
    return hist_for_block


def fill_1d(
    data: da.Array, hist: bh.Histogram, weight: Optional[da.Array] = None
) -> Delayed:
    d_data = data.to_delayed()
    d_histograms = [delayed(blocked_fill)(a, hist) for a in d_data]
    s = delayed(sum)(d_histograms)
    return s
