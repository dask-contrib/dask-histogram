# stdlib
from typing import Any, Optional

# boost-histogram and dask
import boost_histogram as bh
import dask.array as da
from dask.delayed import delayed, Delayed

# additional third party
import numpy as np


def blocked_fill(
    hist: bh.Histogram, data: np.ndarray, weight: Optional[np.ndarray] = None
) -> bh.Histogram:
    hist_for_block = bh.Histogram(*hist.axes, storage=hist._storage_type())
    hist_for_block.fill(data, weight=weight)
    return hist_for_block


def fill_graph(
    hist: bh.Histogram, data: da.Array, weight: Optional[da.Array] = None
) -> HighLevelGraph:
    d_data = data.to_delayed()
    d_histograms = [delayed(blocked_fill)(hist, a) for a in d_data]
    s = delayed(sum)(d_histograms)
    return s


h = bh.Histogram(bh.axis.Regular(50, -5.0, 5.0), storage=bh.storage.Weight())
x = da.random.standard_normal(size=(10_000_000,), chunks=1_000_000)
w = da.random.uniform(size=(10_000_000), chunks=x.chunksize[0])
delayed_hist = fill_graph(h, x, weight=w)
