from __future__ import annotations

from typing import Tuple

import boost_histogram as bh
import dask.array as da
import numpy as np

import dask_histogram.core as dhc
import dask_histogram.routines as dhr


def gen_hist_1D(
    bins: int = 10,
    range: Tuple[float, float] = (-3, 3),
    size: Tuple[int, ...] = (1000,),
) -> dhc.AggHistogram:
    hr = bh.Histogram(bh.axis.Regular(10, -3, 3), storage=bh.storage.Weight())
    x = da.random.standard_normal(size=(1000,), chunks=(250,))
    return dhc.histogram(x, histref=hr)


def test_histogram():
    x = da.random.standard_normal(size=(3000,), chunks=500)
    bins = np.array([-3, -2.2, 0, 1.1, 2.2, 3.3])
    h1, edges1 = dhr.histogram(x, bins=bins)
    h2, edges2 = bh.numpy.histogram(x.compute(), bins=bins)
    np.testing.assert_array_almost_equal(h1.compute(), h2)
    np.testing.assert_array_almost_equal(edges1.compute(), edges2)
