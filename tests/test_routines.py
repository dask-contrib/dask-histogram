from __future__ import annotations

import boost_histogram as bh
import dask.array as da
import numpy as np

import dask_histogram.routines as dhr


def test_histogram():
    x = da.random.standard_normal(size=(3000,), chunks=500)
    bins = np.array([-3, -2.2, 0, 1.1, 2.2, 3.3])
    h1, edges1 = dhr.histogram(x, bins=bins)
    h2, edges2 = bh.numpy.histogram(x.compute(), bins=bins)
    np.testing.assert_array_almost_equal(h1.compute(), h2)
    np.testing.assert_array_almost_equal(edges1.compute(), edges2)


def test_histogram2d():
    x = da.random.standard_normal(size=(3000,), chunks=500)
    y = da.random.standard_normal(size=(3000,), chunks=500)
    xbins = np.array([-3, -2.2, 0, 1.1, 2.2, 3.3])
    ybins = np.array([-4, -1.1, 0, 2.1, 2.2, 3.5])
    h1, edges1x, edges1y = dhr.histogram2d(x, y, bins=[xbins, ybins])
    h2, edges2x, edges2y = bh.numpy.histogram2d(
        x.compute(), y.compute(), bins=[xbins, ybins]
    )
    np.testing.assert_array_almost_equal(h1.compute(), h2)
    np.testing.assert_array_almost_equal(edges1x.compute(), edges2x)
    np.testing.assert_array_almost_equal(edges1y.compute(), edges2y)
