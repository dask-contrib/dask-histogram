import boost_histogram as bh
import boost_histogram.numpy as bhnp
import dask.array as da
import numpy as np

import dask_histogram as dh


def test_histogramdd():
    x = da.random.standard_normal(size=(3_000,), chunks=500)
    y = da.random.standard_normal(size=(3_000,), chunks=500)
    z = da.random.standard_normal(size=(3_000,), chunks=500)
    w = da.random.uniform(0.1, 0.5, size=(3_000,), chunks=500)
    bins = (4, 5, 6)
    range = ((-2.5, 2.5), (-3.5, 3.5), (-2, 2))
    h1 = dh.histogramdd(
        (x, y, z),
        bins=bins,
        range=range,
        weights=w,
        histogram=dh.Histogram,
        storage=dh.storage.Weight(),
    )
    h2 = bhnp.histogramdd(
        (x.compute(), y.compute(), z.compute()),
        bins=bins,
        range=range,
        weights=w,
        histogram=bh.Histogram,
        storage=bh.storage.Weight(),
    )
    h1 = h1.compute()
    np.testing.assert_array_almost_equal(h1.view()["value"], h2.view()["value"])
    np.testing.assert_array_almost_equal(h1.view()["variance"], h2.view()["variance"])


def test_histogramdd_multicolumn_input():
    x = da.random.standard_normal(size=(3_000, 3), chunks=(500, 3))
    bins = [[-3, -2, 2, 3], [-4, -1, 1, 4], [-2, -1, 1, 2]]
    h1 = dh.histogramdd(x, bins=bins, histogram=dh.Histogram)
    h2 = bhnp.histogramdd(x.compute(), bins=bins, histogram=bh.Histogram)
    h1 = h1.compute()
    np.testing.assert_array_almost_equal(h1.view(), h2.view())


def test_histogramdd_series():
    x = da.random.standard_normal(size=(1000,), chunks=(200,)).to_dask_dataframe()
    y = da.random.standard_normal(size=(1000,), chunks=(200,)).to_dask_dataframe()
    w = da.random.uniform(size=(1000,), chunks=(200,)).to_dask_dataframe()
    bins = (12, 12)
    range = ((-3, 3), (-2.6, 2.6))
    h1 = dh.histogramdd(
        (x, y),
        bins=bins,
        range=range,
        weights=w,
        histogram=dh.Histogram,
        storage=dh.storage.Weight(),
    )
    h1.compute()
    h2 = bhnp.histogramdd(
        (x.compute(), y.compute()),
        bins=bins,
        range=range,
        weights=w.compute(),
        histogram=bh.Histogram,
        storage=bh.storage.Weight(),
    )
    np.testing.assert_array_almost_equal(h1.view()["value"], h2.view()["value"])
    np.testing.assert_array_almost_equal(h1.view()["variance"], h2.view()["variance"])


def test_histogramdd_arrays_and_series():
    x = da.random.standard_normal(size=(1000,), chunks=(200,))
    y = da.random.standard_normal(size=(1000,), chunks=(200,)).to_dask_dataframe()
    w = da.random.uniform(size=(1000,), chunks=(200,))
    bins = (12, 12)
    range = ((-3, 3), (-2.6, 2.6))
    h1 = dh.histogramdd(
        (x, y),
        bins=bins,
        range=range,
        weights=w,
        histogram=dh.Histogram,
        storage=dh.storage.Weight(),
    )
    h1.compute()
    h2 = bhnp.histogramdd(
        (x.compute(), y.compute()),
        bins=bins,
        range=range,
        weights=w.compute(),
        histogram=bh.Histogram,
        storage=bh.storage.Weight(),
    )
    np.testing.assert_array_almost_equal(h1.view()["value"], h2.view()["value"])
    np.testing.assert_array_almost_equal(h1.view()["variance"], h2.view()["variance"])


def test_histogramdd_dataframe():
    x = da.random.standard_normal(size=(1000, 3), chunks=(200, 3))
    df = x.to_dask_dataframe(columns=["a", "b", "c"])
    w = da.random.uniform(size=(1000,), chunks=(200,))
    bins = [[-3, -2, 1, 2, 3], [-4, -1, 1, 2, 3, 4], [-3, -2, 0, 2]]
    h1 = dh.histogramdd(
        df,
        bins=bins,
        weights=w,
        histogram=dh.Histogram,
        storage=dh.storage.Weight(),
    )
    h1.compute()
    h2 = bhnp.histogramdd(
        df.compute().to_numpy(),
        bins=bins,
        weights=w.compute(),
        histogram=bh.Histogram,
        storage=bh.storage.Weight(),
    )
    np.testing.assert_array_almost_equal(h1.view()["value"], h2.view()["value"])
    np.testing.assert_array_almost_equal(h1.view()["variance"], h2.view()["variance"])


def test_histogram2d():
    x = da.random.standard_normal(size=(3_000,), chunks=500)
    y = da.random.standard_normal(size=(3_000,), chunks=500)
    w = da.random.uniform(0.1, 0.5, size=(3_000,), chunks=500)
    bins = (4, 6)
    range = ((-2.5, 2.5), (-2, 2))
    h1 = dh.histogram2d(
        x,
        y,
        bins=bins,
        range=range,
        weights=w,
        histogram=dh.Histogram,
        storage=dh.storage.Weight(),
    )
    h2 = bhnp.histogram2d(
        x.compute(),
        y.compute(),
        bins=bins,
        range=range,
        weights=w,
        histogram=bh.Histogram,
        storage=bh.storage.Weight(),
    )
    h1 = h1.compute()
    np.testing.assert_array_almost_equal(h1.view()["value"], h2.view()["value"])
    np.testing.assert_array_almost_equal(h1.view()["variance"], h2.view()["variance"])


def test_histogram():
    x = da.random.standard_normal(size=(3_000,), chunks=500)
    w = da.random.uniform(0.1, 0.5, size=(3_000,), chunks=500)
    bins = 7
    range = (-2.5, 2.5)
    h1 = dh.histogram(
        x,
        bins=bins,
        range=range,
        weights=w,
        histogram=dh.Histogram,
        storage=dh.storage.Weight(),
    )
    h2 = bhnp.histogram(
        x.compute(),
        bins=bins,
        range=range,
        weights=w,
        histogram=bh.Histogram,
        storage=bh.storage.Weight(),
    )
    h1 = h1.compute()
    np.testing.assert_array_almost_equal(h1.view()["value"], h2.view()["value"])
    np.testing.assert_array_almost_equal(h1.view()["variance"], h2.view()["variance"])


def test_histogramdd_da_return():
    x = da.random.standard_normal(size=(3_000,), chunks=500)
    y = da.random.standard_normal(size=(3_000,), chunks=500)
    z = da.random.standard_normal(size=(3_000,), chunks=500)
    w = da.random.uniform(0.1, 0.5, size=(3_000,), chunks=500)
    bins = (4, 5, 6)
    range = ((-2.5, 2.5), (-3.5, 3.5), (-2, 2))
    h1, edges1 = dh.histogramdd((x, y, z), bins=bins, range=range, weights=w)
    h2, edges2 = bhnp.histogramdd(
        (x.compute(), y.compute(), z.compute()), bins=bins, range=range, weights=w
    )
    np.testing.assert_array_almost_equal(h1.compute(), h2)
    for e1, e2 in zip(edges1, edges2):
        np.testing.assert_array_almost_equal(e1.compute(), e2)


def test_histogram2d_da_return():
    x = da.random.standard_normal(size=(3_000,), chunks=500)
    y = da.random.standard_normal(size=(3_000,), chunks=500)
    bins = (5, 6)
    range = ((-2.5, 2.5), (-3.5, 3.5))
    h1, edgesx1, edgesy1 = dh.histogram2d(x, y, bins=bins, range=range)
    h2, edgesx2, edgesy2 = bhnp.histogram2d(
        x.compute(), y.compute(), bins=bins, range=range
    )
    np.testing.assert_array_almost_equal(h1.compute(), h2)
    np.testing.assert_array_almost_equal(edgesx1.compute(), edgesx2)
    np.testing.assert_array_almost_equal(edgesy1.compute(), edgesy2)


def test_histogram_da_return():
    x = da.random.standard_normal(size=(3_000,), chunks=500)
    bins = np.array([-3, -2.2, 0, 1.1, 2.2, 3.3])
    h1, edges1 = dh.histogram(x, bins=bins)
    h2, edges2 = bhnp.histogram(x.compute(), bins=bins)
    np.testing.assert_array_almost_equal(h1.compute(), h2)
    np.testing.assert_array_almost_equal(edges1.compute(), edges2)
