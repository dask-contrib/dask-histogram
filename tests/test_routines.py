import dask.array as da
import boost_histogram.numpy as bhnp
import boost_histogram as bh
import numpy as np

import dask_histogram as dh


def test_histogramdd():
    x = da.random.standard_normal(size=(10_000,), chunks=2_000)
    y = da.random.standard_normal(size=(10_000,), chunks=2_000)
    z = da.random.standard_normal(size=(10_000,), chunks=2_000)
    w = da.random.uniform(0.1, 0.5, size=(10_000,), chunks=2_000)
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
    np.testing.assert_allclose(h1.view()["value"], h2.view()["value"])
    np.testing.assert_allclose(h1.view()["variance"], h2.view()["variance"])


def test_histogram2d():
    x = da.random.standard_normal(size=(10_000,), chunks=2_000)
    y = da.random.standard_normal(size=(10_000,), chunks=2_000)
    w = da.random.uniform(0.1, 0.5, size=(10_000,), chunks=2_000)
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
    np.testing.assert_allclose(h1.view()["value"], h2.view()["value"])
    np.testing.assert_allclose(h1.view()["variance"], h2.view()["variance"])


def test_histogram():
    x = da.random.standard_normal(size=(10_000,), chunks=2_000)
    w = da.random.uniform(0.1, 0.5, size=(10_000,), chunks=2_000)
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
    np.testing.assert_allclose(h1.view()["value"], h2.view()["value"])
    np.testing.assert_allclose(h1.view()["variance"], h2.view()["variance"])
