import boost_histogram as bh
import dask.array as da
import numpy as np

import dask_histogram as dh
from dask_histogram.boost import fill_nd


def test_simple():
    h = bh.Histogram(bh.axis.Regular(12, -3.0, 3.0))
    x = da.random.standard_normal(size=(200,), chunks=50)
    delayed_hist = fill_nd(x, hist=h)
    result = delayed_hist.compute()

    x = x.compute()
    h = bh.Histogram(bh.axis.Regular(12, -3.0, 3.0))
    h.fill(x)

    assert np.allclose(h.counts(), result.counts())
    assert h.sum() == result.sum()


def test_simple_flow():
    h = bh.Histogram(
        bh.axis.Regular(20, -3.2, 3.2),
        bh.axis.Regular(20, -2.9, 2.9),
    )
    x = da.random.standard_normal(size=(100_000, 2), chunks=5_000)
    h.fill(*x.compute().T)

    delayed_hist = fill_nd(*x.T, hist=h)
    result = delayed_hist.compute()

    assert np.allclose(h.counts(flow=True), result.counts(flow=True))


def test_simple_weighted():
    h = bh.Histogram(bh.axis.Regular(12, -3.0, 3.0), storage=bh.storage.Weight())
    x = da.random.standard_normal(size=(200,), chunks=50)
    w = da.random.uniform(0.2, 0.5, size=x.shape, chunks=x.chunksize[0])
    delayed_hist = fill_nd(x, hist=h, weight=w)
    result = delayed_hist.compute()

    x = x.compute()
    w = w.compute()
    h = bh.Histogram(bh.axis.Regular(12, -3.0, 3.0), storage=bh.storage.Weight())
    h.fill(x, weight=w)

    assert repr(h.sum()) == repr(result.sum())


def test_simple_nd():
    h = bh.Histogram(
        bh.axis.Regular(20, -3.5, 3.5),
        bh.axis.Regular(25, -3.2, 3.2),
        bh.axis.Regular(12, 0.4, 0.6),
        storage=bh.storage.Weight(),
    )
    x = da.random.standard_normal(size=(200,), chunks=25)
    y = da.random.standard_normal(size=(200,), chunks=25)
    z = da.random.uniform(0.4, 0.6, size=(200,), chunks=25)
    w = da.random.uniform(0.2, 1.1, size=(200,), chunks=25)

    delayed_hist = fill_nd(x, y, z, hist=h, weight=w)
    result = delayed_hist.compute()

    x = x.compute()
    y = y.compute()
    z = z.compute()
    w = w.compute()
    h = bh.Histogram(*h.axes, storage=bh.storage.Weight())
    h.fill(x, y, z, weight=w)

    assert repr(h.sum()) == repr(result.sum())


def test_class():
    h = dh.Histogram(
        dh.axis.Regular(20, -3.5, 3.5),
        dh.axis.Regular(25, -3.2, 3.2),
        dh.axis.Regular(12, 0.4, 0.6),
        storage=dh.storage.Weight(),
    )
    x = da.random.standard_normal(size=(200,), chunks=25)
    y = da.random.standard_normal(size=(200,), chunks=25)
    z = da.random.uniform(0.4, 0.6, size=(200,), chunks=25)
    w = da.random.uniform(0.2, 1.1, size=(200,), chunks=25)
    h.fill(x, y, z, weight=w)
    h.compute()

    h2 = bh.Histogram(
        bh.axis.Regular(20, -3.5, 3.5),
        bh.axis.Regular(25, -3.2, 3.2),
        bh.axis.Regular(12, 0.4, 0.6),
        storage=bh.storage.Weight(),
    )
    h2.fill(x.compute(), y.compute(), z.compute(), weight=w.compute())
    assert repr(h) == repr(h2)
    assert np.allclose(h.counts(flow=True), h2.counts(flow=True))
    assert np.allclose(h.variances(flow=True), h2.variances(flow=True))
    assert np.allclose(h.counts(), h2.counts())
    assert np.allclose(h.variances(), h2.variances())


def test_class_2():
    h = dh.Histogram(
        bh.axis.Regular(20, -3.5, 3.5),
        bh.axis.Regular(25, -3.2, 3.2),
        bh.axis.Regular(12, 0.4, 0.6),
        storage=bh.storage.Weight(),
    )
    x = da.random.standard_normal(size=(200,), chunks=25)
    y = da.random.standard_normal(size=(200,), chunks=25)
    z = da.random.uniform(0.4, 0.6, size=(200,), chunks=25)
    w = da.random.uniform(0.2, 1.1, size=(200,), chunks=25)
    h.fill(x, y, z, weight=w)
    h.fill(x, y, z, weight=w)
    h.compute()
    h.fill(x, y, z, weight=w)
    h.compute()

    h2 = bh.Histogram(
        bh.axis.Regular(20, -3.5, 3.5),
        bh.axis.Regular(25, -3.2, 3.2),
        bh.axis.Regular(12, 0.4, 0.6),
        storage=bh.storage.Weight(),
    )
    h2.fill(x.compute(), y.compute(), z.compute(), weight=w.compute())
    h2.fill(x.compute(), y.compute(), z.compute(), weight=w.compute())
    h2.fill(x.compute(), y.compute(), z.compute(), weight=w.compute())

    assert repr(h) == repr(h2)
    assert np.allclose(h.counts(), h2.counts())
    assert np.allclose(h.variances(flow=True), h2.variances(flow=True))
