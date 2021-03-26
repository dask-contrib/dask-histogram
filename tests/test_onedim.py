import boost_histogram as bh
import dask.array as da
from dask_histogram.core import fill_1d


def test_simple_weighted():
    h = bh.Histogram(bh.axis.Regular(12, -3.0, 3.0))
    x = da.random.standard_normal(size=(200,), chunks=50)
    delayed_hist = fill_1d(x, h)
    result = delayed_hist.compute()

    x = x.compute()
    h = bh.Histogram(bh.axis.Regular(12, -3.0, 3.0))
    h.fill(x)

    assert h.sum() == result.sum()


# def test_simple_weighted():
#     h = bh.Histogram(bh.axis.Regular(12, -3.0, 3.0), storage=bh.storage.Weight())
#     x = da.random.standard_normal(size=(200,), chunks=50)
#     w = da.random.uniform(size=(200,), chunks=x.chunksize[0])
#     delayed_hist = fill_1d(h, x, weight=w)
#     result = delayed_hist.compute()

#     x = x.compute()
#     w = w.compute()
#     h = bh.Histogram(bh.axis.Regular(12, -3.0, 3.0), storage=bh.storage.Weight())
#     h.fill(x, weight=w)
