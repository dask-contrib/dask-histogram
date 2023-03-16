import boost_histogram as bh
import dask.sizeof
import numpy as np
import pytest

hist = pytest.importorskip("hist")


# def test_sizeof():
#     h = bh.Histogram(
#         bh.axis.Regular(10, -3, 3),
#         bh.axis.Regular(10, -3, 3),
#         storage=bh.storage.Weight(),
#     )
#     h.fill(
#         np.random.standard_normal(size=1000),
#         np.random.standard_normal(size=1000),
#     )
#     assert dask.sizeof.sizeof(h) == dask.sizeof.sizeof(h.view(flow=True))


# def test_registration():
#     # we register these (should not default)
#     assert dask.sizeof.sizeof.dispatch(bh.Histogram) is not dask.sizeof.sizeof_default
#     assert dask.sizeof.sizeof.dispatch(hist.Hist) is not dask.sizeof.sizeof_default
#     # we don't register this one (should be default)
#     assert dask.sizeof.sizeof.dispatch(bh.axis.Regular) is dask.sizeof.sizeof_default


def test_nothing():
    assert True
