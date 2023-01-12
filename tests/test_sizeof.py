import boost_histogram as bh
import dask.sizeof
import numpy as np


def test_sizeof():
    h = bh.Histogram(
        bh.axis.Regular(10, -3, 3),
        bh.axis.Regular(10, -3, 3),
        storage=bh.storage.Weight(),
    )
    h.fill(
        np.random.standard_normal(size=1000),
        np.random.standard_normal(size=1000),
    )
    assert dask.sizeof.sizeof(h) == dask.sizeof.sizeof(h.view(flow=True))


def test_registration():
    assert "hist" in dask.sizeof.sizeof._lazy
    assert bh.Histogram in dask.sizeof.sizeof._lookup
