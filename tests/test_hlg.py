import boost_histogram as bh
import dask.array as da
import dask.datasets as dds
import numpy as np
import pytest

from dask_histogram.hlg import histogram


@pytest.mark.parametrize("weights", [True, None])
def test_1d_array(weights):
    h = bh.Histogram(bh.axis.Regular(10, -3, 3), storage=bh.storage.Weight())
    if weights is not None:
        weights = da.random.uniform(size=(2000,), chunks=(400,))
    x = da.random.standard_normal(size=(2000,), chunks=(400,))
    dh = histogram(x, histref=h, weights=weights, split_every=4)
    h.fill(x.compute(), weight=weights.compute() if weights is not None else None)
    np.testing.assert_allclose(h.counts(flow=True), dh.compute().counts(flow=True))


@pytest.mark.parametrize("weights", [True, None])
def test_multi_array(weights):
    h = bh.Histogram(
        bh.axis.Regular(10, -3, 3),
        bh.axis.Regular(10, -3, 3),
        storage=bh.storage.Weight(),
    )
    if weights is not None:
        weights = da.random.uniform(size=(2000,), chunks=(400,))
    x = da.random.standard_normal(size=(2000,), chunks=(400,))
    y = da.random.standard_normal(size=(2000,), chunks=(400,))
    dh = histogram(x, y, histref=h, weights=weights, split_every=4)
    h.fill(
        x.compute(),
        y.compute(),
        weight=weights.compute() if weights is not None else None,
    )
    np.testing.assert_allclose(h.counts(flow=True), dh.compute().counts(flow=True))


@pytest.mark.parametrize("weights", [True, None])
def test_nd_array(weights):
    h = bh.Histogram(
        bh.axis.Regular(10, 0, 1),
        bh.axis.Regular(10, 0, 1),
        bh.axis.Regular(10, 0, 1),
        storage=bh.storage.Weight(),
    )
    if weights is not None:
        weights = da.random.uniform(0, 1, size=(2000,), chunks=(400,))
    x = da.random.uniform(0, 1, size=(2000, 3), chunks=(400, 3))
    dh = histogram(x, histref=h, weights=weights, split_every=4)
    h.fill(
        *(x.compute().T),
        weight=weights.compute() if weights is not None else None,
    )
    np.testing.assert_allclose(h.counts(flow=True), dh.compute().counts(flow=True))


@pytest.mark.parametrize("weights", [True, None])
def test_df_input(weights):
    h = bh.Histogram(
        bh.axis.Regular(12, 0, 1),
        bh.axis.Regular(12, 0, 1),
        storage=bh.storage.Weight(),
    )
    df = dds.timeseries()
    if weights is not None:
        weights = da.fabs(df["y"].to_dask_array())
    df = df[["x", "y"]]
    dh = histogram(df, histref=h, weights=weights, split_every=100)
    h.fill(
        *(df.compute().to_numpy().T),
        weight=weights.compute() if weights is not None else None,
    )
    np.testing.assert_allclose(h.counts(flow=True), dh.compute().counts(flow=True))
