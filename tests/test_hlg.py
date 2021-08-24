import boost_histogram as bh
import dask.array as da
import dask.array.utils as dau
import dask.datasets as dds
import numpy as np
import pytest

from dask_histogram.hlg import histogram


@pytest.mark.parametrize("weights", [True, None])
def test_1d_array(weights):
    h = bh.Histogram(bh.axis.Regular(10, -3, 3), storage=bh.storage.Weight())
    if weights is not None:
        weights = da.random.uniform(size=(2000,), chunks=(250,))
    x = da.random.standard_normal(size=(2000,), chunks=(250,))
    dh = histogram(x, histref=h, weights=weights, split_every=4)
    h.fill(x.compute(), weight=weights.compute() if weights is not None else None)
    np.testing.assert_allclose(h.counts(flow=True), dh.compute().counts(flow=True))


@pytest.mark.parametrize("weights", [True, None])
@pytest.mark.parametrize("shape", ((2000,), (2000, 2), (2000, 3), (2000, 4)))
def test_array_input(weights, shape):
    oned = len(shape) < 2
    x = da.random.standard_normal(
        size=shape, chunks=(200,) if oned else (200, shape[1])
    )
    xc = (x.compute(),) if oned else x.compute().T
    ax = bh.axis.Regular(10, -3, 3)
    axes = (ax,) if oned else (ax,) * shape[1]
    weights = (
        da.random.uniform(size=(2000,), chunks=(200,)) if weights is not None else None
    )
    h = bh.Histogram(*axes, storage=bh.storage.Weight())
    dh = histogram(x, histref=h, weights=weights, split_every=4)
    h.fill(*xc, weight=weights.compute() if weights is not None else None)
    np.testing.assert_allclose(h.counts(flow=True), dh.compute().counts(flow=True))


@pytest.mark.parametrize("weights", [True, None])
def test_multi_array(weights):
    h = bh.Histogram(
        bh.axis.Regular(10, -3, 3),
        bh.axis.Regular(10, -3, 3),
        storage=bh.storage.Weight(),
    )
    if weights is not None:
        weights = da.random.uniform(size=(2000,), chunks=(250,))
    x = da.random.standard_normal(size=(2000,), chunks=(250,))
    y = da.random.standard_normal(size=(2000,), chunks=(250,))
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
        weights = da.random.uniform(0, 1, size=(2000,), chunks=(250,))
    x = da.random.uniform(0, 1, size=(2000, 3), chunks=(250, 3))
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
    df = dds.timeseries(freq="600s", partition_freq="2d")
    dfc = df.compute()
    if weights is not None:
        weights = da.fabs(df["y"].to_dask_array())
    df = df[["x", "y"]]
    dh = histogram(df, histref=h, weights=weights, split_every=200)
    h.fill(
        *(dfc[["x", "y"]].to_numpy().T),
        weight=weights.compute() if weights is not None else None,
    )
    np.testing.assert_allclose(h.counts(flow=True), dh.compute().counts(flow=True))


@pytest.mark.parametrize("weights", [True, None])
@pytest.mark.parametrize("shape", ((2000,), (2000, 2), (2000, 3), (2000, 4)))
def test_to_dask_array(weights, shape):
    oned = len(shape) < 2
    x = da.random.standard_normal(
        size=shape, chunks=(200,) if oned else (200, shape[1])
    )
    xc = (x.compute(),) if oned else x.compute().T
    ax = bh.axis.Regular(10, -3, 3)
    axes = (ax,) if oned else (ax,) * shape[1]
    weights = (
        da.random.uniform(size=(2000,), chunks=(200,)) if weights is not None else None
    )
    h = bh.Histogram(*axes, storage=bh.storage.Weight())
    dh = histogram(x, histref=h, weights=weights, split_every=4)
    h.fill(*xc, weight=weights.compute() if weights is not None else None)
    c, edges = dh.to_dask_array(flow=False, dd=True)
    dau.assert_eq(c, h.to_numpy()[0])


def test_bop_add():
    ax = bh.axis.Regular(10, -3, 3)
    x = da.random.standard_normal(size=(1000,), chunks=(250,))
    y = da.random.standard_normal(size=(1000,), chunks=(250,))
    h = bh.Histogram(ax)
    dhx = histogram(x, histref=h)
    dhy = histogram(y, histref=h)
    chx = bh.Histogram(ax).fill(x.compute())
    chy = bh.Histogram(ax).fill(y.compute())
    chz = chx + chy
    dhz = dhx + dhy
    dau.assert_eq(chz.to_numpy()[0], dhz.to_dask_array()[0])


def test_bop_div():
    ax = bh.axis.Regular(10, -3, 3)
    x = da.random.standard_normal(size=(1000,), chunks=(250,))
    y = da.random.standard_normal(size=(1000,), chunks=(250,))
    h = bh.Histogram(ax)
    dhx = histogram(x, histref=h)
    dhy = histogram(y, histref=h)
    chx = bh.Histogram(ax).fill(x.compute())
    chy = bh.Histogram(ax).fill(y.compute())
    dhz = dhx / dhy
    chz = chx / chy
    dau.assert_eq(chz.to_numpy()[0], dhz.to_dask_array()[0])
    chx /= chy
    dhx /= dhy
    dau.assert_eq(dhx.to_dask_array()[0], dhz.to_dask_array()[0])
    dau.assert_eq(chx.to_numpy()[0], dhx.to_dask_array()[0])
