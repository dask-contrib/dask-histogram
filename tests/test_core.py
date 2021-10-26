from __future__ import annotations

import boost_histogram as bh
import dask.array as da
import dask.array.utils as dau
import dask.datasets as dds
import dask.delayed as delayed
import numpy as np
import pytest

import dask_histogram.core as dhc


def _gen_storage(weights, sample):
    if weights is not None and sample is not None:
        store = bh.storage.WeightedMean()
    elif weights is None and sample is not None:
        store = bh.storage.Mean()
    elif weights is not None and sample is None:
        store = bh.storage.Weight()
    else:
        store = bh.storage.Double()
    return store


@pytest.mark.parametrize("weights", [True, None])
@pytest.mark.parametrize("sample", [True, None])
def test_1d_array(weights, sample):
    if weights is not None:
        weights = da.random.uniform(size=(2000,), chunks=(250,))
    if sample is not None:
        sample = da.random.uniform(2, 8, size=(2000,), chunks=(250,))
    store = _gen_storage(weights, sample)
    h = bh.Histogram(bh.axis.Regular(10, -3, 3), storage=store)
    x = da.random.standard_normal(size=(2000,), chunks=(250,))
    dh = dhc.factory(x, histref=h, weights=weights, split_every=4, sample=sample)
    h.fill(
        x.compute(),
        weight=weights.compute() if weights is not None else None,
        sample=sample.compute() if sample is not None else None,
    )
    np.testing.assert_allclose(h.counts(flow=True), dh.compute().counts(flow=True))


@pytest.mark.parametrize("weights", [True, None])
@pytest.mark.parametrize("shape", ((2000,), (2000, 2), (2000, 3), (2000, 4)))
@pytest.mark.parametrize("sample", [True, None])
def test_array_input(weights, shape, sample):
    oned = len(shape) < 2
    x = da.random.standard_normal(
        size=shape, chunks=(200,) if oned else (200, shape[1])
    )
    xc = (x.compute(),) if oned else x.compute().T
    ax = bh.axis.Regular(10, -3, 3)
    axes = (ax,) if oned else (ax,) * shape[1]
    if weights:
        weights = da.random.uniform(size=(2000,), chunks=(200,))
    if sample:
        sample = da.random.uniform(3, 9, size=(2000,), chunks=(200,))
    store = _gen_storage(weights, sample)
    h = bh.Histogram(*axes, storage=store)
    dh = dhc.factory(x, histref=h, weights=weights, split_every=4, sample=sample)
    h.fill(
        *xc,
        weight=weights.compute() if weights is not None else None,
        sample=sample.compute() if sample is not None else None,
    )
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
    dh = dhc.factory(x, y, histref=h, weights=weights, split_every=4)
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
    dh = dhc.factory(x, histref=h, weights=weights, split_every=4)
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
    dh = dhc.factory(df, histref=h, weights=weights, split_every=200)
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
    dh = dhc.factory(x, histref=h, weights=weights, split_every=4)
    h.fill(*xc, weight=weights.compute() if weights is not None else None)
    c, _ = dh.to_dask_array(flow=False, dd=True)
    dau.assert_eq(c, h.to_numpy()[0])


def gen_hist_1D(
    bins: int = 10,
    range: tuple[float, float] = (-3, 3),
    size: tuple[int, ...] = (1000,),
    chunks: tuple[int, ...] = (250,),
) -> dhc.AggHistogram:
    hr = bh.Histogram(
        bh.axis.Regular(bins, range[0], range[1]),
        storage=bh.storage.Weight(),
    )
    x = da.random.standard_normal(size=size, chunks=chunks)
    return dhc.factory(x, histref=hr)


@delayed
def get_number(n):
    return n


@delayed
def get_array(size):
    return np.arange(size)


@pytest.mark.parametrize("other", [get_number(5), get_array(10)])
def test_add(other):
    h = gen_hist_1D()
    concrete = other.compute()
    computed_array = (h.compute() + concrete).to_numpy()[0]

    ht = (h + other).to_dask_array()[0]
    dau.assert_eq(ht, computed_array)

    ht = (h + concrete).to_dask_array()[0]
    dau.assert_eq(ht, computed_array)

    h = gen_hist_1D()
    computed_array = (h.compute() + concrete).to_numpy()[0]
    h += concrete
    ht = h.to_dask_array()[0]
    dau.assert_eq(ht, computed_array)

    h = gen_hist_1D()
    computed_array = (h.compute() + concrete).to_numpy()[0]
    h += other
    ht = h.to_dask_array()[0]
    dau.assert_eq(ht, computed_array)


@pytest.mark.parametrize("other", [get_number(5), get_array(10)])
def test_sub(other):
    h = gen_hist_1D()
    concrete = other.compute()
    computed_array = (h.compute() - concrete).to_numpy()[0]

    ht = (h - other).to_dask_array()[0]
    dau.assert_eq(ht, computed_array)

    ht = (h - concrete).to_dask_array()[0]
    dau.assert_eq(ht, computed_array)

    h = gen_hist_1D()
    computed_array = (h.compute() - concrete).to_numpy()[0]
    h -= concrete
    ht = h.to_dask_array()[0]
    dau.assert_eq(ht, computed_array)

    h = gen_hist_1D()
    computed_array = (h.compute() - concrete).to_numpy()[0]
    h -= other
    ht = h.to_dask_array()[0]
    dau.assert_eq(ht, computed_array)


@pytest.mark.parametrize("other", [get_number(5), get_array(10)])
def test_mul(other):
    h = gen_hist_1D()
    concrete = other.compute()
    computed_array = (h.compute() * concrete).to_numpy()[0]

    ht = (h * other).to_dask_array()[0]
    dau.assert_eq(ht, computed_array)

    ht = (h * concrete).to_dask_array()[0]
    dau.assert_eq(ht, computed_array)

    h = gen_hist_1D()
    computed_array = (h.compute() * concrete).to_numpy()[0]
    h *= concrete
    ht = h.to_dask_array()[0]
    dau.assert_eq(ht, computed_array)

    h = gen_hist_1D()
    computed_array = (h.compute() * concrete).to_numpy()[0]
    h *= other
    ht = h.to_dask_array()[0]
    dau.assert_eq(ht, computed_array)


@pytest.mark.parametrize("other", [get_number(5), get_array(10)])
def test_div(other):
    other += 1
    h = gen_hist_1D()
    concrete = other.compute()
    computed_array = (h.compute() / concrete).to_numpy()[0]

    ht = (h / other).to_dask_array()[0]
    dau.assert_eq(ht, computed_array)

    ht = (h / concrete).to_dask_array()[0]
    dau.assert_eq(ht, computed_array)

    h = gen_hist_1D()
    computed_array = (h.compute() / concrete).to_numpy()[0]
    h /= concrete
    ht = h.to_dask_array()[0]
    dau.assert_eq(ht, computed_array)

    h = gen_hist_1D()
    computed_array = (h.compute() / concrete).to_numpy()[0]
    h /= other
    ht = h.to_dask_array()[0]
    dau.assert_eq(ht, computed_array)


def test_bad_weight_structure():
    x = da.random.standard_normal(size=(100,), chunks=(20,))
    w = da.random.standard_normal(size=(80,), chunks=(20,))
    with pytest.raises(
        ValueError, match="weights must have as many partitions as the data."
    ):
        dhc.factory(x, axes=(bh.axis.Regular(10, -3, 3),), weights=w)
    w = da.random.standard_normal(size=(80, 3), chunks=(20, 3))
    with pytest.raises(ValueError, match="weights must be one dimensional."):
        dhc.factory(x, axes=(bh.axis.Regular(10, -3, 3),), weights=w)
