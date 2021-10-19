import boost_histogram as bh
import boost_histogram.numpy as bhnp
import dask.array as da
import numpy as np
import pytest

import dask_histogram.boost as dhb
import dask_histogram.core as dhc


@pytest.mark.parametrize("use_weights", [True, False])
def test_obj_1D(use_weights):
    x = da.random.standard_normal(size=(2000,), chunks=(400,))
    if use_weights:
        weights = da.random.uniform(0.5, 0.75, size=x.shape, chunks=x.chunks)
        storage = dhb.storage.Weight()
    else:
        weights = None
        storage = dhb.storage.Double()

    h = dhb.Histogram(dhb.axis.Regular(12, -3, 3), storage=storage)
    h.fill(x, weight=weights)
    h.compute()

    control = bh.Histogram(*h.axes, storage=h._storage_type())
    if use_weights:
        control.fill(x.compute(), weight=weights.compute())
    else:
        control.fill(x.compute())

    assert np.allclose(h.counts(), control.counts())
    if use_weights:
        assert np.allclose(h.variances(), control.variances())


@pytest.mark.parametrize("use_weights", [True, False])
def test_obj_2D(use_weights):
    x = da.random.standard_normal(size=(2000,), chunks=(400,))
    y = da.random.standard_normal(size=(2000,), chunks=(400,))
    if use_weights:
        weights = da.random.uniform(0.5, 0.75, size=x.shape, chunks=x.chunks)
        storage = dhb.storage.Weight()
    else:
        weights = None
        storage = dhb.storage.Double()

    h = dhb.Histogram(
        dhb.axis.Regular(9, -3.5, 3.5),
        dhb.axis.Regular(9, -3.2, 3.2),
        storage=storage,
    )
    h.fill(x, y, weight=weights)
    h.compute()

    control = bh.Histogram(*h.axes, storage=h._storage_type())
    if use_weights:
        control.fill(x.compute(), y.compute(), weight=weights.compute())
    else:
        control.fill(x.compute(), y.compute())

    assert np.allclose(h.counts(), control.counts())
    if use_weights:
        assert np.allclose(h.variances(), control.variances())


@pytest.mark.parametrize("use_weights", [True, False])
def test_obj_3D_rectangular(use_weights):
    x = da.random.standard_normal(size=(2000, 3), chunks=(400, 3))
    if use_weights:
        weights = da.random.uniform(0.5, 0.75, size=x.shape[0], chunks=x.chunksize[0])
        storage = dhb.storage.Weight()
    else:
        weights = None
        storage = dhb.storage.Double()

    h = dhb.Histogram(
        dhb.axis.Regular(8, -3.5, 3.5),
        dhb.axis.Regular(7, -3.3, 3.3),
        dhb.axis.Regular(9, -3.2, 3.2),
        storage=storage,
    )
    h.fill(x, weight=weights)
    h.compute()

    control = bh.Histogram(*h.axes, storage=h._storage_type())
    if use_weights:
        control.fill(*(x.compute().T), weight=weights.compute())
    else:
        control.fill(*(x.compute().T))

    assert np.allclose(h.counts(), control.counts())
    if use_weights:
        assert np.allclose(h.variances(), control.variances())


def test_clear_fills():
    x = da.random.standard_normal(size=(8, 2), chunks=(4, 2))
    h = dhb.Histogram(
        dhb.axis.Regular(8, -3.5, 3.5),
        dhb.axis.Regular(9, -3.2, 3.2),
    )
    h.fill(x)
    assert h.staged_fills()
    h.clear_fills()
    assert not h.staged_fills()


def test_concrete_fill():
    x = da.random.standard_normal(size=(500, 2), chunks=(4, 2))
    h = dhb.Histogram(
        dhb.axis.Regular(8, -3.5, 3.5),
        dhb.axis.Regular(9, -3.2, 3.2),
    )

    h.fill(x)
    material_x = x.compute()
    h.fill(*material_x.T)
    h.compute()

    h2 = bh.Histogram(*h.axes)
    h2.fill(*material_x.T)
    h2.fill(*material_x.T)

    np.testing.assert_array_almost_equal(h2.values(), h.values())

    with pytest.raises(
        TypeError,
        match="concrete_fill does not support Dask collections, "
        "only materialized data; use the Histogram.fill method.",
    ):
        h.concrete_fill(x)


def test_histogramdd():
    x = da.random.standard_normal(size=(3_000,), chunks=500)
    y = da.random.standard_normal(size=(3_000,), chunks=500)
    z = da.random.standard_normal(size=(3_000,), chunks=500)
    w = da.random.uniform(0.1, 0.5, size=(3_000,), chunks=500)
    bins = (4, 5, 6)
    range = ((-2.5, 2.5), (-3.5, 3.5), (-2, 2))
    h1 = dhb.histogramdd(
        (x, y, z),
        bins=bins,
        range=range,
        weights=w,
        histogram=dhb.Histogram,
        storage=dhb.storage.Weight(),
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
    h1 = dhb.histogramdd(x, bins=bins, histogram=dhb.Histogram)
    h2 = bhnp.histogramdd(x.compute(), bins=bins, histogram=bh.Histogram)
    h1 = h1.compute()
    np.testing.assert_array_almost_equal(h1.view(), h2.view())


def test_histogramdd_series():
    x = da.random.standard_normal(size=(1000,), chunks=(200,)).to_dask_dataframe()
    y = da.random.standard_normal(size=(1000,), chunks=(200,)).to_dask_dataframe()
    w = da.random.uniform(size=(1000,), chunks=(200,)).to_dask_dataframe()
    bins = (12, 12)
    range = ((-3, 3), (-2.6, 2.6))
    h1 = dhb.histogramdd(
        (x, y),
        bins=bins,
        range=range,
        weights=w,
        histogram=dhb.Histogram,
        storage=dhb.storage.Weight(),
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
    h1 = dhb.histogramdd(
        (x, y),
        bins=bins,
        range=range,
        weights=w,
        histogram=dhb.Histogram,
        storage=dhb.storage.Weight(),
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
    h1 = dhb.histogramdd(
        df,
        bins=bins,
        weights=w,
        histogram=dhb.Histogram,
        storage=dhb.storage.Weight(),
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
    h1 = dhb.histogram2d(
        x,
        y,
        bins=bins,
        range=range,
        weights=w,
        histogram=dhb.Histogram,
        storage=dhb.storage.Weight(),
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
    h1 = dhb.histogram(
        x,
        bins=bins,
        range=range,
        weights=w,
        histogram=dhb.Histogram,
        storage=dhb.storage.Weight(),
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
    h1, edges1 = dhb.histogramdd((x, y, z), bins=bins, range=range, weights=w)
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
    h1, edgesx1, edgesy1 = dhb.histogram2d(x, y, bins=bins, range=range)
    h2, edgesx2, edgesy2 = bhnp.histogram2d(
        x.compute(), y.compute(), bins=bins, range=range
    )
    np.testing.assert_array_almost_equal(h1.compute(), h2)
    np.testing.assert_array_almost_equal(edgesx1.compute(), edgesx2)
    np.testing.assert_array_almost_equal(edgesy1.compute(), edgesy2)


def test_histogram_da_return():
    x = da.random.standard_normal(size=(3_000,), chunks=500)
    bins = np.array([-3, -2.2, 0, 1.1, 2.2, 3.3])
    h1, edges1 = dhb.histogram(x, bins=bins)
    h2, edges2 = bhnp.histogram(x.compute(), bins=bins)
    np.testing.assert_array_almost_equal(h1.compute(), h2)
    np.testing.assert_array_almost_equal(edges1.compute(), edges2)


def test_to_delayed():
    x = da.random.standard_normal(size=(2_000, 3), chunks=(500, 3))
    bins = [
        [-3, -2.2, 0, 1.1, 2.2, 3.3],
        [-4, -1.1, 0, 2.2, 3.3, 4.4],
        [-2, -0.9, 0, 0.3, 2.2],
    ]
    dh = dhb.Histogram(
        dhb.axis.Variable(bins[0]),
        dhb.axis.Variable(bins[1]),
        dhb.axis.Variable(bins[2]),
    )
    dh.fill(x)
    dh.compute()
    dh.fill(x)
    ch = dhc.clone(dh)
    ch.fill(*(x.compute().T))
    ch.fill(*(x.compute().T))
    np.testing.assert_array_almost_equal(
        dh.to_delayed().compute().to_numpy()[0], ch.to_numpy()[0]
    )
