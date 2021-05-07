import boost_histogram as bh
import dask.array as da
import numpy as np
import pytest

import dask_histogram as dh


@pytest.mark.parametrize("use_weights", [True, False])
def test_obj_1D(use_weights):
    x = da.random.standard_normal(size=(2000,), chunks=(400,))
    if use_weights:
        weights = da.random.uniform(0.5, 0.75, size=x.shape, chunks=x.chunks)
        storage = dh.storage.Weight()
    else:
        weights = None
        storage = dh.storage.Double()

    h = dh.Histogram(dh.axis.Regular(12, -3, 3), storage=storage)
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
        storage = dh.storage.Weight()
    else:
        weights = None
        storage = dh.storage.Double()

    h = dh.Histogram(
        dh.axis.Regular(9, -3.5, 3.5),
        dh.axis.Regular(9, -3.2, 3.2),
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
        storage = dh.storage.Weight()
    else:
        weights = None
        storage = dh.storage.Double()

    h = dh.Histogram(
        dh.axis.Regular(8, -3.5, 3.5),
        dh.axis.Regular(7, -3.3, 3.3),
        dh.axis.Regular(9, -3.2, 3.2),
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
    h = dh.Histogram(
        dh.axis.Regular(8, -3.5, 3.5),
        dh.axis.Regular(9, -3.2, 3.2),
    )
    h.fill(x)
    assert h.staged_fills()
    h.clear_fills()
    assert not h.staged_fills()


def test_concrete_fill():
    x = da.random.standard_normal(size=(500, 2), chunks=(4, 2))
    h = dh.Histogram(
        dh.axis.Regular(8, -3.5, 3.5),
        dh.axis.Regular(9, -3.2, 3.2),
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
