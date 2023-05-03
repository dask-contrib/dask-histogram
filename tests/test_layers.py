import dask.array as da

import dask_histogram.boost as dhb
from dask_histogram.layers import MockableDataFrameTreeReduction


def test_mdftr():
    x = da.random.standard_normal(size=(2000,), chunks=(400,))
    weights = da.random.uniform(0.5, 0.75, size=x.shape, chunks=x.chunks)
    storage = dhb.storage.Weight()

    h = dhb.Histogram(dhb.axis.Regular(12, -3, 3), storage=storage)
    h.fill(x, weight=weights)

    for layer in h.dask:
        if isinstance(layer, MockableDataFrameTreeReduction):
            mocked = layer.mock()
            assert mocked.npartitions_input == 1
            assert layer.npartitions_input == 5
