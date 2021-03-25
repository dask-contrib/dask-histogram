from typing import Any, Optional
import boost_histogram as bh
from dask.highlevelgraph import HighLevelGraph
import dask.array as da
from dask.base import tokenize
from dask.core import flatten
import numpy as np


def blocked_fill(
    hist: bh.Histogram, data: np.ndarray, weight: Optional[np.ndarray] = None
) -> bh.Histogram:
    hist_for_block = bh.Histogram(*hist.axes, storage=hist._storage_type())
    hist_for_block.fill(data, weight=weight)
    return hist_for_block


def fill_graph(
    hist: bh.Histogram, data: da.Array, weight: Optional[da.Array] = None
) -> HighLevelGraph:
    name = f"filled-{tokenize(hist, data, weight)}"
    dkeys = flatten(data.__dask_keys__())
    if weight is not None:
        wkeys = flatten(weight.__dask_keys__())
    dsk = {
        (name, i, 0): (blocked_fill, hist, dk, wk)
        for i, (dk, wk) in enumerate(zip(dkeys, wkeys))
    }
    deps = (
        data,
        weight,
    )
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=deps)
    return graph


h = bh.Histogram(bh.axis.Regular(50, -5.0, 5.0), storage=bh.storage.Weight())
x = da.random.standard_normal(size=(10_000_000,), chunks=1_000_000)
w = da.random.uniform(size=(10_000_000), chunks=x.chunksize[0])
graph = fill_graph(h, x, weight=w)
