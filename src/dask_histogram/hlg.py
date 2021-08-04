"""Dask Histogram High Level Graph API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Tuple

import boost_histogram as bh
import dask.array as da
import dask.bag as db
from dask.bag.core import empty_safe_aggregate, partition_all
from dask.base import DaskMethodsMixin, tokenize
from dask.blockwise import blockwise as dask_blockwise
from dask.highlevelgraph import HighLevelGraph
from dask.multiprocessing import get as mpget
from dask.utils import key_split

if TYPE_CHECKING:
    from .boost import DaskCollection


def _histogram_on_block1(data: DaskCollection, histref: bh.Histogram) -> bh.Histogram:
    h = bh.Histogram(*histref.axes, storage=histref._storage_type())
    h.fill(data)
    return h


def _histogram_on_block2(x, y, histref: bh.Histogram) -> bh.Histogram:
    h = bh.Histogram(*histref.axes, storage=histref._storage_type())
    h.fill(x, y)
    return h


class Histo(db.Item):
    def __init__(self, dsk: HighLevelGraph, key: str) -> None:
        self.dask = dsk
        self.key = key
        self.name: str = key

    def __str__(self) -> str:
        return f"dask_histogram.Histo<{key_split(self.name)}>"

    __repr__ = __str__


def finalize(results: Any) -> Any:
    if not results:
        return results
    return sum(results)


class ParitionedHisto(DaskMethodsMixin):
    def __init__(self, dsk: HighLevelGraph, name: str, npartitions: int) -> None:
        self.dask: HighLevelGraph = dsk
        self.name: str = name
        self.npartitions: int = npartitions

    def __dask_graph__(self) -> HighLevelGraph:
        return self.dask

    def __dask_keys__(self) -> List[Any]:
        return [(self.name, i) for i in range(self.npartitions)]

    def __dask_layers__(self) -> Tuple[str, ...]:
        return (self.name,)

    def __dask_tokenize__(self) -> str:
        return self.name

    def __dask_postcompute__(self) -> Any:
        return finalize, ()

    def _rebuild(self, dsk, *, rename=None) -> Any:
        name = self.name
        if rename:
            name = rename.get(name, name)
        return type(self)(dsk, name, self.npartitions)

    def __str__(self) -> str:
        return "dask_histogram.PartitionedHisto<%s, npartitions=%d>" % (
            key_split(self.name),
            self.npartitions,
        )

    __repr__ = __str__
    __dask_scheduler__ = staticmethod(mpget)


def reduction(histo, split_every=None) -> Histo:
    if split_every is None:
        split_every = 4
    if split_every is False:
        split_every = histo.npartitions

    token = tokenize(histo, sum, split_every)
    k = histo.npartitions
    dsk = {}
    b = histo.name
    fmt = "histo-aggregate-%s" % token
    d = 0

    while k > split_every:
        c = f"{fmt}{d}"
        for i, inds in enumerate(partition_all(split_every, range(k))):
            dsk[(c, i)] = (
                empty_safe_aggregate,
                sum,
                [(b, j) for j in inds],
                False,
            )
        k = i + 1
        b = c
        d += 1

    dsk[(fmt, 0)] = (
        empty_safe_aggregate,
        sum,
        [(b, j) for j in range(k)],
        True,
    )
    from pprint import pprint

    pprint(dsk)
    g = HighLevelGraph.from_collections(fmt, dsk, dependencies=[histo])
    dsk[fmt] = dsk.pop((fmt, 0))  # type: ignore
    return Histo(g, fmt)


def histogram(*args, axes=None, storage=None, aggregate_split_every=10) -> Histo:
    if storage is None:
        storage = bh.storage.Weight()

    r = bh.Histogram(
        *axes,
        storage=storage,
    )

    if len(args) == 1:
        x = args[0]
        name = "histogram-{}".format(tokenize(x, axes))
        g = dask_blockwise(
            _histogram_on_block1,
            name,
            "i",
            x.name,
            "i",
            numblocks={x.name: x.numblocks},
            histref=r,
        )
        hlg = HighLevelGraph.from_collections(name, g, dependencies=(x,))
        return reduction(
            ParitionedHisto(hlg, name, x.npartitions),
            split_every=aggregate_split_every,
        )

    elif len(args) == 2:
        x = args[0]
        y = args[1]
        name = "histogram-{}".format(tokenize(x, y, axes))
        g = dask_blockwise(
            _histogram_on_block2,
            name,
            "i",
            x.name,
            "i",
            y.name,
            "i",
            numblocks={x.name: x.numblocks, y.name: y.numblocks},
            histref=r,
        )
        hlg = HighLevelGraph.from_collections(name, g, dependencies=(x, y))
        return reduction(
            ParitionedHisto(hlg, name, x.npartitions),
            split_every=aggregate_split_every,
        )


if __name__ == "__main__":
    x = da.random.standard_normal(size=(5000,), chunks=(250,))
    y = da.random.standard_normal(size=(5000,), chunks=(250,))
    h = histogram(
        x,
        y,
        axes=(bh.axis.Regular(10, -3, 3), bh.axis.Regular(10, -3, 3)),
        aggregate_split_every=4,
    )
    h.visualize()
