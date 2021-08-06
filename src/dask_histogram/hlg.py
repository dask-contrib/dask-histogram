"""Dask Histogram High Level Graph API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

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


def _clone_ref(partedhist: Any) -> bh.Histogram:
    return bh.Histogram(*partedhist.axes, storage=partedhist._storage_type())


def _histogram_on_block1(data: Any, *, histref: bh.Histogram) -> bh.Histogram:
    return _clone_ref(histref).fill(data)


def _histogram_on_block2(x: Any, y: Any, *, histref: bh.Histogram) -> bh.Histogram:
    return _clone_ref(histref).fill(x, y)


def _blocked_sa(
    sample: Any,
    weight: Any = None,
    *,
    histref: bh.Histogram = None,
) -> bh.Histogram:
    return _clone_ref(histref).fill(sample, weight=weight)


def _blocked_ma(
    *sample: Any,
    weight: Any = None,
    histref: bh.Histogram = None,
) -> bh.Histogram:
    return _clone_ref(histref).fill(*sample, weight=weight)


def _blocked_df(
    sample: Any,
    weight: Any = None,
    *,
    histref: bh.Histogram = None,
) -> bh.Histogram:
    return _clone_ref(histref).fill(*(sample[c] for c in sample.columns), weight=weight)


class Histogram(db.Item):
    def __init__(self, dsk: HighLevelGraph, key: str) -> None:
        self.dask = dsk
        self.key = key
        self.name: str = key

    def __str__(self) -> str:
        return f"dask_histogram.Histogram<{key_split(self.name)}>"

    __repr__ = __str__


def finalize(results: Any) -> Any:
    # if not results:
    #     return results
    # return sum(results)
    return results


class PartitionedHistogram(DaskMethodsMixin):
    def __init__(self, dsk: HighLevelGraph, name: str, npartitions: int) -> None:
        self.dask: HighLevelGraph = dsk
        self.name: str = name
        self.npartitions: int = npartitions

    def __dask_graph__(self) -> HighLevelGraph:
        return self.dask

    def __dask_keys__(self) -> List[Tuple[str, int]]:
        return [(self.name, i) for i in range(self.npartitions)]

    def __dask_layers__(self) -> Tuple[str]:
        return (self.name,)

    def __dask_tokenize__(self) -> str:
        return self.name

    def __dask_postcompute__(self) -> Any:
        return finalize, ()

    def _rebuild(self, dsk: Any, *, rename: Any = None) -> Any:
        name = self.name
        if rename:
            name = rename.get(name, name)
        return type(self)(dsk, name, self.npartitions)

    def __str__(self) -> str:
        return "dask_histogram.PartitionedHistogram,<%s, npartitions=%d>" % (
            key_split(self.name),
            self.npartitions,
        )

    __repr__ = __str__
    __dask_scheduler__ = staticmethod(mpget)


def _reduction(
    partedhist: PartitionedHistogram,
    split_every: Optional[int] = None,
) -> Histogram:
    if split_every is None:
        split_every = 4
    if split_every is False:
        split_every = partedhist.npartitions

    token = tokenize(partedhist, sum, split_every)
    k = partedhist.npartitions
    dsk = {}
    b = partedhist.name
    fmt = "hist-aggregate-%s" % token
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

    g = HighLevelGraph.from_collections(fmt, dsk, dependencies=[partedhist])
    dsk[fmt] = dsk.pop((fmt, 0))  # type: ignore
    return Histogram(g, fmt)


def _indexify(name: str, *args: DaskCollection) -> Tuple[str, ...]:
    pairs = [(name, "i")] + [(a.name, "i") for a in args]
    return sum(pairs, ())


def histo(
    *args, weights=None, axes=None, storage=None, aggregate_split_every=10
) -> Histogram:
    if storage is None:
        storage = bh.storage.Weight()

    r = bh.Histogram(
        *axes,
        storage=storage,
    )

    if len(args) == 1:
        x = args[0]
        name = "histo-{}".format(tokenize(x, weights, axes))
        if weights is None:
            g = dask_blockwise(
                _blocked_sa,
                *_indexify(name, x),
                numblocks={x.name: x.numblocks},
                histref=r,
            )
            hlg = HighLevelGraph.from_collections(name, g, dependencies=(x,))
            return _reduction(
                PartitionedHistogram(hlg, name, x.npartitions),
                split_every=aggregate_split_every,
            )
        else:
            g = dask_blockwise(
                _blocked_sa,
                *_indexify(name, x, weights),
                numblocks={x.name: x.numblocks, weights.name: weights.numblocks},
                histref=r,
            )
            hlg = HighLevelGraph.from_collections(name, g, dependencies=(x, weights))
            return _reduction(
                PartitionedHistogram(hlg, name, x.npartitions),
                split_every=aggregate_split_every,
            )

    elif len(args) == 2:
        x = args[0]
        y = args[1]
        name = "histo-{}".format(tokenize(x, y, axes))
        g = dask_blockwise(
            _histogram_on_block2,
            *_indexify(name, x, y),
            numblocks={x.name: x.numblocks, y.name: y.numblocks},
            histref=r,
        )
        hlg = HighLevelGraph.from_collections(name, g, dependencies=(x, y))
        return _reduction(
            PartitionedHistogram(hlg, name, x.npartitions),
            split_every=aggregate_split_every,
        )

    else:
        raise NotImplementedError("WIP")


if __name__ == "__main__":
    x = da.random.standard_normal(size=(5000,), chunks=(250,))
    y = da.random.standard_normal(size=(5000,), chunks=(250,))
    w = da.random.uniform(0, 1, size=(5000,), chunks=(250,))
    histo = histo(
        x,
        axes=(bh.axis.Regular(10, -3, 3),),
        aggregate_split_every=4,
        weights=None,
    )
    histo.visualize()
