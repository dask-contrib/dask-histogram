"""Dask Histogram High Level Graph API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import boost_histogram as bh
import dask.bag as db
from dask.bag.core import empty_safe_aggregate, partition_all
from dask.base import DaskMethodsMixin, tokenize
from dask.blockwise import blockwise
from dask.dataframe.core import partitionwise_graph as partitionwise
from dask.highlevelgraph import HighLevelGraph
from dask.threaded import get as tget
from dask.utils import is_dataframe_like, key_split

from .boost import clone

if TYPE_CHECKING:
    from .boost import DaskCollection


def _blocked_sa_w(
    sample: Any,
    weights: Any,
    *,
    histref: bh.Histogram = None,
) -> bh.Histogram:
    """Blocked calculation; single argument; weighted."""
    if sample.ndim == 1:
        return clone(histref).fill(sample, weight=weights)
    elif sample.ndim == 2:
        return clone(histref).fill(*(sample.T), weight=weights)
    else:
        raise ValueError("Data must be one or two dimensional.")


def _blocked_sa(
    sample: Any,
    *,
    histref: bh.Histogram = None,
) -> bh.Histogram:
    """Blocked calculation; single argument; unweighted."""
    if sample.ndim == 1:
        return clone(histref).fill(sample, weight=None)
    elif sample.ndim == 2:
        return clone(histref).fill(*(sample.T), weight=None)
    else:
        raise ValueError("Data must be one or two dimensional.")


def _blocked_ma_w(
    *sample: Any,
    histref: bh.Histogram = None,
) -> bh.Histogram:
    """Blocked calculation; multiargument; unweighted."""
    weights = sample[-1]
    sample = sample[:-1]
    return clone(histref).fill(*sample, weight=weights)


def _blocked_ma(
    *sample: Any,
    histref: bh.Histogram = None,
) -> bh.Histogram:
    """Blocked calculation; multiargument; unweighted."""
    return clone(histref).fill(*sample, weight=None)


def _blocked_df_w(
    sample: Any,
    weights: Any,
    *,
    histref: bh.Histogram = None,
) -> bh.Histogram:
    """Blocked calculation; single argument; weighted."""
    return clone(histref).fill(*(sample[c] for c in sample.columns), weight=weights)


def _blocked_df(
    sample: Any,
    histref: bh.Histogram = None,
) -> bh.Histogram:
    return clone(histref).fill(*(sample[c] for c in sample.columns), weight=None)


class AggHistogram(db.Item):
    def __init__(self, dsk: HighLevelGraph, key: str, histref: bh.Histogram) -> None:
        self.dask: HighLevelGraph = dsk
        self.key: str = key
        self.name: str = key
        self._histref: bh.Histogram = histref

    @property
    def histref(self) -> bh.Histogram:
        return self._histref

    def __str__(self) -> str:
        return f"dask_histogram.AggHistogram<{key_split(self.name)}>"

    __repr__ = __str__
    __dask_scheduler__ = staticmethod(tget)


def finalize(results: Any) -> Any:
    return results


class PartitionedHistogram(DaskMethodsMixin):
    def __init__(
        self, dsk: HighLevelGraph, name: str, npartitions: int, histref: bh.Histogram
    ) -> None:
        self.dask: HighLevelGraph = dsk
        self.name: str = name
        self.npartitions: int = npartitions
        self._histref: bh.Histogram = histref

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
        return type(self)(dsk, name, self.npartitions, self.histref)

    def __str__(self) -> str:
        return "dask_histogram.PartitionedHistogram,<%s, npartitions=%d>" % (
            key_split(self.name),
            self.npartitions,
        )

    __repr__ = __str__
    __dask_scheduler__ = staticmethod(tget)

    @property
    def histref(self) -> bh.Histogram:
        return self._histref


def _reduction(
    partedhist: PartitionedHistogram,
    split_every: Optional[int] = None,
) -> AggHistogram:
    if split_every is None:
        split_every = 4
    if split_every is False:
        split_every = partedhist.npartitions

    token = tokenize(partedhist, sum, split_every)
    fmt = f"hist-aggregate-{token}"
    k = partedhist.npartitions
    b = partedhist.name
    d = 0
    dsk = {}
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
    return AggHistogram(g, fmt, histref=partedhist.histref)


def _indexify(
    name: str,
    *args: DaskCollection,
    idx: str = "i",
    weights: Optional[DaskCollection] = None,
) -> Tuple[str, ...]:
    """Generate name and index pairs for blockwise use."""
    pairs = [(name, "i")] + [(a.name, idx) for a in args]
    if weights is not None:
        pairs.append((weights.name, "i"))
    return sum(pairs, ())


def _numblocks_or_npartitions(coll: DaskCollection) -> Tuple[int, ...]:
    if hasattr(coll, "numblocks"):
        return coll.numblocks
    elif hasattr(coll, "npartitions"):
        return (coll.npartitions,)
    else:
        raise AttributeError("numblocks or npartitions expected on collection.")


def _gen_numblocks(*args, weights=None):
    result = {a.name: _numblocks_or_npartitions(a) for a in args}
    if weights is not None:
        result[weights.name] = _numblocks_or_npartitions(weights)
    print(result)
    return result


def _dependencies(
    *args: DaskCollection,
    weights: Optional[DaskCollection] = None,
) -> Tuple[DaskCollection, ...]:
    if weights is not None:
        return (*args, weights)
    return args


def single_argument_histogram(
    x: DaskCollection,
    histref: bh.Histogram,
    weights: Optional[DaskCollection] = None,
    agg_split_every: int = 10,
) -> AggHistogram:
    name = "histogram-{}".format(tokenize(x, histref, weights))
    bwg = partitionwise(_blocked_sa, name, x, weight=weights, histref=histref)
    dependencies = _dependencies(x, weights=weights)
    hlg = HighLevelGraph.from_collections(name, bwg, dependencies=dependencies)
    ph = PartitionedHistogram(hlg, name, x.npartitions, histref=histref)
    return _reduction(ph, split_every=agg_split_every)


# def multi_argument_histogram(
#     *data: DaskCollection,
#     histref: bh.Histogram,
#     weights: Optional[DaskCollection] = None,
#     agg_split_every: int = 10,
# ) -> AggHistogram:
#     name = "histogram-{}".format(tokenize(*data, histref, weights))
#     bwg = blockwise(
#         _blocked_ma,
#         *_indexify(name, *data, idx="i", weights=weights),
#         numblocks=_gen_numblocks(*data, weights=weights),
#         histref=histref,
#     )
#     dependencies = _dependencies(*data, weights=weights)


def histo_manual_blockwise(
    *args, weights=None, axes=None, storage=None, aggregate_split_every=10
) -> AggHistogram:
    if storage is None:
        storage = bh.storage.Weight()

    r = bh.Histogram(
        *axes,
        storage=storage,
    )

    if len(args) == 1:
        x = args[0]
        return single_argument_histogram(
            x,
            histref=r,
            weights=weights,
            agg_split_every=aggregate_split_every,
        )
    elif len(args) == 2:
        x = args[0]
        y = args[1]
        name = "histogram-{}".format(tokenize(x, y, axes))
        g = blockwise(
            _blocked_ma,
            *_indexify(name, x, y),
            numblocks={x.name: x.numblocks, y.name: y.numblocks},
            histref=r,
        )
        hlg = HighLevelGraph.from_collections(name, g, dependencies=(x, y))
        return _reduction(
            PartitionedHistogram(hlg, name, x.npartitions, histref=r),
            split_every=aggregate_split_every,
        )

    else:
        raise NotImplementedError("WIP")


def histogram(
    *data: DaskCollection,
    histref: bh.Histogram,
    weights: Optional[DaskCollection] = None,
    split_every: int = 10,
) -> AggHistogram:
    name = "histogram-{}".format(tokenize(data, histref, weights))
    if len(data) == 1 and not is_dataframe_like(data[0]):
        x = data[0]
        if weights is not None:
            g = partitionwise(_blocked_sa_w, name, x, weights, histref=histref)
        else:
            g = partitionwise(_blocked_sa, name, x, histref=histref)
    elif len(data) == 1 and is_dataframe_like(data[0]):
        x = data[0]
        if weights is not None:
            g = partitionwise(_blocked_df_w, name, x, weights, histref=histref)
        else:
            g = partitionwise(_blocked_df, name, x, histref=histref)
    else:
        if weights is not None:
            g = partitionwise(_blocked_ma_w, name, *data, weights, histref=histref)
        else:
            g = partitionwise(_blocked_ma, name, *data, histref=histref)

    dependencies = _dependencies(*data, weights=weights)
    hlg = HighLevelGraph.from_collections(name, g, dependencies=dependencies)
    ph = PartitionedHistogram(hlg, name, data[0].npartitions, histref=histref)
    return _reduction(ph, split_every=split_every)
