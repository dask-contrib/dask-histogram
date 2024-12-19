"""Dask Histogram core High Level Graph API."""

from __future__ import annotations

import operator
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Hashable, Literal, Mapping, Sequence

import boost_histogram as bh
import dask.config
import numpy as np
from dask.base import DaskMethodsMixin, dont_optimize, is_dask_collection, tokenize
from dask.blockwise import BlockwiseDep, blockwise, fuse_roots, optimize_blockwise
from dask.context import globalmethod
from dask.core import flatten
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph
from dask.local import identity
from dask.threaded import get as tget
from dask.utils import is_dataframe_like, key_split

from dask_histogram.layers import MockableDataFrameTreeReduction

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from dask_histogram.typing import DaskCollection

__all__ = (
    "AggHistogram",
    "PartitionedHistogram",
    "clone",
    "factory",
)


def hist_safe_sum(items):
    return sum(item for item in items if not isinstance(item, tuple))


def clone(histref: bh.Histogram | None = None) -> bh.Histogram:
    """Create a Histogram object based on another.

    The axes and storage of the `histref` will be used to create a new
    Histogram object.

    Parameters
    ----------
    histref : bh.Histogram
        The reference Histogram.

    Returns
    -------
    bh.Histogram
        New Histogram with identical axes and storage.

    """
    if histref is None:
        return bh.Histogram()
    return bh.Histogram(*histref.axes, storage=histref.storage_type())


def _blocked_sa(
    data: Any,
    *,
    histref: tuple | bh.Histogram | None = None,
) -> bh.Histogram:
    """Blocked calculation; single argument; unweighted; no sample."""
    thehist = (
        clone(histref)
        if not isinstance(histref, tuple)
        else bh.Histogram(*histref[0], storage=histref[1], metadata=histref[2])
    )
    if data.ndim == 1:
        return thehist.fill(data)
    elif data.ndim == 2:
        return thehist.fill(*(data.T))
    else:
        raise ValueError("Data must be one or two dimensional.")


def _blocked_sa_s(
    data: Any,
    sample: Any,
    *,
    histref: tuple | bh.Histogram | None = None,
) -> bh.Histogram:
    """Blocked calculation; single argument; unweighted; with sample."""
    thehist = (
        clone(histref)
        if not isinstance(histref, tuple)
        else bh.Histogram(*histref[0], storage=histref[1], metadata=histref[2])
    )
    if data.ndim == 1:
        return thehist.fill(data, sample=sample)
    elif data.ndim == 2:
        return thehist.fill(*(data.T), sample=sample)
    else:
        raise ValueError("Data must be one or two dimensional.")


def _blocked_sa_w(
    data: Any,
    weights: Any,
    *,
    histref: tuple | bh.Histogram | None = None,
) -> bh.Histogram:
    """Blocked calculation; single argument; weighted; no sample."""
    thehist = (
        clone(histref)
        if not isinstance(histref, tuple)
        else bh.Histogram(*histref[0], storage=histref[1], metadata=histref[2])
    )
    if data.ndim == 1:
        return thehist.fill(data, weight=weights)
    elif data.ndim == 2:
        return thehist.fill(*(data.T), weight=weights)
    else:
        raise ValueError("Data must be one or two dimensional.")


def _blocked_sa_w_s(
    data: Any,
    weights: Any,
    sample: Any,
    *,
    histref: tuple | bh.Histogram | None = None,
) -> bh.Histogram:
    """Blocked calculation; single argument; weighted; with sample."""
    thehist = (
        clone(histref)
        if not isinstance(histref, tuple)
        else bh.Histogram(*histref[0], storage=histref[1], metadata=histref[2])
    )
    if data.ndim == 1:
        return thehist.fill(data, weight=weights, sample=sample)
    elif data.ndim == 2:
        return thehist.fill(*(data.T), weight=weights, sample=sample)
    else:
        raise ValueError("Data must be one or two dimensional.")


def _blocked_ma(
    *data: Any,
    histref: tuple | bh.Histogram | None = None,
) -> bh.Histogram:
    """Blocked calculation; multiargument; unweighted; no sample."""
    thehist = (
        clone(histref)
        if not isinstance(histref, tuple)
        else bh.Histogram(*histref[0], storage=histref[1], metadata=histref[2])
    )
    return thehist.fill(*data)


def _blocked_ma_s(
    *data: Any,
    histref: tuple | bh.Histogram | None = None,
) -> bh.Histogram:
    """Blocked calculation; multiargument; unweighted; with sample."""
    sample = data[-1]
    data = data[:-1]
    thehist = (
        clone(histref)
        if not isinstance(histref, tuple)
        else bh.Histogram(*histref[0], storage=histref[1], metadata=histref[2])
    )
    return thehist.fill(*data, sample=sample)


def _blocked_ma_w(
    *data: Any,
    histref: tuple | bh.Histogram | None = None,
) -> bh.Histogram:
    """Blocked calculation; multiargument; weighted; no sample."""
    weights = data[-1]
    data = data[:-1]
    thehist = (
        clone(histref)
        if not isinstance(histref, tuple)
        else bh.Histogram(*histref[0], storage=histref[1], metadata=histref[2])
    )
    return thehist.fill(*data, weight=weights)


def _blocked_ma_w_s(
    *data: Any,
    histref: tuple | bh.Histogram | None = None,
) -> bh.Histogram:
    """Blocked calculation; multiargument; weighted; with sample."""
    weights = data[-2]
    sample = data[-1]
    data = data[:-2]
    thehist = (
        clone(histref)
        if not isinstance(histref, tuple)
        else bh.Histogram(*histref[0], storage=histref[1], metadata=histref[2])
    )
    return thehist.fill(*data, weight=weights, sample=sample)


def _blocked_df(
    data: Any,
    *,
    histref: tuple | bh.Histogram | None = None,
) -> bh.Histogram:
    thehist = (
        clone(histref)
        if not isinstance(histref, tuple)
        else bh.Histogram(*histref[0], storage=histref[1], metadata=histref[2])
    )
    return thehist.fill(*(data[c] for c in data.columns), weight=None)


def _blocked_df_s(
    data: Any,
    sample: Any,
    *,
    histref: tuple | bh.Histogram | None = None,
) -> bh.Histogram:
    thehist = (
        clone(histref)
        if not isinstance(histref, tuple)
        else bh.Histogram(*histref[0], storage=histref[1], metadata=histref[2])
    )
    return thehist.fill(*(data[c] for c in data.columns), sample=sample)


def _blocked_df_w(
    data: Any,
    weights: Any,
    *,
    histref: tuple | bh.Histogram | None = None,
) -> bh.Histogram:
    """Blocked calculation; single argument; weighted; no sample."""
    thehist = (
        clone(histref)
        if not isinstance(histref, tuple)
        else bh.Histogram(*histref[0], storage=histref[1], metadata=histref[2])
    )
    return thehist.fill(*(data[c] for c in data.columns), weight=weights)


def _blocked_df_w_s(
    data: Any,
    weights: Any,
    sample: Any,
    *,
    histref: tuple | bh.Histogram | None = None,
) -> bh.Histogram:
    """Blocked calculation; single argument; weighted; with sample."""
    thehist = (
        clone(histref)
        if not isinstance(histref, tuple)
        else bh.Histogram(*histref[0], storage=histref[1], metadata=histref[2])
    )
    return thehist.fill(*(data[c] for c in data.columns), weight=weights, sample=sample)


def _blocked_dak(
    data: Any,
    weights: Any | None,
    sample: Any | None,
    *,
    histref: tuple | bh.Histogram | None = None,
) -> bh.Histogram:
    import awkward as ak

    thedata = (
        ak.typetracer.length_zero_if_typetracer(data)
        if isinstance(data, ak.Array)
        else data
    )
    theweights = (
        ak.typetracer.length_zero_if_typetracer(weights)
        if isinstance(weights, ak.Array)
        else weights
    )
    thesample = (
        ak.typetracer.length_zero_if_typetracer(sample)
        if isinstance(sample, ak.Array)
        else sample
    )

    thehist = (
        clone(histref)
        if not isinstance(histref, tuple)
        else bh.Histogram(*histref[0], storage=histref[1], metadata=histref[2])
    )
    return thehist.fill(thedata, weight=theweights, sample=thesample)


def _blocked_dak_ma(
    *data: Any,
    histref: tuple | bh.Histogram | None = None,
) -> bh.Histogram:
    import awkward as ak

    thedata = [
        (
            ak.typetracer.length_zero_if_typetracer(datum)
            if isinstance(datum, ak.Array)
            else datum
        )
        for datum in data
    ]

    thehist = (
        clone(histref)
        if not isinstance(histref, tuple)
        else bh.Histogram(*histref[0], storage=histref[1], metadata=histref[2])
    )
    return thehist.fill(*tuple(thedata))


def _blocked_dak_ma_w(
    *data: Any,
    histref: tuple | bh.Histogram | None = None,
) -> bh.Histogram:
    import awkward as ak

    thedata = [
        (
            ak.typetracer.length_zero_if_typetracer(datum)
            if isinstance(datum, ak.Array)
            else datum
        )
        for datum in data[:-1]
    ]
    theweights = (
        ak.typetracer.length_zero_if_typetracer(data[-1])
        if isinstance(data[-1], ak.Array)
        else data[-1]
    )

    thehist = (
        clone(histref)
        if not isinstance(histref, tuple)
        else bh.Histogram(*histref[0], storage=histref[1], metadata=histref[2])
    )

    if ak.backend(*data) != "typetracer":
        thehist.fill(*tuple(thedata), weight=theweights)

    return thehist


def _blocked_dak_ma_s(
    *data: Any,
    histref: tuple | bh.Histogram | None = None,
) -> bh.Histogram:
    import awkward as ak

    thedata = [
        (
            ak.typetracer.length_zero_if_typetracer(datum)
            if isinstance(datum, ak.Array)
            else datum
        )
        for datum in data[:-1]
    ]
    thesample = (
        ak.typetracer.length_zero_if_typetracer(data[-1])
        if isinstance(data[-1], ak.Array)
        else data[-1]
    )

    thehist = (
        clone(histref)
        if not isinstance(histref, tuple)
        else bh.Histogram(*histref[0], storage=histref[1], metadata=histref[2])
    )
    return thehist.fill(*tuple(thedata), sample=thesample)


def _blocked_dak_ma_w_s(
    *data: Any,
    histref: tuple | bh.Histogram | None = None,
) -> bh.Histogram:
    import awkward as ak

    thedata = [
        (
            ak.typetracer.length_zero_if_typetracer(datum)
            if isinstance(datum, ak.Array)
            else datum
        )
        for datum in data[:-2]
    ]
    theweights = (
        ak.typetracer.length_zero_if_typetracer(data[-2])
        if isinstance(data[-2], ak.Array)
        else data[-2]
    )
    thesample = (
        ak.typetracer.length_zero_if_typetracer(data[-1])
        if isinstance(data[-1], ak.Array)
        else data[-1]
    )

    thehist = (
        clone(histref)
        if not isinstance(histref, tuple)
        else bh.Histogram(*histref[0], storage=histref[1], metadata=histref[2])
    )
    return thehist.fill(*tuple(thedata), weight=theweights, sample=thesample)


def _blocked_multi(
    repacker: Callable,
    *flattened_inputs: tuple[Any],
) -> bh.Histogram:
    data_list, weights, samples, histref = repacker(flattened_inputs)

    weights = weights or (None for _ in range(len(data_list)))
    samples = samples or (None for _ in range(len(data_list)))

    thehist = (
        clone(histref)
        if not isinstance(histref, tuple)
        else bh.Histogram(*histref[0], storage=histref[1], metadata=histref[2])
    )

    for (
        datatuple,
        weight,
        sample,
    ) in zip(data_list, weights, samples):
        data = datatuple
        if len(data) == 1 and data[0].ndim == 2:
            data = data[0].T
        thehist.fill(*data, weight=weight, sample=sample)

    return thehist


def _blocked_multi_df(
    repacker: Callable,
    *flattened_inputs: tuple[Any],
) -> bh.Histogram:
    data_list, weights, samples, histref = repacker(flattened_inputs)

    weights = weights or (None for _ in range(len(data_list)))
    samples = samples or (None for _ in range(len(data_list)))

    thehist = (
        clone(histref)
        if not isinstance(histref, tuple)
        else bh.Histogram(*histref[0], storage=histref[1], metadata=histref[2])
    )

    for (
        datatuple,
        weight,
        sample,
    ) in zip(data_list, weights, samples):
        data = datatuple
        if len(datatuple) == 1:
            data = data[0]
        thehist.fill(*(data[c] for c in data.columns), weight=weight, sample=sample)

    return thehist


def _blocked_multi_dak(
    repacker: Callable,
    *flattened_inputs: tuple[Any],
) -> bh.Histogram:
    import awkward as ak

    data_list, weights, samples, histref = repacker(flattened_inputs)

    weights = weights or (None for _ in range(len(data_list)))
    samples = samples or (None for _ in range(len(data_list)))

    thehist = (
        clone(histref)
        if not isinstance(histref, tuple)
        else bh.Histogram(*histref[0], storage=histref[1], metadata=histref[2])
    )

    backend = ak.backend(*flattened_inputs)

    for (
        data,
        weight,
        sample,
    ) in zip(data_list, weights, samples):
        if backend != "typetracer":
            thehist.fill(*data, weight=weight, sample=sample)
        else:
            for datum in data:
                if isinstance(datum, ak.highlevel.Array):
                    ak.typetracer.touch_data(datum)
            if isinstance(weight, ak.highlevel.Array):
                ak.typetracer.touch_data(weight)
            if isinstance(sample, ak.highlevel.Array):
                ak.typetracer.touch_data(sample)

    return thehist


def optimize(
    dsk: Mapping,
    keys: Hashable | list[Hashable] | set[Hashable],
    **kwargs: Any,
) -> Mapping:
    keys = tuple(flatten(keys))

    if not isinstance(dsk, HighLevelGraph):
        dsk = HighLevelGraph.from_collections(str(id(dsk)), dsk, dependencies=())

    dsk = optimize_blockwise(dsk, keys=keys)
    dsk = fuse_roots(dsk, keys=keys)  # type: ignore
    dsk = dsk.cull(set(keys))  # type: ignore
    return dsk


def _get_optimization_function():
    # Here we try to run optimizations from dask-awkward (if we detect
    # that dask-awkward has been imported). There is no cost to
    # running this optimization even in cases where it's unncessary
    # because if no AwkwardInputLayers from dask-awkward are
    # detected then the original graph is returned unchanged.
    try:
        from dask_awkward.lib.optimize import all_optimizations

        return all_optimizations
    except (ImportError, ModuleNotFoundError):
        pass
    return optimize


class AggHistogram(DaskMethodsMixin):
    """Aggregated Histogram collection.

    The class constructor is typically used internally;
    :py:func:`dask_histogram.factory` is recommended for users (along
    with the `dask_histogram.routines` module).

    See Also
    --------
    dask_histogram.factory

    """

    def __init__(
        self,
        dsk: HighLevelGraph,
        name: str,
        histref: bh.Histogram,
        layer: Any | None = None,
    ) -> None:
        self._dask: HighLevelGraph = dsk
        self._name: str = name
        self._meta: bh.Histogram = histref

        # NOTE: Layer only used by `Item.from_delayed`, to handle
        # Delayed objects created by other collections. e.g.:
        # Item.from_delayed(da.ones(1).to_delayed()[0]) See
        # Delayed.__init__
        self._layer = layer or name
        if isinstance(dsk, HighLevelGraph) and self._layer not in dsk.layers:
            raise ValueError(
                f"Layer {self._layer} not in the HighLevelGraph's layers: {list(dsk.layers)}"
            )

    def __dask_graph__(self) -> HighLevelGraph:
        return self._dask

    def __dask_keys__(self) -> list[tuple[str, int]]:
        return [self.key]

    def __dask_layers__(self) -> tuple[str, ...]:
        if isinstance(self._dask, HighLevelGraph) and len(self._dask.layers) == 1:
            return tuple(self._dask.layers)
        return (self.name,)

    def __dask_tokenize__(self) -> Any:
        return self.key

    def __dask_postcompute__(self) -> Any:
        return _finalize_agg_histogram, ()

    def __dask_postpersist__(self) -> Any:
        return self._rebuild, ()

    __dask_optimize__ = globalmethod(
        _get_optimization_function(), key="histogram_optimize", falsey=dont_optimize
    )

    __dask_scheduler__ = staticmethod(tget)

    def _rebuild(
        self,
        dsk: HighLevelGraph,
        *,
        rename: Mapping[str, str] | None = None,
    ) -> Any:
        name = self._name
        if rename:
            name = rename.get(name, name)
        return type(self)(dsk, name, self.histref)

    @property
    def name(self) -> str:
        return self._name

    @property
    def dask(self) -> HighLevelGraph:
        return self._dask

    @property
    def key(self) -> tuple[str, Literal[0]]:
        return (self.name, 0)

    @property
    def histref(self):
        """Empty reference boost-histogram object."""
        return self._meta

    @property
    def _storage_type(self) -> type[bh.storage.Storage]:
        """Storage type of the histogram."""
        return self.histref.storage_type

    @property
    def ndim(self) -> int:
        """Total number of dimensions."""
        return self.histref.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the histogram as an array."""
        return self.histref.shape

    @property
    def size(self) -> int:
        """Size of the histogram."""
        return self.histref.size

    def __str__(self) -> str:
        return (
            "dask_histogram.AggHistogram<"
            f"{key_split(self.name)}, "
            f"ndim={self.ndim}, "
            f"storage={self._storage_type()}"
            ">"
        )

    __repr__ = __str__

    def __reduce__(self):
        return (AggHistogram, (self._dask, self._name, self._meta))

    def to_dask_array(self, flow: bool = False, dd: bool = False) -> Any:
        """Convert histogram object to dask.array form.

        Parameters
        ----------
        flow : bool
            Include the flow bins.
        dd : bool
            Use the histogramdd return syntax, where the edges are in a tuple.
            Otherwise, this is the histogram/histogram2d return style.

        Returns
        -------
        contents : dask.array.Array
            The bin contents
        *edges : dask.array.Array
            The edges for each dimension

        """
        return to_dask_array(self, flow=flow, dd=dd)

    def to_boost(self) -> bh.Histogram:
        """Convert to a boost_histogram.Histogram via computation.

        This is an alias of `.compute()`.

        """
        return self.compute()

    def to_delayed(self, optimize_graph: bool = True) -> Delayed:
        keys = self.__dask_keys__()
        graph = self.__dask_graph__()
        layer = self.__dask_layers__()[0]
        if optimize_graph:
            graph = self.__dask_optimize__(graph, keys)
            layer = f"delayed-{self.name}"
            graph = HighLevelGraph.from_collections(layer, graph, dependencies=())
        return Delayed(keys[0], graph, layer=layer)

    def values(self, flow: bool = False) -> NDArray[Any]:
        return self.to_boost().values(flow=flow)

    def variances(self, flow: bool = False) -> Any:
        return self.to_boost().variances(flow=flow)

    def counts(self, flow: bool = False) -> NDArray[Any]:
        return self.to_boost().counts(flow=flow)

    def __array__(self) -> NDArray[Any]:
        return self.compute().__array__()

    def __iadd__(self, other: Any) -> AggHistogram:
        return _iadd(self, other)

    def __add__(self, other: Any) -> AggHistogram:
        return self.__iadd__(other)

    def __radd__(self, other: Any) -> AggHistogram:
        return self.__iadd__(other)

    def __isub__(self, other: Any) -> AggHistogram:
        return _isub(self, other)

    def __sub__(self, other: Any) -> AggHistogram:
        return self.__isub__(other)

    def __itruediv__(self, other: Any) -> AggHistogram:
        return _itruediv(self, other)

    def __truediv__(self, other: Any) -> AggHistogram:
        return self.__itruediv__(other)

    def __idiv__(self, other: Any) -> AggHistogram:
        return self.__itruediv__(other)

    def __div__(self, other: Any) -> AggHistogram:
        return self.__idiv__(other)

    def __imul__(self, other: Any) -> AggHistogram:
        return _imul(self, other)

    def __mul__(self, other: Any) -> AggHistogram:
        return self.__imul__(other)

    def __rmul__(self, other: Any) -> AggHistogram:
        return self.__mul__(other)


def _finalize_partitioned_histogram(results: Any) -> Any:
    return results


def _finalize_agg_histogram(results: Any) -> Any:
    return results[0]


class PartitionedHistogram(DaskMethodsMixin):
    """Partitioned Histogram collection.

    The class constructor is typically used internally;
    :py:func:`dask_histogram.factory` is recommended for users (along
    with the `dask_histogram.routines` module).

    See Also
    --------
    dask_histogram.factory
    dask_histogram.AggHistogram

    """

    def __init__(
        self,
        dsk: HighLevelGraph,
        name: str,
        npartitions: int,
        histref: bh.Histogram | tuple,
    ) -> None:
        self._dask: HighLevelGraph = dsk
        self._name: str = name
        self._npartitions: int = npartitions
        self._meta: tuple | bh.Histogram = histref

    @property
    def name(self) -> str:
        return self._name

    @property
    def dask(self) -> HighLevelGraph:
        return self._dask

    @property
    def npartitions(self) -> int:
        return self._npartitions

    def __dask_graph__(self) -> HighLevelGraph:
        return self.dask

    def __dask_keys__(self) -> list[tuple[str, int]]:
        return [(self.name, i) for i in range(self.npartitions)]

    def __dask_layers__(self) -> tuple[str]:
        return (self.name,)

    def __dask_tokenize__(self) -> str:
        return self.name

    def __dask_postcompute__(self) -> Any:
        return _finalize_partitioned_histogram, ()

    def _rebuild(self, dsk: Any, *, rename: Any = None) -> Any:
        name = self.name
        if rename:
            name = rename.get(name, name)
        return type(self)(dsk, name, self.npartitions, self.histref)

    __dask_optimize__ = globalmethod(
        _get_optimization_function(), key="histogram_optimize", falsey=dont_optimize
    )

    __dask_scheduler__ = staticmethod(tget)

    def __str__(self) -> str:
        return "dask_histogram.PartitionedHistogram,<%s, npartitions=%d>" % (
            key_split(self.name),
            self.npartitions,
        )

    __repr__ = __str__

    def __reduce__(self):
        return (
            PartitionedHistogram,
            (
                self._dask,
                self._name,
                self._npartitions,
                self._meta,
            ),
        )

    @property
    def histref(self):
        """boost_histogram.Histogram: reference histogram."""
        return self._meta

    def collapse(self, split_every: int | None = None) -> AggHistogram:
        """Translate into a reduced aggregated histogram."""
        return _reduction(self, split_every=split_every)

    def to_delayed(self, optimize_graph: bool = True) -> list[Delayed]:
        keys = self.__dask_keys__()
        graph = self.__dask_graph__()
        layer = self.__dask_layers__()[0]
        if optimize_graph:
            graph = self.__dask_optimize__(graph, keys)
            layer = f"delayed-{self.name}"
            graph = HighLevelGraph.from_collections(layer, graph, dependencies=())
        return [Delayed(k, graph, layer=layer) for k in keys]


def _reduction(
    ph: PartitionedHistogram,
    split_every: int | None = None,
) -> AggHistogram:
    if split_every is None:
        split_every = dask.config.get("histogram.aggregation.split-every", 8)
    if split_every is False:
        split_every = ph.npartitions

    token = tokenize(ph, sum, split_every)

    label = "histreduce"

    name_comb = f"{label}-combine-{token}"
    name_agg = f"{label}-agg-{token}"

    mdftr = MockableDataFrameTreeReduction(
        name=name_agg,
        name_input=ph.name,
        npartitions_input=ph.npartitions,
        concat_func=hist_safe_sum,
        tree_node_func=identity,
        finalize_func=identity,
        split_every=split_every,
        tree_node_name=name_comb,
    )

    graph = HighLevelGraph.from_collections(name_agg, mdftr, dependencies=(ph,))

    return AggHistogram(graph, name_agg, histref=ph.histref)


def _dependencies(
    *args: DaskCollection,
    weights: DaskCollection | None = None,
    sample: DaskCollection | None = None,
) -> tuple[DaskCollection, ...]:
    dask_args = [arg for arg in args if is_dask_collection(arg)]
    if is_dask_collection(weights):
        dask_args.append(weights)  # type: ignore[arg-type]
    if is_dask_collection(sample):
        dask_args.append(sample)  # type: ignore[arg-type]
    return tuple(dask_args)


def _weight_sample_check(
    *data: DaskCollection,
    weights: DaskCollection | None = None,
    sample: DaskCollection | None = None,
) -> int:
    if weights is None and sample is None:
        return 0
    if weights is not None:
        if weights.ndim != 1:
            raise ValueError("weights must be one dimensional.")
        if data[0].npartitions != weights.npartitions:
            raise ValueError("weights must have as many partitions as the data.")
    if sample is not None:
        if sample.ndim != 1:
            raise ValueError("sample must be one dimensional.")
        if data[0].npartitions != sample.npartitions:
            raise ValueError("sample must have as many partitions as the data.")
    return 0


def _is_dask_dataframe(obj):
    return (
        obj.__class__.__module__ == "dask.dataframe.core"
        and obj.__class__.__name__ == "DataFrame"
    )


def _is_dask_series(obj):
    return (
        obj.__class__.__module__ == "dask.dataframe.core"
        and obj.__class__.__name__ == "Series"
    )


def _partitionwise(func, layer_name, *args, **kwargs):
    from dask.array.core import Array as DaskArray

    pairs = []
    numblocks = {}
    for arg in args:
        if isinstance(arg, DaskArray):
            if arg.ndim == 1:
                pairs.extend([arg.name, "i"])
            elif arg.ndim == 0:
                pairs.extend([arg.name, ""])
            elif arg.ndim == 2:
                pairs.extend([arg.name, "ij"])
            else:
                raise ValueError("Can't add multi-dimensional array to dataframes")
            numblocks[arg._name] = arg.numblocks

        elif _is_dask_dataframe(arg) or _is_dask_series(arg):
            pairs.extend([arg._name, "i"])
            numblocks[arg._name] = (arg.npartitions,)
        elif isinstance(arg, BlockwiseDep):
            if len(arg.numblocks) == 1:
                pairs.extend([arg, "i"])
            elif len(arg.numblocks) == 2:
                pairs.extend([arg, "ij"])
            else:
                raise ValueError(
                    f"BlockwiseDep arg {arg!r} has {len(arg.numblocks)} dimensions; "
                    "only 1 or 2 are supported."
                )
        else:
            pairs.extend([arg, None])
    return blockwise(
        func,
        layer_name,
        "i",
        *pairs,
        numblocks=numblocks,
        concatenate=True,
        **kwargs,
    )


def _partitioned_histogram_multifill(
    data: tuple[DaskCollection | tuple],
    histref: bh.Histogram | tuple,
    weights: tuple[DaskCollection] | None = None,
    samples: tuple[DaskCollection] | None = None,
):
    name = f"hist-on-block-{tokenize(data, histref, weights, samples)}"

    from dask.base import unpack_collections

    flattened_deps, repacker = unpack_collections(data, weights, samples, histref)

    if is_dask_awkward_like(flattened_deps[0]):
        from dask_awkward.lib.core import partitionwise_layer as dak_pwl

        unpacked_multifill = partial(_blocked_multi_dak, repacker)
        graph = dak_pwl(unpacked_multifill, name, *flattened_deps)
    elif is_dataframe_like(flattened_deps[0]):
        unpacked_multifill = partial(_blocked_multi_df, repacker)
        graph = _partitionwise(unpacked_multifill, name, *flattened_deps)
    else:
        unpacked_multifill = partial(_blocked_multi, repacker)
        graph = _partitionwise(unpacked_multifill, name, *flattened_deps)

    hlg = HighLevelGraph.from_collections(name, graph, dependencies=flattened_deps)
    return PartitionedHistogram(
        hlg, name, flattened_deps[0].npartitions, histref=histref
    )


def _partitioned_histogram(
    *data: DaskCollection,
    histref: bh.Histogram | tuple,
    weights: DaskCollection | None = None,
    sample: DaskCollection | None = None,
    split_every: int | None = None,
) -> PartitionedHistogram:
    name = f"hist-on-block-{tokenize(data, histref, weights, sample, split_every)}"
    dask_data = tuple(datum for datum in data if is_dask_collection(datum))
    if len(dask_data) == 0:
        dask_data = data
    data_is_df = is_dataframe_like(dask_data[0])
    data_is_dak = is_dask_awkward_like(dask_data[0])
    if is_dask_collection(weights):
        _weight_sample_check(*dask_data, weights=weights)

    # Single awkward array object.
    if len(data) == 1 and data_is_dak:
        from dask_awkward.lib.core import partitionwise_layer as dak_pwl

        f = partial(_blocked_dak, histref=histref)

        g = dak_pwl(f, name, data[0], weights, sample)

    # Single object, not a dataframe
    elif len(data) == 1 and not data_is_df:
        x = data[0]
        if weights is not None and sample is not None:
            g = _partitionwise(
                _blocked_sa_w_s, name, x, weights, sample, histref=histref
            )
        elif weights is not None and sample is None:
            g = _partitionwise(_blocked_sa_w, name, x, weights, histref=histref)
        elif weights is None and sample is not None:
            g = _partitionwise(_blocked_sa_s, name, x, sample, histref=histref)
        else:
            g = _partitionwise(_blocked_sa, name, x, histref=histref)

    # Single object, is a dataframe
    elif len(data) == 1 and data_is_df:
        x = data[0]
        if weights is not None and sample is not None:
            g = _partitionwise(
                _blocked_df_w_s, name, x, weights, sample, histref=histref
            )
        elif weights is not None and sample is None:
            g = _partitionwise(_blocked_df_w, name, x, weights, histref=histref)
        elif weights is None and sample is not None:
            g = _partitionwise(_blocked_df_s, name, x, sample, histref=histref)
        else:
            g = _partitionwise(_blocked_df, name, x, histref=histref)

    # Multiple objects
    else:
        # Awkward array collection detected as first argument
        if data_is_dak:
            from dask_awkward.lib.core import partitionwise_layer as dak_pwl

            if weights is not None and sample is None:
                g = dak_pwl(_blocked_dak_ma_w, name, *data, weights, histref=histref)
            elif weights is not None and sample is not None:
                g = dak_pwl(
                    _blocked_dak_ma_w_s,
                    name,
                    *data,
                    weights,
                    sample,
                    histref=histref,
                )
            elif weights is None and sample is not None:
                g = dak_pwl(_blocked_dak_ma_s, name, *data, sample, histref=histref)
            else:
                g = dak_pwl(_blocked_dak_ma, name, *data, histref=histref)
        # Not an awkward array collection
        elif weights is not None and sample is not None:
            g = _partitionwise(
                _blocked_ma_w_s, name, *data, weights, sample, histref=histref
            )
        elif weights is not None and sample is None:
            g = _partitionwise(_blocked_ma_w, name, *data, weights, histref=histref)
        elif weights is None and sample is not None:
            g = _partitionwise(_blocked_ma_s, name, *data, sample, histref=histref)
        else:
            g = _partitionwise(_blocked_ma, name, *data, histref=histref)

    dependencies = _dependencies(*data, weights=weights, sample=sample)
    hlg = HighLevelGraph.from_collections(name, g, dependencies=dependencies)
    return PartitionedHistogram(hlg, name, dask_data[0].npartitions, histref=histref)


def to_dask_array(agghist: AggHistogram, flow: bool = False, dd: bool = False) -> Any:
    """Convert `agghist` to a `dask.array` return style.

    Parameters
    ----------
    agghist : AggHistogram
        The aggregated histogram collection to convert.
    flow : bool
        If ``True``, include under- and over-flow bins
    dd : bool
        If True, use ``histogramdd`` style return.

    See Also
    --------
    dask_histogram.AggHistogram.to_dask_array

    Returns
    -------
    Union[Tuple[DaskCollection, List[DaskCollection]], Tuple[DaskCollection, ...]]
        The first return is always the bin counts. If `dd` is ``True``
        the second return is a list where each element is an array of
        bin edges for each axis. If `dd` is ``False``, the bin edge
        arrays will not be stored in a list (`histogram2d` style
        return).

    """
    from dask.array import Array, asarray

    name = f"to-dask-array-{tokenize(agghist)}"
    thehist = agghist.histref
    if isinstance(thehist, tuple):
        thehist = bh.Histogram(
            *agghist.histref[0], storage=agghist.histref[1], metadata=agghist.histref[2]
        )
    zeros = (0,) * thehist.ndim
    dsk = {(name, *zeros): (lambda x, f: x.to_numpy(flow=f)[0], agghist.key, flow)}
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=(agghist,))
    shape = thehist.shape
    if flow:
        shape = tuple(i + 2 for i in shape)
    int_storage = thehist.storage_type in (
        bh.storage.Int64,
        bh.storage.AtomicInt64,
    )
    dt = int if int_storage else float
    c = Array(graph, name=name, shape=shape, chunks=shape, dtype=dt)
    axes = thehist.axes

    if flow:
        edges = [
            asarray(np.concatenate([[-np.inf], ax.edges, [np.inf]])) for ax in axes
        ]
    else:
        edges = [asarray(ax.edges) for ax in axes]
    if dd:
        return c, edges
    return (c, *tuple(edges))


class BinaryOpAgg:
    def __init__(
        self,
        func: Callable[[Any, Any], Any],
        name: str | None = None,
    ) -> None:
        self.func = func
        self.__name__ = func.__name__ if name is None else name

    def __call__(self, a: AggHistogram, b: AggHistogram) -> AggHistogram:
        name = f"{self.__name__}-hist-{tokenize(a, b)}"
        deps = []
        if is_dask_collection(a):
            deps.append(a)
        if is_dask_collection(b):
            deps.append(b)
        k1 = a.__dask_keys__()[0] if is_dask_collection(a) else a
        k2 = b.__dask_keys__()[0] if is_dask_collection(b) else b
        llg = {(name, 0): (self.func, k1, k2)}
        g = HighLevelGraph.from_collections(name, llg, dependencies=deps)
        try:
            ref = a.histref
        except AttributeError:
            ref = b.histref
        return AggHistogram(g, name, histref=ref)


_iadd = BinaryOpAgg(operator.iadd, name="add")
_isub = BinaryOpAgg(operator.isub, name="sub")
_imul = BinaryOpAgg(operator.imul, name="mul")
_itruediv = BinaryOpAgg(operator.itruediv, name="div")


def factory(
    *data: DaskCollection,
    histref: bh.Histogram | tuple | None = None,
    axes: Sequence[bh.axis.Axis] | None = None,
    storage: bh.storage.Storage | None = None,
    weights: DaskCollection | None = None,
    sample: DaskCollection | None = None,
    split_every: int | None = None,
    keep_partitioned: bool = False,
) -> AggHistogram:
    """Daskified Histogram collection factory function.

    Given some data represented by Dask collections and the
    characteristics of a histogram (either a reference
    :py:obj:`boost_histogram.Histogram` object or a set of axes), this
    routine will create an :py:obj:`AggHistogram` or
    :py:obj:`PartitionedHistogram` collection.

    Parameters
    ----------
    *data : DaskCollection
        The data to histogram. The supported forms of input data:

        * Single one dimensional dask array or Series: for creating a
          1D histogram.
        * Single multidimensional dask array or DataFrame: for
          creating multidimensional histograms.
        * Multiple one dimensional dask arrays or Series: for creating
          multidimensional histograms.
    histref : bh.Histogram or tuple, optional
        A reference histogram object, required if `axes` is not used.
        The dimensionality of `histref` must be compatible with the
        input data. If a tuple, it must be three elements where
        element one is a tuple of axes, element two is the storage,
        and element three is the metadata.
    axes : Sequence[bh.axis.Axis], optional
        The axes of the histogram, required if `histref` is not used.
        The total number of axes must be equal to the number of
        dimensions of the resulting histogram given the structure of
        `data`.
    storage : bh.storage.Storage, optional
        Storage type of the histogram, only compatible with use of the
        `axes` argument.
    weights : DaskCollection, optional
        Weights associated with the `data`. The partitioning/chunking
        of the weights must be compatible with the input data.
    sample : DaskCollection, optional
        Provide samples if the histogram storage allows it. The
        partitioning/chunking of the samples must be compatible with
        the input data.
    split_every : int, optional
        How many blocks to use in each split during aggregation.
    keep_partitioned : bool, optional
        **Deprecated argument**. Use :py:func:`partitioned_factory`.

    Returns
    -------
    AggHistogram or PartitionedHistogram
        The resulting histogram collection.

    Raises
    ------
    ValueError
        If `histref` and `axes` are both not ``None``, or if `storage`
        is used with `histref`.

    Examples
    --------
    Creating a three dimensional histogram using the `axes` argument:

    >>> import boost_histogram as bh
    >>> import dask.array as da
    >>> import dask_histogram as dh
    >>> x = da.random.uniform(size=(10000,), chunks=(2000,))
    >>> y = da.random.uniform(size=(10000,), chunks=(2000,))
    >>> z = da.random.uniform(size=(10000,), chunks=(2000,))
    >>> bins = [
    ...    [0.0, 0.3, 0.4, 0.5, 1.0],
    ...    [0.0, 0.1, 0.2, 0.8, 1.0],
    ...    [0.0, 0.2, 0.3, 0.4, 1.0],
    ... ]
    >>> axes = [bh.axis.Variable(b) for b in bins]
    >>> h = dh.factory(x, y, z, axes=axes)
    >>> h.shape
    (4, 4, 4)
    >>> h.compute()
    Histogram(
      Variable([0, 0.3, 0.4, 0.5, 1]),
      Variable([0, 0.1, 0.2, 0.8, 1]),
      Variable([0, 0.2, 0.3, 0.4, 1]),
      storage=Double()) # Sum: 10000.0

    Creating a weighted one dimensional histogram with the `histref`
    argument, then converting to the dask.array histogramming return
    style.

    >>> x = da.random.uniform(size=(10000,), chunks=(2000,))
    >>> w = da.random.uniform(size=(10000,), chunks=(2000,))
    >>> ref = bh.Histogram(bh.axis.Regular(10, 0, 1))
    >>> h = dh.factory(x, histref=ref, weights=w)
    >>> counts, edges = h.to_dask_array()
    >>> counts
    dask.array<to-dask-array, shape=(10,), dtype=float64, chunksize=(10,), chunktype=numpy.ndarray>
    >>> edges
    dask.array<array, shape=(11,), dtype=float64, chunksize=(11,), chunktype=numpy.ndarray>

    """
    if keep_partitioned:
        raise ValueError(
            "keep_partitioned=True is no longer supported; "
            "use dask_histogram.partitioned_factory."
        )
    ph = partitioned_factory(
        *data,
        histref=histref,
        axes=axes,
        storage=storage,
        weights=weights,
        sample=sample,
    )
    return ph.collapse(split_every=split_every)


def partitioned_factory(
    *data: DaskCollection,
    histref: bh.Histogram | tuple | None = None,
    axes: Sequence[bh.axis.Axis] | None = None,
    storage: bh.storage.Storage | None = None,
    weights: DaskCollection | None = None,
    sample: DaskCollection | None = None,
) -> PartitionedHistogram:
    """Daskified Histogram collection factory function; keep partitioned.

    This is a version of the :py:func:`factory` function that **remains
    partitioned**. The :py:func:`factory` function includes a step in the
    task graph that aggregates all partitions into a single final
    histogram.

    See Also
    --------
    dask_histogram.factory

    """
    if histref is None and axes is None:
        raise ValueError("Either histref or axes must be defined.")
    if histref is not None and storage is not None:
        raise ValueError("Storage cannot be defined along with histref.")
    elif histref is None:
        if storage is None:
            storage = bh.storage.Double()
        histref = bh.Histogram(*axes, storage=storage)  # type: ignore

    return _partitioned_histogram(
        *data, histref=histref, weights=weights, sample=sample
    )


def is_dask_awkward_like(x: Any) -> bool:
    """Check if an object is Awkward collection like.

    Parameters
    ----------
    x : Any
        The object of interest.

    Returns
    -------
    bool
        ``True`` if `x` is an Awkward Dask collection.

    """
    return (
        hasattr(x, "__dask_graph__") and hasattr(x, "layout") and hasattr(x, "fields")
    )
