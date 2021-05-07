"""Dask compatible boost-histogram API."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Optional, Union

import boost_histogram as bh
import dask.array as da
import numpy as np
from dask.delayed import Delayed, delayed
from dask.utils import is_arraylike, is_dataframe_like
from dask.base import is_dask_collection

if TYPE_CHECKING:
    import dask.dataframe as dd

    DaskCollection = Union[da.Array, dd.Series, dd.DataFrame]
else:
    DaskCollection = object

import dask_histogram

__all__ = ("Histogram",)


@delayed
def _blocked_fill_1d(
    data: Any,
    meta_hist: Histogram,
    weight: Optional[Any] = None,
):
    """Single delayed (1D) histogram concrete fill."""
    hfb = Histogram(*meta_hist.axes, storage=meta_hist._storage_type())
    hfb.concrete_fill(data, weight=weight)
    return hfb


@delayed
def _blocked_fill_rectangular(
    sample: Any,
    meta_hist: Histogram,
    weight: Optional[Any] = None,
):
    hfb = Histogram(*meta_hist.axes, storage=meta_hist._storage_type())
    hfb.concrete_fill(*(sample.T), weight=weight)
    return hfb


@delayed
def _blocked_fill_dataframe(
    sample: Any,
    meta_hist: Histogram,
    weight: Optional[Any] = None,
):
    hfb = Histogram(*meta_hist.axes, storage=meta_hist._storage_type())
    hfb.concrete_fill(*(sample[c] for c in sample.columns), weight=weight)
    return hfb


@delayed
def _blocked_fill_multiarg(
    *args: np.ndarray,
    meta_hist: Histogram,
    weight: Optional[Any] = None,
):
    """Single delayed (nD) histogram concrete fill."""
    hfb = Histogram(*meta_hist.axes, storage=meta_hist._storage_type())
    hfb.concrete_fill(*args, weight=weight)
    return hfb


def _fill_1d(
    data: DaskCollection,
    *,
    meta_hist: Histogram,
    weight: Optional[DaskCollection] = None,
) -> Delayed:
    """Fill a one dimensional histogram.

    This function is compatible with dask.array.Array objects and
    dask.dataframe.Series objects.

    """
    data = data.to_delayed()  # type: ignore
    if weight is None:
        hists = [_blocked_fill_1d(a, meta_hist) for a in data]
    else:
        weights = weight.to_delayed()
        hists = [_blocked_fill_1d(a, meta_hist, w) for a, w in zip(data, weights)]
    return delayed(sum)(hists)


def _fill_rectangular(
    sample: DaskCollection,
    *,
    meta_hist: Histogram,
    weight: Optional[DaskCollection] = None,
) -> Delayed:
    """Fill nD histogram given a rectangular (multi-column) sample.

    Array Input
    -----------

    If a multi-column dask.array.Array is passed to `fill`, we want to
    avoid having to compute the transpose of the _entire collection_
    (this may be an expensive and unncessary computation).

    For this to work the input data can be chunked only along the row
    axis; we convert the whole collection (nD array) to delayed which
    gives us a 2D array of Delayed objects with a shape of the form
    (n_row_chunks, n_column_chunks). We transpose this and take the
    first and only element along the n_column_chunks dimension. This
    gives us a list of Delayed objects along the row dimension (each
    object wraps a multidimensional NumPy array).

    Finally, we loop over the list of Delayed objects and compute the
    transpose on each chunk _as necessary_ when the materialized array
    (a subset of the original complete collection) is used.

    DataFrame Input
    ---------------

    DataFrames are a bit simpler; we just use to_delayed() and pass to
    the dedicated @delayed fill function.

    """
    if is_arraylike(sample):
        sample = sample.to_delayed().T[0]  # type: ignore
        ff = _blocked_fill_rectangular
    elif is_dataframe_like(sample):
        sample = sample.to_delayed()  # type: ignore
        ff = _blocked_fill_dataframe
    else:
        raise TypeError(
            f"sample must be dask array or dataframe like; found {type(sample)}"
        )

    if weight is None:
        hists = [ff(s, meta_hist=meta_hist, weight=None) for s in sample]
    else:
        weights = weight.to_delayed()
        if len(weights) != len(sample):
            raise ValueError("sample and weight must have the same number of chunks.")
        hists = [ff(s, meta_hist=meta_hist, weight=w) for s, w in zip(sample, weights)]

    return delayed(sum)(hists)


def _fill_multiarg(
    *samples: DaskCollection,
    meta_hist: Histogram,
    weight: Optional[DaskCollection] = None,
) -> Delayed:
    """Fill nD histogram given a multiarg (vectors) sample.

    This function is compatible with multiple one dimensional
    dask.array.Array objects as well as multiple dask.dataframe.Series
    objects; they just must have equally sized chunks/partitions.

    """
    D = len(samples)
    # each entry is data along a specific dimension
    delayed_samples = [a.to_delayed() for a in samples]
    # check that all dimensions are chunked identically
    npartitions = len(delayed_samples[0])
    for i in range(1, D):
        if len(delayed_samples[i]) != npartitions:
            raise ValueError("All dimensions must be chunked/partitioned identically.")
    # We need to create a data structure that will connect coordinate
    # chunks. We loop over the number of partitions and connect the
    # ith chunk along each dimension (the loop over j is the loop over
    # the total number of dimensions).
    delayed_samples = [
        tuple(delayed_samples[j][i] for j in range(D)) for i in range(npartitions)
    ]

    if weight is None:
        hists = [
            _blocked_fill_multiarg(*d, meta_hist=meta_hist) for d in delayed_samples
        ]
    else:
        weights = weight.to_delayed()
        if len(weights) != npartitions:
            raise ValueError(
                "sample and weight must have the same number of chunks/partitions."
            )
        hists = [
            _blocked_fill_multiarg(*d, meta_hist=meta_hist, weight=w)
            for d, w in zip(delayed_samples, weights)
        ]

    return delayed(sum)(hists)


class Histogram(bh.Histogram, family=dask_histogram):
    """Histogram object capable of lazy computation.

    Parameters
    ----------
    *axes : boost_histogram.axis.Axis
        Provide one or more Axis objects.
    storage : boost_histogram.storage.Storage, optional
        Select a storage to use in the histogram. The default storage
        type is :py:class:`boost_histogram.storage.Double`.
    metadata : Any
        Data that is passed along if a new histogram is created.

    See Also
    --------
    histogram
    histogram2d
    histogramdd

    Examples
    --------
    A two dimensional histogram with one fixed bin width axis and
    another variable bin width axis:

    Note that (for convenience) the ``boost_histogram.axis`` namespace
    is mirrored as ``dask_histogram.axis`` and the
    ``boost_histogram.storage`` namespace is mirrored as
    ``dask_histogram.storage``.

    >>> import dask.array as da
    >>> import dask_histogram as dh
    >>> x = da.random.standard_normal(size=(1000,), chunks=200)
    >>> y = da.random.standard_normal(size=(1000,), chunks=200)
    >>> w = da.random.uniform(0.2, 0.8, size=(1000,), chunks=200)
    >>> h = dh.Histogram(
    ...     dh.axis.Regular(10, -3, 3),
    ...     dh.axis.Variable([-3, -2, -1, 0, 1.1, 2.2, 3.3]),
    ...     storage=dh.storage.Weights()
    ... ).fill(x, y, weight=w).compute()

    """

    __slots__ = ("_dq",)

    def __init__(
        self,
        *axes: bh.axis.Axis,
        storage: bh.storage.Storage = bh.storage.Double(),
        metadata: Any = None,
    ) -> None:
        super().__init__(*axes, storage=storage, metadata=metadata)
        self._dq: Optional[Delayed] = None

    def concrete_fill(
        self, *args: Any, weight: Optional[Any] = None, sample=None, threads=None
    ) -> Histogram:
        """Fill the histogram with concrete data (not a Dask collection).

        Calls the super class fill function
        :py:func:`boost_histogram.Histogram.fill`.

        Parameters
        ----------
        *args : array_like
            Provide one value or array per dimension
        weight : array_like, optional
            Provide weights (only if the storage supports them)
        sample : array_like
            Provide samples (only if the storage supports them)
        threads : int, optional
            Fill with threads. Defaults to None, which does not
            activate threaded filling. Using 0 will automatically pick
            the number of available threads (usually two per core).

        Returns
        -------
        dask_histogram.Histogram
            Class instance now filled with concrete data.

        """
        if any(is_dask_collection(a) for a in args) or is_dask_collection(weight):
            raise TypeError(
                "concrete_fill does not support Dask collections, only materialized "
                "data; use the Histogram.fill method."
            )
        return super().fill(*args, weight=weight, sample=sample, threads=threads)

    def fill(
        self,
        *args: DaskCollection,
        weight: Optional[Any] = None,
        sample=None,
        threads=None,
    ) -> Histogram:
        """Stage a fill call using a Dask collection as input.

        If materialized NumPy ararys are passed to this function, all
        arguments are forwarded :func:`concrete_fill`.

        Parameters
        ----------
        *args : one or more Dask collections
            Provide one dask collection per dimension, or a single
            columnar Dask collection (DataFrame or 2D Array) where the
            total number of columns equals to the total number of
            histogram dimensions.

            * A single one dimensional collection
              (:obj:`dask.array.Array` or
              :obj:`dask.dataframe.Series`)
            * Multiple one dimensional collections, each representing
              one an array of one coordinate of the dataset to be
              histogrammed.
            * A single two dimensional collection
              (:obj:`dask.array.Array` or
              :obj:`dask.dataframe.DataFrame`), each column
              representing one coordinate of the dataset to be
              histogrammed.

            If multiple one dimensional arguments are passed (i.e. an
            `x` array and a `y` array for a two dimensional
            histogram), the collections must have equal
            chunking/partitioning.

            If a single two dimensional array is passed (i.e. an array
            of shape ``(2000, 3)`` for a three dimensional histogram),
            chunking can only exist along the 0th (row) axis.
            (coordinates cannot be separated by a chunk boundry, only
            whole individual samples can be separated).

        weight : dask.array.Array, optional
            Weights associated with each sample. The weights must be
            chunked/partitioned in a way compatible with the dataset.
        sample : Any
            Unsupported argument from boost_histogram.Histogram.fill.
        threads : Any
            Unsupported argument from boost_histogram.Histogram.fill.

        Returns
        -------
        dask_histogram.Histogram
            Class instance with a staged (delayed) fill added.

        """
        # Pass to concrete fill if non-dask-collection
        if all(not is_dask_collection(a) for a in args):
            return self.concrete_fill(
                *args,
                weight=weight,
                sample=sample,
                threads=threads,
            )

        if len(args) == 1 and args[0].ndim == 1:
            func = _fill_1d
        elif len(args) == 1 and args[0].ndim == 2:
            func = _fill_rectangular  # type: ignore
        elif len(args) > 1:
            func = _fill_multiarg  # type: ignore
        else:
            raise ValueError(f"Cannot interpret input data: {args}")

        new_fill = func(*args, meta_hist=self, weight=weight)

        if self._dq is None:
            self._dq = new_fill
        else:
            self._dq = delayed(sum)([self._dq, new_fill])
        return self

    def compute(self) -> Histogram:
        """Compute any staged (delayed) fills.

        Returns
        -------
        dask_histogram.Histogram
            Concrete histogram with all staged (delayed) fills executed.

        """
        if self._dq is None:
            return self
        if not self.empty():
            result_view = self.view(flow=True) + self._dq.compute().view(flow=True)
        else:
            result_view = self._dq.compute().view(flow=True)
        self[...] = result_view
        self._dq = None
        return self

    def clear_fills(self) -> None:
        """Drop any uncomputed fills."""
        self._dq = None

    def staged_fills(self) -> bool:
        """Check if histogram has staged fills.

        Returns
        -------
        bool
            True if the object contains staged delayed fills.

        """
        return self._dq is not None

    def to_delayed(self) -> Delayed:
        """Histogram as a delayed object.

        Wraps the current state of the Histogram in
        :py:func:`dask.delayed.delayed` if no fills are staged;
        otherwise, the most downstream delayed Histogram is returned,
        such that:

        .. code-block:: python

            dask.compute(h.to_delayed())

        will yield a histogram with the same counts and variances
        yielded by:

        .. code-block:: python

            h.compute()

        In both cases if ``h`` doesn't have any delayed fill calls
        staged, then no concrete fill computations will be triggered.

        Returns
        -------
        dask.delayed.Delayed
            Wrapping of the histogram as a delayed object.

        """
        if self.staged_fills() and not self.empty():
            return delayed(operator.add)(delayed(self), self._dq)
        elif self.staged_fills():
            return self._dq
        return delayed(self)

    def __repr__(self) -> str:
        """Text representation of the histogram.

        Mostly copied from the parent boost_histogram.Histogram class;
        appeneded to the end of the string information about staged
        fills.

        """
        newline = "\n  "
        sep = "," if len(self.axes) > 0 else ""
        ret = "{self.__class__.__name__}({newline}".format(
            self=self, newline=newline if len(self.axes) > 1 else ""
        )
        ret += f",{newline}".join(repr(ax) for ax in self.axes)
        ret += "{comma}{newline}storage={storage}".format(
            storage=self._storage_type(),
            newline=newline
            if len(self.axes) > 1
            else " "
            if len(self.axes) > 0
            else "",
            comma=sep,
        )
        ret += ")"
        outer = self.sum(flow=True)
        if outer:
            inner = self.sum(flow=False)
            ret += f" # Sum: {inner}"
            if inner != outer:
                ret += f" ({outer} with flow)"
        if self.staged_fills() and outer:
            ret += " (has staged fills)"
        elif self.staged_fills():
            ret += " # (has staged fills)"
        return ret

    def visualize(self, **kwargs) -> Any:
        """Render the task graph with graphviz.

        See :py:func:`dask.visualize` for supported keyword arguments.

        """
        return self.to_delayed().visualize(**kwargs)


# def _tree_reduce(hists: List[Delayed]) -> Delayed:
#     """Tree summation of delayed histogram objects.

#     Parameters
#     ----------
#     hists : List[Delayed]
#         Delayed histograms to be aggregated.

#     Returns
#     -------
#     Delayed
#         Final histogram aggregation.

#     """
#     hist_list = hists
#     while len(hist_list) > 1:
#         updated_list = []
#         # even N, do all
#         if len(hist_list) % 2 == 0:
#             for i in range(0, len(hist_list), 2):
#                 lazy_comp = delayed(operator.add)(hist_list[i], hist_list[i + 1])
#                 updated_list.append(lazy_comp)
#         # odd N, hold back the tail and add it later
#         else:
#             for i in range(0, len(hist_list[:-1]), 2):
#                 lazy_comp = delayed(operator.add)(hist_list[i], hist_list[i + 1])
#                 updated_list.append(lazy_comp)
#             updated_list.append(hist_list[-1])

#         hist_list = updated_list
#     return hist_list[0]
