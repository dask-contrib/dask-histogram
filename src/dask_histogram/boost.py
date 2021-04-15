"""Dask compatible boost-histogram API."""

from __future__ import annotations

import operator
from typing import Any, List, TypeVar, Optional

import boost_histogram as bh
import dask.array as da
import numpy as np
from dask.delayed import Delayed, delayed

import dask_histogram


DH = TypeVar("DH", bound="Histogram")


def tree_reduce(hists: List[Delayed]) -> Delayed:
    """Tree summation of delayed histogram objects.

    Parameters
    ----------
    hists : List[Delayed]
        Delayed histograms to be aggregated.

    Returns
    -------
    Delayed
        Final histogram aggregation.

    """
    hist_list = hists
    while len(hist_list) > 1:
        updated_list = []
        # even N, do all
        if len(hist_list) % 2 == 0:
            for i in range(0, len(hist_list), 2):
                lazy_comp = delayed(operator.add)(hist_list[i], hist_list[i + 1])
                updated_list.append(lazy_comp)
        # odd N, hold back the tail and add it later
        else:
            for i in range(0, len(hist_list[:-1]), 2):
                lazy_comp = delayed(operator.add)(hist_list[i], hist_list[i + 1])
                updated_list.append(lazy_comp)
            updated_list.append(hist_list[-1])

        hist_list = updated_list
    return hist_list[0]


@delayed
def blocked_fill_1d(data: np.ndarray, meta_hist: DH, weight: Optional[da.Array] = None):
    """Single delayed (1D) histogram concrete fill."""
    hfb = Histogram(*meta_hist.axes, storage=meta_hist._storage_type())
    hfb.concrete_fill(data, weight=weight)
    return hfb


def fill_1d(
    data: da.Array, meta_hist: DH, weight: Optional[da.Array] = None
) -> Delayed:
    """Prepare a set of delayed one dimensional histogram fills."""
    d_data = data.to_delayed()
    if weight is None:
        d_histograms = [blocked_fill_1d(a, meta_hist) for a in d_data]
    else:
        d_weight = weight.to_delayed()
        d_histograms = [
            blocked_fill_1d(a, meta_hist, w) for a, w in zip(d_data, d_weight)
        ]
    return tree_reduce(d_histograms)


@delayed
def blocked_fill_nd(
    *args: np.ndarray, meta_hist: DH, weight: Optional[da.Array] = None
):
    """Single delayed (nD) histogram concrete fill."""
    hfb = Histogram(*meta_hist.axes, storage=meta_hist._storage_type())
    hfb.concrete_fill(*args, weight=weight)
    return hfb


# @delayed
# def blocked_fill_nd_rectangular(
#     sample: np.ndarray,
#     meta_hist: DH,
#     weight: Optional[da.Array] = None,
# ):
#     """Single delayed (nD) histogram concrete fill with transpose call."""
#     hfb = Histogram(*meta_hist.axes, storage=meta_hist._storage_type())
#     print(sample.T.shape)
#     hfb.concrete_fill(*(sample.T), weight=weight)
#     return hfb


# def fill_nd_rectangular(
#     sample: da.Array,
#     meta_hist: DH,
#     weight: Optional[Any] = None,
# ) -> Delayed:
#     """Fill nD histogram given a rectangular (multi-column) sample."""
#     delayeds = sample.to_delayed()
#     print(len(meta_hist.axes))
#     print(sample.shape)
#     if weight is None:
#         d_histograms = [
#             blocked_fill_nd_rectangular(s, meta_hist=meta_hist, weight=None)
#             for s in delayeds
#         ]
#     else:
#         d_weight = weight.to_delayed()
#         if len(d_weight) != len(delayeds):
#             raise ValueError(
#                 "data sample and weight must have the same number of chunks"
#             )
#         d_histograms = [
#             blocked_fill_nd_rectangular(s, meta_hist=meta_hist, weight=w)
#             for s, w in zip(delayeds, d_weight)
#         ]
#     return tree_reduce(d_histograms)


def fill_nd_multiarg(
    *samples: da.Array,
    meta_hist: DH,
    weight: Optional[Any] = None,
) -> Delayed:
    """Fill nD histogram given a multiarg (vectors) sample."""
    D = len(samples)
    # each entry is data along a specific dimension
    stack_of_delayeds = [a.to_delayed() for a in samples]
    # check that all dimensions are chunked identically
    npartitions = len(stack_of_delayeds[0])
    for i in range(1, D):
        if len(stack_of_delayeds[i]) != npartitions:
            raise ValueError("All dimensions must be chunked identically")
    # we need to create a data structure that will connect coordinate
    # chunks we loop over the number of partitions and connect the
    # ith chunk along each dimension (loop over j is the loop over
    # the dimensions).
    partitioned_fused_coordinates = [
        tuple(stack_of_delayeds[j][i] for j in range(D)) for i in range(npartitions)
    ]

    if weight is None:
        d_histograms = [
            blocked_fill_nd(*d, meta_hist=meta_hist)
            for d in partitioned_fused_coordinates
        ]
    else:
        d_weight = weight.to_delayed()
        if len(d_weight) != npartitions:
            raise ValueError(
                "data sample and weight must have the same number of chunks"
            )
        d_histograms = [
            blocked_fill_nd(*d, meta_hist=meta_hist, weight=w)
            for d, w in zip(partitioned_fused_coordinates, d_weight)
        ]

    return tree_reduce(d_histograms)


def fill_nd(
    *args: da.Array, meta_hist: DH, weight: Optional[da.Array] = None
) -> Delayed:
    """Prepare a set of delayed n-dimensional histogram fills."""
    if len(args) == 1 and args[0].ndim == 1:
        return fill_1d(args[0], meta_hist=meta_hist, weight=weight)
    elif len(args) == 1 and args[0].ndim > 1:
        # return fill_nd_rectangular(args[0], meta_hist=meta_hist, weight=weight)
        raise NotImplementedError("Rectangular input is not supported yet.")
    else:
        return fill_nd_multiarg(*args, meta_hist=meta_hist, weight=weight)


class Histogram(bh.Histogram, family=dask_histogram):
    """Histogram capable of lazy computation."""

    __slots__ = ("_dq",)

    def __init__(
        self,
        *axes: bh.axis.Axis,
        storage: bh.storage.Storage = bh.storage.Double(),
        metadata: Any = None,
    ) -> None:
        """Construct new histogram fillable with Dask collections.

        Parameters
        ----------
        *axes : boost_histogram.axis.Axis
            Provide one or more Axis objects.
        storage : boost_histogram.storage, optional
            Select a storage to use in the histogram. The default
            storage type is :py:class:`bh.storage.Double`.
        metadata : Any
            Data that is passed along if a new histogram is created.

        """
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
        super().fill(*args, weight=weight, sample=sample, threads=threads)
        return self

    def fill(
        self, *args, weight: Optional[da.Array] = None, sample=None, threads=None
    ) -> Histogram:
        """Queue a fill call using a Dask collection as input.

        Parameters
        ----------
        *args : dask.array.Array
            Dask array for each dimension.
        weight : dask.array.Array, optional
            Weights associated with each sample.
        sample : Any
            Unsupported argument from boost_histogram.Histogram.fill
        threads : Any
            Unsupported argument from boost_histogram.Histogram.fill

        Returns
        -------
        dask_histogram.Histogram
            Class instance with a queued delayed fill added.

        """
        new_fill = fill_nd(*args, meta_hist=self, weight=weight)
        if self._dq is None:
            self._dq = new_fill
        else:
            self._dq = tree_reduce([self._dq, new_fill])
        return self

    def compute(self) -> Histogram:
        """Compute any queued (delayed) fills.

        Returns
        -------
        dask_histogram.Histogram
            Concrete histogram with all queued fills executed.

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

        In both cases if ``h`` doesn't have any queued fill calls,
        then no concrete fill computations will be triggered.

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

    def visualize(self, **kwargs) -> None:
        """Render the task graph with graphviz.

        See :py:func:`dask.visualize`.

        """
        return self.to_delayed().visualize(**kwargs)
