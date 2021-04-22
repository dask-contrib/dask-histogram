"""Dask compatible boost-histogram API."""

from __future__ import annotations

import operator
from typing import Any, List, Optional

import boost_histogram as bh
import numpy as np
from dask.delayed import Delayed, delayed

import dask_histogram


def _tree_reduce(hists: List[Delayed]) -> Delayed:
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
def _blocked_fill_1d(data: Any, meta_hist: Histogram, weight: Optional[Any] = None):
    """Single delayed (1D) histogram concrete fill."""
    hfb = Histogram(*meta_hist.axes, storage=meta_hist._storage_type())
    hfb.concrete_fill(data, weight=weight)
    return hfb


def _fill_1d(data: Any, meta_hist: Histogram, weight: Optional[Any] = None) -> Delayed:
    """Prepare a set of delayed one dimensional histogram fills."""
    data = data.to_delayed()
    if weight is None:
        hists = [_blocked_fill_1d(a, meta_hist) for a in data]
    else:
        weights = weight.to_delayed()
        hists = [_blocked_fill_1d(a, meta_hist, w) for a, w in zip(data, weights)]
    return delayed(sum)(hists)


# @delayed
# def blocked_fill_nd_rectangular(
#     sample: np.ndarray,
#     meta_hist: Histogram,
#     weight: Optional[np.ndarray] = None,
# ):
#     """Single delayed (nD) histogram concrete fill with transpose call."""
#     hfb = Histogram(*meta_hist.axes, storage=meta_hist._storage_type())
#     print(sample.T.shape)
#     hfb.concrete_fill(*(sample.T), weight=weight)
#     return hfb


# def fill_nd_rectangular(
#     sample: da.Array,
#     meta_hist: Histogram,
#     weight: Optional[Any] = None,
# ) -> Delayed:
#     """Fill nD histogram given a rectangular (multi-column) sample."""
#     delayeds = sample.to_delayed()
#     print(len(meta_hist.axes))
#     print(sample.shape)
#     if weight is None:
#         hists = [
#             blocked_fill_nd_rectangular(s, meta_hist=meta_hist, weight=None)
#             for s in delayeds
#         ]
#     else:
#         weights = weight.to_delayed()
#         if len(weights) != len(delayeds):
#             raise ValueError(
#                 "data sample and weight must have the same number of chunks"
#             )
#         hists = [
#             blocked_fill_nd_rectangular(s, meta_hist=meta_hist, weight=w)
#             for s, w in zip(delayeds, weights)
#         ]
#     return _tree_reduce(hists)


@delayed
def _blocked_fill_nd_rectangular(
    sample: Any, meta_hist: Histogram, weight: Optional[Any]
):
    hfb = Histogram(*meta_hist.axes, storage=meta_hist._storage_type())
    sample = sample.T
    hfb.concrete_fill(*sample, weight=weight)
    return hfb


@delayed
def _blocked_fill_nd_multiarg(
    *args: np.ndarray, meta_hist: Histogram, weight: Optional[Any] = None
):
    """Single delayed (nD) histogram concrete fill."""
    hfb = Histogram(*meta_hist.axes, storage=meta_hist._storage_type())
    hfb.concrete_fill(*args, weight=weight)
    return hfb


def _fill_nd_multiarg(
    *samples: Any,
    meta_hist: Histogram,
    weight: Optional[Any] = None,
) -> Delayed:
    """Fill nD histogram given a multiarg (vectors) sample."""
    D = len(samples)
    # each entry is data along a specific dimension
    samples = [a.to_delayed() for a in samples]
    # check that all dimensions are chunked identically
    npartitions = len(samples[0])
    for i in range(1, D):
        if len(samples[i]) != npartitions:
            raise ValueError("All dimensions must be chunked identically")
    # We need to create a data structure that will connect coordinate
    # chunks. We loop over the number of partitions and connect the
    # ith chunk along each dimension (the loop over j is the loop over
    # the total number of dimensions).
    samples = [tuple(samples[j][i] for j in range(D)) for i in range(npartitions)]

    if weight is None:
        hists = [_blocked_fill_nd_multiarg(*d, meta_hist=meta_hist) for d in samples]
    else:
        weights = weight.to_delayed()
        if len(weights) != npartitions:
            raise ValueError(
                "data sample and weight must have the same number of chunks"
            )
        hists = [
            _blocked_fill_nd_multiarg(*d, meta_hist=meta_hist, weight=w)
            for d, w in zip(samples, weights)
        ]

    return delayed(sum)(hists)


def _fill_nd(*args: Any, meta_hist: Histogram, weight: Optional[Any] = None) -> Delayed:
    """Prepare a set of delayed n-dimensional histogram fills."""
    if len(args) == 1 and args[0].ndim == 1:
        return _fill_1d(args[0], meta_hist=meta_hist, weight=weight)
    elif len(args) == 1 and args[0].ndim > 1:
        # return fill_nd_rectangular(args[0], meta_hist=meta_hist, weight=weight)
        raise NotImplementedError("Rectangular input is not supported yet.")
    else:
        return _fill_nd_multiarg(*args, meta_hist=meta_hist, weight=weight)


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
        self, *args, weight: Optional[Any] = None, sample=None, threads=None
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
        new_fill = _fill_nd(*args, meta_hist=self, weight=weight)
        if self._dq is None:
            self._dq = new_fill
        else:
            self._dq = _tree_reduce([self._dq, new_fill])
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

        See :py:func:`dask.visualize` for available keyword arguments.

        """
        return self.to_delayed().visualize(**kwargs)
