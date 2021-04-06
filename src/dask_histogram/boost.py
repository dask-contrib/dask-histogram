"""Dask compatible boost-histogram API."""

from __future__ import annotations

import operator
from typing import Any, List, Optional

import boost_histogram as bh
import dask.array as da
import numpy as np
from dask.delayed import Delayed, delayed

import dask_histogram


def tree_aggregate(dhists: List[Delayed]) -> Delayed:
    """Tree summation of delayed histogram objects.

    Parameters
    ----------
    dhists : List[Delayed]
        Delayed histograms to be aggregated.

    Returns
    -------
    Delayed
        Final histogram aggregation.

    """
    hist_list = dhists
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
def blocked_fill_1d(
    data: np.ndarray, hist: Histogram, weight: Optional[da.Array] = None
):
    """Single delayed (1D) histogram concrete fill."""
    hist_for_block = Histogram(*hist.axes, storage=hist._storage_type())
    hist_for_block.concrete_fill(data, weight=weight)
    return hist_for_block


def fill_1d(
    data: da.Array, hist: Histogram, weight: Optional[da.Array] = None
) -> Delayed:
    """Prepare a set of delayed one dimensional histogram fills."""
    d_data = data.to_delayed()
    if weight is None:
        d_histograms = [blocked_fill_1d(a, hist) for a in d_data]
    else:
        d_weight = weight.to_delayed()
        d_histograms = [blocked_fill_1d(a, hist, w) for a, w in zip(d_data, d_weight)]
    return tree_aggregate(d_histograms)


@delayed
def blocked_fill_nd(
    *args: np.ndarray, hist: Histogram, weight: Optional[da.Array] = None
):
    """Single delayed (nD) histogram concrete fills."""
    hist_for_block = Histogram(*hist.axes, storage=hist._storage_type())
    hist_for_block.concrete_fill(*args, weight=weight)
    return hist_for_block


def fill_nd(
    *args: da.Array, hist: bh.Histogram, weight: Optional[da.Array] = None
) -> Delayed:
    """Prepare a set of delayed n-dimensional histogram fills."""
    # total number of dimensions
    D = len(args)

    # if D == 1 go to simpler implementation.
    if D == 1:
        return fill_1d(*args, hist, weight)

    # each entry is data along a specific dimension
    stack_of_delayeds = [a.to_delayed() for a in args]
    # we assume all dimensions are chunked identically
    npartitions = len(stack_of_delayeds[0])
    # we need to create a data structure that will connect coordinate
    # chunks we loop over the number of partitions and connect the
    # ith chunk along each dimension (loop over j is the loop over
    # the dimensions).
    partitioned_fused_coordinates = [
        tuple(stack_of_delayeds[j][i] for j in range(D)) for i in range(npartitions)
    ]

    if weight is None:
        d_histograms = [
            blocked_fill_nd(*d, hist=hist) for d in partitioned_fused_coordinates
        ]
    else:
        d_weight = weight.to_delayed()
        d_histograms = [
            blocked_fill_nd(*d, hist=hist, weight=w)
            for d, w in zip(partitioned_fused_coordinates, d_weight)
        ]

    return tree_aggregate(d_histograms)


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
        ---------
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
        new_fill = fill_nd(*args, hist=self, weight=weight)
        if self._dq is None:
            self._dq = new_fill
        else:
            self._dq = tree_aggregate([self._dq, new_fill])
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

    def pending_fills(self) -> bool:
        """Check if histogram has pending fills.

        Returns
        -------
        bool
            True of instance contains pending delayed fills.

        """
        return self._dq is not None

    def to_delayed(self) -> Delayed:
        """Histogram as a delayed object.

        Wraps the current state of the Histogram in
        :py:func:`dask.delayed.delayed` if no fills are pending;
        otherwise, the most downstream delayed Histogram is returned,
        such that

        .. code-block:: python

            dask.compute(h.to_delayed())

        yields a histogram with the same counts and variances yielded by

        .. code-block:: python

            h.compute()

        In both cases if ``h`` doesn't have any queued fill calls,
        then no concrete fill computations will be triggered.

        Returns
        -------
        dask.delayed.Delayed
            Wrapping of the histogram as a delayed object.

        """
        if self.pending_fills():
            return self._dq
        return delayed(self)
