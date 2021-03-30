from typing import Optional

import boost_histogram as bh
import dask.array as da
import numpy as np
from dask.delayed import Delayed, delayed

import dask_histogram


@delayed
def blocked_fill_1d(
    data: np.ndarray, hist: bh.Histogram, weight: Optional[da.Array] = None
):
    hist_for_block = bh.Histogram(*hist.axes, storage=hist._storage_type())
    hist_for_block.fill(data, weight=weight)
    return hist_for_block


def fill_1d(
    data: da.Array, hist: bh.Histogram, weight: Optional[da.Array] = None
) -> Delayed:
    d_data = data.to_delayed()
    if weight is None:
        d_histograms = [blocked_fill_1d(a, hist) for a in d_data]
    else:
        d_weight = weight.to_delayed()
        d_histograms = [blocked_fill_1d(a, hist, w) for a, w in zip(d_data, d_weight)]
    s = delayed(sum)(d_histograms)
    return s


@delayed
def blocked_fill_nd(
    *args: np.ndarray, hist: bh.Histogram, weight: Optional[da.Array] = None
):
    hist_for_block = bh.Histogram(*hist.axes, storage=hist._storage_type())
    hist_for_block.fill(*args, weight=weight)
    return hist_for_block


def fill_nd(
    *args: da.Array, hist: bh.Histogram, weight: Optional[da.Array] = None
) -> Delayed:
    # total number of dimensions
    D = len(args)
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
    s = delayed(sum)(d_histograms)
    return s


class Histogram(bh.Histogram, family=dask_histogram):
    __slots__ = ("_dq",)

    def __init__(self, *axes, storage, metadata=None) -> None:
        """Construct new histogram fillable with Dask collections.

        Parameters
        ---------
        *axes : boost_histogram.Axis
            Provide one or more boost_histogram.Axes objects.
        storage : boost_histogram.storage
            Select a storage to use in the histogram.
        metadata : Any
            Data that is passed along if a new histogram is created.

        """
        super().__init__(*axes, storage=storage, metadata=metadata)
        self._dq = None

    def fill(self, *args, weight: Optional[da.Array] = None, sample=None, threads=None):
        """Queue up a fill call with a Dask collection.

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
        Histogram
            The class instance.

        """
        new_fill = fill_nd(*args, hist=self, weight=weight)
        if self._dq is None:
            self._dq = new_fill
        else:
            self._dq = delayed(sum)([self._dq, new_fill])
        return self

    def compute(self) -> None:
        """Compute any queued (delayed) fills."""
        if self._dq is None:
            return
        if not self.empty():
            result_view = self.view(flow=True) + self._dq.compute().view(flow=True)
        else:
            result_view = self._dq.compute().view(flow=True)
        self[...] = result_view
        self._dq = None

    @property
    def dq(self) -> Delayed:
        """Delayed object holding the queued fills."""
        return self._dq
