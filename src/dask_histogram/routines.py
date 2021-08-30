"""Routines for staging histogram computations with a dask.array like API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import boost_histogram as bh
import dask.array as da
from dask.base import is_dask_collection
from dask.utils import is_arraylike, is_dataframe_like

from dask_histogram.core import AggHistogram
from dask_histogram.core import histogram as core_histogram

from .bins import normalize_bins_range

if TYPE_CHECKING:
    from .typing import BinArg, BinType, DaskCollection, RangeArg, RangeType
else:
    DaskCollection = object

__all__ = ("histogramdd", "histogram2d", "histogram")


def histogram(
    x: DaskCollection,
    bins: BinType = 10,
    range: RangeType = None,
    normed: Optional[bool] = None,
    weights: Optional[DaskCollection] = None,
    density: bool = False,
    *,
    histogram: Optional[Any] = None,
    storage: bh.storage.Storage = bh.storage.Double(),
    threads: Optional[int] = None,
) -> Union[AggHistogram, Tuple[da.Array, ...]]:
    h = histogramdd(
        (x,),
        bins=bins,
        range=range,
        normed=normed,
        weights=weights,
        density=density,
        histogram=True,
        storage=storage,
        threads=threads,
    )
    if histogram is None:
        return h.to_dask_array(flow=False, dd=False)
    return h


def histogram2d(*args, **kwargs):
    pass


def histogramdd(
    a: Union[DaskCollection, Tuple[DaskCollection, ...]],
    bins: BinArg = 10,
    range: RangeArg = None,
    normed: Optional[bool] = None,
    weights: Optional[DaskCollection] = None,
    density: bool = False,
    *,
    histogram: Optional[Any] = None,
    storage: bh.storage.Storage = bh.storage.Double(),
    threads: Optional[int] = None,
) -> Union[
    AggHistogram, Union[Tuple[da.Array, ...], Tuple[da.Array, Tuple[da.Array, ...]]]
]:
    # Check for invalid argument combinations.
    if normed is not None:
        raise KeyError(
            "normed=True is deprecated in NumPy and not supported by dask-histogram."
        )
    if density and histogram is not None:
        raise KeyError(
            "dask-histogram does not support the density keyword when returning a "
            "dask-histogram object."
        )

    # If input is a multidimensional array or dataframe, we wrap it in
    # a tuple that will be passed to fill and unrolled in the backend.
    if (is_arraylike(a) and a.ndim > 1) or is_dataframe_like(a):  # type: ignore
        ndim = a.shape[1]  # type: ignore
        a = (a,)  # type: ignore
    else:
        ndim = len(a)
        for entry in a:
            if not is_dask_collection(entry):
                raise ValueError(
                    "non-dask collection was passed; this function only supports dask "
                    "collections as input"
                )

    bins, range = normalize_bins_range(ndim, bins, range)

    # Create the axes based on the bins and range values.
    axes: List[Any] = []
    for _, (b, r) in enumerate(zip(bins, range)):  # type: ignore
        if r is None:
            axes.append(bh.axis.Variable(b))  # type: ignore
        else:
            axes.append(bh.axis.Regular(bins=b, start=r[0], stop=r[1]))  # type: ignore

    # Finally create the histogram object.
    ah = core_histogram(*a, axes=axes, storage=storage, weights=weights)

    if histogram is not None:
        return ah
    return ah.to_dask_array(flow=False, dd=True)
