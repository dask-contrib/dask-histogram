"""Routines for staging histogram computations with a dask.array like API."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import boost_histogram as bh
import dask.array as da
from dask.base import is_dask_collection
from dask.utils import is_arraylike, is_dataframe_like

from dask_histogram.core import AggHistogram, factory

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
    normed: bool | None = None,
    weights: DaskCollection | None = None,
    density: bool = False,
    *,
    histogram: Any | None = None,
    storage: bh.storage.Storage = bh.storage.Double(),
    threads: int | None = None,
) -> AggHistogram | tuple[da.Array, ...]:
    """FIXME: Short description.

    FIXME: Long description.

    Parameters
    ----------
    x : DaskCollection
        FIXME: Add docs.
    bins : BinType
        FIXME: Add docs.
    range : RangeType
        FIXME: Add docs.
    normed : Optional[bool]
        FIXME: Add docs.
    weights : Optional[DaskCollection]
        FIXME: Add docs.
    density : bool
        FIXME: Add docs.
    histogram : Optional[Any]
        FIXME: Add docs.
    storage : bh.storage.Storage
        FIXME: Add docs.
    threads : Optional[int]
        FIXME: Add docs.

    Returns
    -------
    Union[AggHistogram, Tuple[da.Array, ...]]
        FIXME: Add docs.

    Examples
    --------
    FIXME: Add docs.

    """
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
        return h.to_dask_array(flow=False, dd=False)  # type: ignore
    return h


def histogram2d(
    x: DaskCollection,
    y: DaskCollection,
    bins: BinArg = 10,
    range: RangeArg = None,
    normed: bool | None = None,
    weights: DaskCollection | None = None,
    density: bool = False,
    *,
    histogram: Any | None = None,
    storage: bh.storage.Storage = bh.storage.Double(),
    threads: int | None = None,
) -> AggHistogram | tuple[da.Array, ...]:
    """FIXME: Short description.

    FIXME: Long description.

    Parameters
    ----------
    x : DaskCollection
        FIXME: Add docs.
    y : DaskCollection
        FIXME: Add docs.
    bins : BinArg
        FIXME: Add docs.
    range : RangeArg
        FIXME: Add docs.
    normed : Optional[bool]
        FIXME: Add docs.
    weights : Optional[DaskCollection]
        FIXME: Add docs.
    density : bool
        FIXME: Add docs.
    histogram : Optional[Any]
        FIXME: Add docs.
    storage : bh.storage.Storage
        FIXME: Add docs.
    threads : Optional[int]
        FIXME: Add docs.

    Returns
    -------
    Union[AggHistogram, Tuple[da.Array, ...]]
        FIXME: Add docs.

    Examples
    --------
    FIXME: Add docs.

    """
    h = histogramdd(
        (x, y),
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
        return h.to_dask_array(flow=False, dd=False)  # type: ignore
    return h


def histogramdd(
    a: DaskCollection | tuple[DaskCollection, ...],
    bins: BinArg = 10,
    range: RangeArg = None,
    normed: bool | None = None,
    weights: DaskCollection | None = None,
    density: bool = False,
    *,
    histogram: Any | None = None,
    storage: bh.storage.Storage = bh.storage.Double(),
    threads: int | None = None,
) -> (AggHistogram | tuple[da.Array, ...] | tuple[da.Array, tuple[da.Array, ...]]):
    """Histogram Dask data in multiple dimensions.

    Parameters
    ----------
    a : DaskCollection or tuple[DaskCollection, ...]
        Data to histogram. Acceptable input data can be of the form:

        * A dask.array.Array of shape (N, D) where each row is a
          sample and each column is a specific coordinate for the
          samples.
        * A sequence of dask collections where each collection (e.g.
          array or series) contains all values for one coordinate of
          all samples.
    bins : sequence of arrays, int, or sequence of ints
        The bin specification.

        The possible binning configurations are:

        * A sequence of arrays describing the monotonically increasing
          bin edges along each dimension.
        * A single int describing the total number of bins that will
          be used in each dimension (this requires the `range`
          argument to be defined).
        * A sequence of ints describing the total number of bins to be
          used in each dimension (this requires the `range` argument
          to be defined).

        When bins are described by arrays, the rightmost edge is
        included. Bins described by arrays also allows for non-uniform
        bin widths.
    range : tuple(tuple(float, float), ...) optional
        A sequence of length D, each a (min, max) tuple giving the
        outer bin edges to be used if the edges are not given
        explicitly in `bins`. If defined, this argument is required to
        have an entry for each dimension. Unlike
        :func:`numpy.histogramdd`, if `bins` does not define bin
        edges, this argument is required (this function will not
        automatically use the min and max of of the value in a given
        dimension because the input data may be lazy in dask).
    normed : bool, optional
        An unsupported argument that has been deprecated in the NumPy
        API (preserved to maintain calls dependent on argument order).
    weights : dask.array.Array or dask.dataframe.Series, optional
        An array of values weighing each sample in the input data. The
        chunks of the weights must be identical to the chunking along
        the 0th (row) axis of the data sample.
    density : bool
        If ``False`` (default), the returned array represents the
        number of samples in each bin. If ``True``, the returned array
        represents the probability density function at each bin.
    histogram : dask_histogram.Histogram, optional
        If not ``None``, an AggHistogram collection object is
        returned, the default behavior is an array style return.
    storage : boost_histogram.storage.Storage
        Define the storage used by the :py:class:`Histogram` object.
    threads : int, optional
        Enable threading on :py:func:`Histogram.fill` calls.

    Returns
    -------
    AggHistogram or tuple[da.Array, ...] or tuple[da.Array, tuple[da.Array, ...]]
        FIXME: Add docs.

    Examples
    --------
    FIXME: Add docs.

    """
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
    if threads is not None:
        warnings.warn(
            "threads argument is not None; Dask may compete with boost-histogram "
            "for thread usage."
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
    axes: list[Any] = []
    for _, (b, r) in enumerate(zip(bins, range)):  # type: ignore
        if r is None:
            axes.append(bh.axis.Variable(b))  # type: ignore
        else:
            axes.append(bh.axis.Regular(bins=b, start=r[0], stop=r[1]))  # type: ignore

    # Finally create the histogram object.
    ah = factory(*a, axes=axes, storage=storage, weights=weights)

    if histogram is not None:
        return ah
    return ah.to_dask_array(flow=False, dd=True)
