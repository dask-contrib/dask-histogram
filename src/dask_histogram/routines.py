"""Routines for staging histogram computations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import boost_histogram.axis as _axis
import boost_histogram.storage as _storage
import dask.array as da
from dask.base import is_dask_collection
from dask.utils import is_arraylike, is_dataframe_like

if TYPE_CHECKING:
    import dask.dataframe as dd

    DaskCollection = Union[da.Array, dd.Series, dd.DataFrame]
else:
    DaskCollection = object

from .bins import BinType, RangeType, normalize_bins_range
from .boost import Histogram

__all__ = ("histogramdd", "histogram2d", "histogram")


def histogramdd(
    a: Union[DaskCollection, Tuple[DaskCollection, ...]],
    bins: BinType = 10,
    range: RangeType = None,
    normed: Optional[bool] = None,
    weights: Optional[DaskCollection] = None,
    density: bool = False,
    *,
    histogram: Optional[Any] = None,
    storage: _storage.Storage = _storage.Double(),
    threads: Optional[int] = None,
):
    """Histogram dask data in multiple dimensions.

    Parameters
    ----------
    a : dask collection or tuple of dask collections
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
        If `dh.Histogram`, object based output is enabled.
    storage : boost_histogram.storage.Storage
        Define the storage used by the :py:class:`Histogram` object.
    threads : int, optional
        Enable threading on :py:func:`Histogram.fill` calls.

    See Also
    --------
    histogram
    histogram2d

    Examples
    --------
    Creating a three dimensional histogram with variable width bins in
    each dimension. First, using three 1D arrays for each coordinate:

    >>> import dask.array as da
    >>> import dask_histogram as dh
    >>> x = da.random.standard_normal(size=(10000,), chunks=(2000,))
    >>> y = da.random.standard_normal(size=(10000,), chunks=(2000,))
    >>> z = da.random.standard_normal(size=(10000,), chunks=(2000,))
    >>> bins = [
    ...    [-3, -2, 0, 1, 3],
    ...    [-3, -1, 1, 2, 3],
    ...    [-3, -2, 0, 2, 3],
    ... ]
    >>> h, edges = dh.histogramdd((x, y, z), bins=bins)
    >>> type(h)
    <class 'dask.array.core.Array'>
    >>> h.shape
    (4, 4, 4)
    >>> len(edges)
    3

    Now the same histogram but instead of a
    :py:func:`dask.array.histogramdd` style return (which mirrors the
    return style of :py:func:`numpy.histogramdd`), we use the
    `histogram` argument to trigger the return of a
    :obj:`dask_histogram.Histogram` object:

    >>> import dask.array as da
    >>> import dask_histogram as dh
    >>> x = da.random.standard_normal(size=(10000,), chunks=(2000,))
    >>> y = da.random.standard_normal(size=(10000,), chunks=(2000,))
    >>> z = da.random.standard_normal(size=(10000,), chunks=(2000,))
    >>> bins = [
    ...    [-3, -2, 0, 1, 3],
    ...    [-3, -1, 1, 2, 3],
    ...    [-3, -2, 0, 2, 3],
    ... ]
    >>> h = dh.histogramdd((x, y, z), bins=bins, histogram=dh.Histogram)
    >>> h
    Histogram(
      Variable([-3, -2, 0, 1, 3]),
      Variable([-3, -1, 1, 2, 3]),
      Variable([-3, -2, 0, 2, 3]),
      storage=Double()) # (has staged fills)
    >>> h.staged_fills()
    True
    >>> h = h.compute()
    >>> h
    Histogram(
      Variable([-3, -2, 0, 1, 3]),
      Variable([-3, -1, 1, 2, 3]),
      Variable([-3, -2, 0, 2, 3]),
      storage=Double()) # Sum: 9919.0 (10000.0 with flow)

    Another 3D histogram example but with an alternative dataset form
    (a single array with three columns), fixed bin widths, sample
    weights, and usage of the boost-histogram ``Weight()`` storage:

    >>> import dask.array as da
    >>> import dask_histogram as dh
    >>> a = da.random.standard_normal(size=(10000, 3), chunks=(2000, 3))
    >>> w = da.random.uniform(0.5, 0.7, size=(10000,), chunks=2000)
    >>> bins = (7, 5, 6)
    >>> range = ((-3, 3), (-2.9, 2.9), (-3.1, 3.1))
    >>> h = dh.histogramdd(
    ...     a,
    ...     bins=bins,
    ...     range=range,
    ...     weights=w,
    ...     histogram=dh.Histogram,
    ...     storage=dh.storage.Weight()
    ... )
    >>> h
    Histogram(
      Regular(7, -3, 3),
      Regular(5, -2.9, 2.9),
      Regular(6, -3.1, 3.1),
      storage=Weight()) # Sum: WeightedSum(value=0, variance=0) (has staged fills)
    >>> h.staged_fills()
    True
    >>> h = h.compute()
    >>> h.staged_fills()
    False

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
    axes = []
    for _, (b, r) in enumerate(zip(bins, range)):  # type: ignore
        if r is None:
            axes.append(_axis.Variable(b))  # type: ignore
        else:
            axes.append(_axis.Regular(bins=b, start=r[0], stop=r[1]))  # type: ignore

    # Finally create and fill the histogram object.
    hist = Histogram(*axes, storage=storage).fill(*a, weight=weights)

    if histogram != Histogram:
        hist, edges = hist.to_dask_array(flow=False, dd=True)
        return hist, [da.asarray(entry) for entry in edges]
    return hist


def histogram2d(
    x: DaskCollection,
    y: DaskCollection,
    bins: BinType = 10,
    range: RangeType = None,
    normed: Optional[bool] = None,
    weights: Optional[DaskCollection] = None,
    density: bool = False,
    *,
    histogram: Optional[Any] = None,
    storage: _storage.Storage = _storage.Double(),
    threads: Optional[int] = None,
):
    """Histogram data in two dimensions.

    Parameters
    ----------
    x : dask.array.Array or dask.dataframe.Series
        Array representing the `x` coordinates of the data to the
        histogrammed.
    y : dask.array.Array or dask.dataframe.Series
        Array representing the `y` coordinates of the data to the
        histogrammed.
    bins : int, (int, int), array, (array, array), optional
        The bin specification:

        * If a singe int, both dimensions will that that number of bins
        * If a pair of ints, the first int is the total number of bins
          along the `x`-axis, and the second is the total number of
          bins along the `y`-axis.
        * If a single array, the array represents the bin edges along
          each dimension.
        * If a pair of arrays, the first array corresponds to the
          edges along `x`-axis, the second corresponds to the edges
          along the `y`-axis.
    range : ((float, float), (float, float)), optional
        If integers are passed to the `bins` argument, `range` is
        required to define the min and max of each axis, that is:
        `((xmin, xmax), (ymin, ymax))`.
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
        If `dh.Histogram`, object based output is enabled.
    storage : boost_histogram.storage.Storage
        Define the storage used by the :py:class:`Histogram` object.
    threads : int, optional
        Enable threading on :py:func:`Histogram.fill` calls.

    See Also
    --------
    histogram
    histogramdd

    Examples
    --------
    FIXME: Add docs.

    """
    hist = histogramdd(
        (x, y),
        bins=bins,
        range=range,
        normed=normed,
        weights=weights,
        density=density,
        histogram=Histogram,
        storage=storage,
        threads=threads,
    )

    if histogram != Histogram:
        hist, edgex, edgey = hist.to_dask_array(flow=False, dd=False)
        return hist, da.asarray(edgex), da.asarray(edgey)
    return hist


def histogram(
    x: DaskCollection,
    bins: BinType = 10,
    range: RangeType = None,
    normed: Optional[bool] = None,
    weights: Optional[DaskCollection] = None,
    density: bool = False,
    *,
    histogram: Optional[Any] = None,
    storage: _storage.Storage = _storage.Double(),
    threads: Optional[int] = None,
):
    """Histogram dask data in one dimension.

    Parameters
    ----------
    x : dask.array.Array or dask.dataframe.Series
        Data to be histogrammed.
    bins : int or sequence of scalars.
        If `bins` is an int, it defines the total number of bins to be
        used (this requires the `range` argument to be defined). If
        `bins` is a sequence of scalars (e.g. an array) then it
        defines the bin edges.
    range : (float, float)
        The minimum and maximum of the histogram axis.
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
        If `dh.Histogram`, object based output is enabled.
    storage : boost_histogram.storage.Storage
        Define the storage used by the :py:class:`Histogram` object.
    threads : int, optional
        Enable threading on :py:func:`Histogram.fill` calls.

    See Also
    --------
    histogram2d
    histogramdd

    Examples
    --------
    FIXME: Add docs.

    """
    hist = histogramdd(
        (x,),
        bins=bins,
        range=range,
        normed=normed,
        weights=weights,
        density=density,
        histogram=Histogram,
        storage=storage,
        threads=threads,
    )

    if histogram != Histogram:
        hist, edges = hist.to_dask_array(flow=False, dd=False)
        return hist, da.asarray(edges)
    return hist
