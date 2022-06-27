"""Routines for staging histogram computations with a dask.array like API."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import boost_histogram as bh
from dask.base import is_dask_collection
from dask.utils import is_arraylike, is_dataframe_like

from dask_histogram.bins import normalize_bins_range
from dask_histogram.core import factory

if TYPE_CHECKING:
    from dask_histogram.typing import (
        BinArg,
        BinType,
        DaskCollection,
        RangeArg,
        RangeType,
    )
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
    split_every: int | None = None,
) -> Any:
    """Histogram Dask data in one dimension.

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
    histogram : Any, optional
        If not ``None``, a collection instance is returned instead of
        the array style return.
    storage : boost_histogram.storage.Storage
        Define the storage used by the :py:class:`Histogram` object.
    threads : int, optional
        Ignored argument kept for compatibility with boost-histogram.
        We let Dask have complete control over threads.

    Returns
    -------
    tuple(dask.array.Array, dask.array.Array) or AggHistogram
        The default return is the style of
        :func:`dask.array.histogram`: An array of bin contents and an
        array of bin edges. If the `histogram` argument is used then
        the return is a :obj:`dask_histogram.AggHistogram` collection
        instance.

    See Also
    --------
    histogram2d
    histogramdd

    Examples
    --------
    Gaussian distribution with object return style and ``Weight`` storage:

    >>> import dask_histogram as dh
    >>> import dask.array as da
    >>> import boost_histogram as bh
    >>> x = da.random.standard_normal(size=(1000,), chunks=(250,))
    >>> h = dh.histogram(
    ...     x, bins=10, range=(-3, 3), histogram=True, storage=bh.storage.Weight()
    ... )

    Now with variable width bins and the array return style:

    >>> bins = [-3, -2.2, -1.0, -0.2, 0.2, 1.2, 2.2, 3.2]
    >>> h, edges = dh.histogram(x, bins=bins)

    Now with weights and the object return style:

    >>> w = da.random.uniform(0.0, 1.0, size=x.shape[0], chunks=x.chunksize[0])
    >>> h = dh.histogram(x, bins=bins, weights=w, histogram=True)
    >>> h
    dask_histogram.AggHistogram<hist-aggregate, ndim=1, storage=Double()>

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
        split_every=split_every,
    )
    if histogram is None:
        return h.to_dask_array(flow=False, dd=False)
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
    split_every: int | None = None,
) -> Any:
    """Histogram Dask data in two dimensions.

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
    histogram : Any, optional
        If not ``None``, a collection instance is returned instead of
        the array style return.
    storage : boost_histogram.storage.Storage
        Define the storage used by the :py:class:`Histogram` object.
    threads : int, optional
        Ignored argument kept for compatibility with boost-histogram.
        We let Dask have complete control over threads.

    Returns
    -------
    tuple(dask.array.Array, dask.array.Array, dask.array.Array) or AggHistogram
        The default return is the style of
        :func:`dask.array.histogram2d`: An array of bin contents, an
        array of the x-edges, and an array of the y-edges. If the
        `histogram` argument is used then the return is a
        :obj:`dask_histogram.AggHistogram` collection instance.

    See Also
    --------
    histogram
    histogramdd

    Examples
    --------
    Uniform distributions along each dimension with the array return style:

    >>> import dask_histogram as dh
    >>> import dask.array as da
    >>> x = da.random.uniform(0.0, 1.0, size=(1000,), chunks=200)
    >>> y = da.random.uniform(0.4, 0.6, size=(1000,), chunks=200)
    >>> h, edgesx, edgesy = dh.histogram2d(x, y, bins=(12, 4), range=((0, 1), (0.4, 0.6)))

    Now with the collection object return style:

    >>> h = dh.histogram2d(
    ...     x, y, bins=(12, 4), range=((0, 1), (0.4, 0.6)), histogram=True
    ... )
    >>> type(h)
    <class 'dask_histogram.core.AggHistogram'>

    With variable bins and sample weights from a
    :py:obj:`dask.dataframe.Series` originating from a
    :py:obj:`dask.dataframe.DataFrame` column (`df` below must have
    `npartitions` equal to the size of the chunks in `x` and `y`):

    >>> x = da.random.uniform(0.0, 1.0, size=(1000,), chunks=200)
    >>> y = da.random.uniform(0.4, 0.6, size=(1000,), chunks=200)
    >>> df = dask_dataframe_factory()  # doctest: +SKIP
    >>> w = df["weights"]              # doctest: +SKIP
    >>> binsx = [0.0, 0.2, 0.6, 0.8, 1.0]
    >>> binsy = [0.40, 0.45, 0.50, 0.55, 0.60]
    >>> h, edges1, edges2 = dh.histogram2d(
    ...     x, y, bins=[binsx, binsy], weights=w
    ... ) #  doctest: +SKIP

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
        split_every=split_every,
    )
    if histogram is None:
        return h.to_dask_array(flow=False, dd=False)
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
    split_every: int | None = None,
) -> Any:
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
    histogram : Any, optional
        If not ``None``, a collection instance is returned instead of
        the array style return.
    storage : boost_histogram.storage.Storage
        Define the storage used by the :py:class:`Histogram` object.
    threads : int, optional
        Enable threading on :py:func:`Histogram.fill` calls.

    Returns
    -------
    tuple[da.Array, tuple[da.Array, ...]] or AggHistogram
        The default return is the style of
        :func:`dask.array.histogramdd`: An array of bin contents and
        arrays of bin edges. If the `histogram` argument is used then
        the return is a :obj:`dask_histogram.AggHistogram` collection
        instance.

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
    `histogram` argument to trigger the return of a collection object:

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
    >>> h = dh.histogramdd((x, y, z), bins=bins, histogram=True)
    >>> h
    dask_histogram.AggHistogram<hist-aggregate, ndim=3, storage=Double()>
    >>> h.ndim
    3
    >>> h = h.compute()
    >>> h  # doctest: +SKIP
    Histogram(
      Variable([-3, -2, 0, 1, 3]),
      Variable([-3, -1, 1, 2, 3]),
      Variable([-3, -2, 0, 2, 3]),
      storage=Double()) # Sum: 9919.0 (10000.0 with flow)

    Another 3D histogram example but with an alternative dataset form
    (a single array with three columns), fixed bin widths, sample
    weights, and usage of the boost-histogram ``Weight()`` storage:

    >>> a = da.random.standard_normal(size=(10000, 3), chunks=(2000, 3))
    >>> w = da.random.uniform(0.5, 0.7, size=(10000,), chunks=2000)
    >>> bins = (7, 5, 6)
    >>> range = ((-3, 3), (-2.9, 2.9), (-3.1, 3.1))
    >>> h = dh.histogramdd(
    ...     a,
    ...     bins=bins,
    ...     range=range,
    ...     weights=w,
    ...     histogram=True,
    ...     storage=dh.storage.Weight()
    ... )
    >>> h
    dask_histogram.AggHistogram<hist-aggregate, ndim=3, storage=Weight()>

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
    for _, (b, r) in enumerate(zip(bins, range)):
        if r is None:
            axes.append(bh.axis.Variable(b))  # type: ignore
        else:
            axes.append(bh.axis.Regular(bins=b, start=r[0], stop=r[1]))  # type: ignore

    # Finally create the histogram object.
    ah = factory(
        *a,
        axes=axes,
        storage=storage,
        weights=weights,
        split_every=split_every,
    )

    if histogram is not None:
        return ah
    return ah.to_dask_array(flow=False, dd=True)
