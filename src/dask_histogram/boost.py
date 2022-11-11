"""Dask-ified boost-histogram like API."""

from __future__ import annotations

import operator
import warnings
from typing import TYPE_CHECKING, Any

import boost_histogram as bh
import boost_histogram.axis as axis
import boost_histogram.storage as storage
import dask.array as da
from dask.base import is_dask_collection
from dask.delayed import Delayed, delayed
from dask.utils import is_arraylike, is_dataframe_like

from dask_histogram.bins import normalize_bins_range
from dask_histogram.core import AggHistogram, factory

if TYPE_CHECKING:
    from dask_histogram.typing import (
        BinArg,
        BinType,
        DaskCollection,
        RangeArg,
        RangeType,
    )

import dask_histogram

__all__ = ("Histogram", "histogram", "histogram2d", "histogramdd")


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
    dask_histogram.histogram
    dask_histogram.histogram2d
    dask_histogram.histogramdd

    Examples
    --------
    A two dimensional histogram with one fixed bin width axis and
    another variable bin width axis:

    Note that (for convenience) the ``boost_histogram.axis`` namespace
    is mirrored as ``dask_histogram.axis`` and the
    ``boost_histogram.storage`` namespace is mirrored as
    ``dask_histogram.storage``.

    >>> import dask.array as da
    >>> import dask_histogram.boost as dhb
    >>> x = da.random.standard_normal(size=(1000,), chunks=200)
    >>> y = da.random.standard_normal(size=(1000,), chunks=200)
    >>> w = da.random.uniform(0.2, 0.8, size=(1000,), chunks=200)
    >>> h = dhb.Histogram(
    ...     dhb.axis.Regular(10, -3, 3),
    ...     dhb.axis.Variable([-3, -2, -1, 0, 1.1, 2.2, 3.3]),
    ...     storage=dhb.storage.Weight()
    ... ).fill(x, y, weight=w).compute()

    """

    def __init__(
        self,
        *axes: bh.axis.Axis,
        storage: bh.storage.Storage = bh.storage.Double(),
        metadata: Any = None,
    ) -> None:
        """Construct a Histogram object."""
        super().__init__(*axes, storage=storage, metadata=metadata)
        self._staged: AggHistogram | None = None

    def concrete_fill(
        self,
        *args: Any,
        weight: Any | None = None,
        sample: Any | None = None,
        threads: int | None = None,
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

    def fill(  # type: ignore
        self,
        *args: DaskCollection,
        weight: DaskCollection | None = None,
        sample: Any | None = None,
        threads: Any | None = None,
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

        weight : dask.array.Array or dask.dataframe.Series, optional
            Weights associated with each sample. The weights must be
            chunked/partitioned in a way compatible with the dataset.
        sample : dask.array.Array or dask.dataframe.Series, optional
            Provide samples if the histogram storage allows it. The
            partitioning/chunking of the samples must be compatible
            with the input data.
        threads : int, optional
            Ignored argument kept for compatibility with boost-histogram.
            We let Dask have complete control over threads.

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
                threads=None,
            )

        if len(args) == 1 and args[0].ndim == 1:
            pass
        elif len(args) == 1 and args[0].ndim == 2:
            pass
        elif len(args) > 1:
            pass
        else:
            raise ValueError(f"Cannot interpret input data: {args}")

        new_fill = factory(*args, histref=self, weights=weight, sample=sample)
        if self._staged is not None:
            self._staged += new_fill
        else:
            self._staged = new_fill

        return self

    def compute(self) -> Histogram:
        """Compute any staged (delayed) fills.

        Returns
        -------
        dask_histogram.Histogram
            Concrete histogram with all staged (delayed) fills executed.

        """
        if self._staged is None:
            return self
        if not self.empty():
            result_view = self.view(flow=True) + self._staged.compute().view(flow=True)
        else:
            result_view = self._staged.compute().view(flow=True)
        self[...] = result_view
        self._staged = None
        return self

    def clear_fills(self) -> None:
        """Drop any uncomputed fills."""
        self._staged = None

    def staged_fills(self) -> bool:
        """Check if histogram has staged fills.

        Returns
        -------
        bool
            True if the object contains staged delayed fills.

        """
        return self._staged is not None

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
        staged, then no concrete fill computations will be triggered
        and the eventual computed object from the Delayed object will
        be a ``dask_histogram.boost.Histogram`` object. If staged
        fills exist the computed object from the Delayed object will
        be a ``boost_histogram.Histogram``.

        Returns
        -------
        dask.delayed.Delayed
            Wrapping of the histogram as a delayed object.

        Examples
        --------
        >>> import dask_histogram.boost as dhb
        >>> import dask
        >>> h = dhb.Histogram(dhb.axis.Regular(10, -3, 3))
        >>> x = da.random.standard_normal(size=(100,), chunks=(20,))
        >>> h.fill(x)
        Histogram(Regular(10, -3, 3), storage=Double()) # (has staged fills)
        >>> h, = dask.compute(h.to_delayed())

        """
        if self.staged_fills() and not self.empty():
            return delayed(operator.add)(self._staged, delayed(self))
        elif self._staged is not None:
            return self._staged.to_delayed()
        return delayed(bh.Histogram(self))

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
            storage=self.storage_type(),
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

    def visualize(self, *args: Any, **kwargs: Any) -> Any:
        """Render the task graph with graphviz.

        See :py:func:`dask.visualize` for supported keyword arguments.

        """
        return self.to_delayed().visualize(*args, **kwargs)

    def agg_histogram(self) -> AggHistogram | None:
        if self._staged is None:
            warnings.warn("No staged AggHistogram; returning None")
        return self._staged

    def to_dask_array(self, flow: bool = False, dd: bool = True) -> Any:
        """Convert to dask.array style of return arrays.

        Edges are converted to match NumPy standards, with upper edge
        inclusive, unlike boost-histogram, where upper edge is
        exclusive.

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
        if self._staged is not None:
            return self._staged.to_dask_array(flow=flow, dd=dd)
        else:
            counts, edges = self.to_numpy(flow=flow, dd=True, view=False)
            counts = da.from_array(counts)
            edges = [da.from_array(ea) for ea in edges]  # type: ignore
            if dd:
                return counts, edges  # type: ignore
            return tuple([counts, *edges])  # type: ignore


def histogramdd(
    a: DaskCollection | tuple[DaskCollection, ...],
    bins: BinArg = 10,
    range: RangeArg = None,
    normed: bool | None = None,
    weights: DaskCollection | None = None,
    density: bool = False,
    *,
    histogram: Any | None = None,
    storage: storage.Storage = storage.Double(),
    threads: int | None = None,
) -> Any:
    """Histogram Dask data in multiple dimensions.

    Parameters
    ----------
    a : dask collection or tuple of dask collections
        Data to histogram. Acceptable input data can be of the form:

        * A dask.array.Array of shape (N, D) where each row is a
          sample and each column is a specific coordinate for the
          data.
        * A sequence of dask collections where each collection (e.g.
          array or series) contains all values for one coordinate of
          all data.
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
        Ignored argument kept for compatibility with boost-histogram.
        We let Dask have complete control over threads.

    Returns
    -------
    tuple(dask.array.Array, tuple(dask.array.Array)) or Histogram
        The default return is the style of
        :func:`dask.array.histogramdd`: An array of bin contents and a
        tuple of edges arrays (one for each dimension). If the
        `histogram` argument is used then the return is a
        :obj:`dask_histogram.Histogram` object.

    See Also
    --------
    histogram
    histogram2d

    Examples
    --------
    Creating a three dimensional histogram with variable width bins in
    each dimension. First, using three 1D arrays for each coordinate:

    >>> import dask.array as da
    >>> import dask_histogram.boost as dhb
    >>> x = da.random.standard_normal(size=(10000,), chunks=(2000,))
    >>> y = da.random.standard_normal(size=(10000,), chunks=(2000,))
    >>> z = da.random.standard_normal(size=(10000,), chunks=(2000,))
    >>> bins = [
    ...    [-3, -2, 0, 1, 3],
    ...    [-3, -1, 1, 2, 3],
    ...    [-3, -2, 0, 2, 3],
    ... ]
    >>> h, edges = dhb.histogramdd((x, y, z), bins=bins)
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
    >>> import dask_histogram.boost as dhb
    >>> x = da.random.standard_normal(size=(10000,), chunks=(2000,))
    >>> y = da.random.standard_normal(size=(10000,), chunks=(2000,))
    >>> z = da.random.standard_normal(size=(10000,), chunks=(2000,))
    >>> bins = [
    ...    [-3, -2, 0, 1, 3],
    ...    [-3, -1, 1, 2, 3],
    ...    [-3, -2, 0, 2, 3],
    ... ]
    >>> h = dhb.histogramdd((x, y, z), bins=bins, histogram=dhb.Histogram)
    >>> h
    Histogram(
      Variable([-3, -2, 0, 1, 3]),
      Variable([-3, -1, 1, 2, 3]),
      Variable([-3, -2, 0, 2, 3]),
      storage=Double()) # (has staged fills)
    >>> h.staged_fills()
    True
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

    >>> import dask.array as da
    >>> import dask_histogram.boost as dhb
    >>> a = da.random.standard_normal(size=(10000, 3), chunks=(2000, 3))
    >>> w = da.random.uniform(0.5, 0.7, size=(10000,), chunks=2000)
    >>> bins = (7, 5, 6)
    >>> range = ((-3, 3), (-2.9, 2.9), (-3.1, 3.1))
    >>> h = dhb.histogramdd(
    ...     a,
    ...     bins=bins,
    ...     range=range,
    ...     weights=w,
    ...     histogram=dhb.Histogram,
    ...     storage=dhb.storage.Weight()
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
    for _, (b, r) in enumerate(zip(bins, range)):
        if r is None:
            axes.append(axis.Variable(b))  # type: ignore
        else:
            axes.append(axis.Regular(bins=b, start=r[0], stop=r[1]))  # type: ignore

    # Finally create and fill the histogram object.
    hist = Histogram(*axes, storage=storage).fill(*a, weight=weights)

    if histogram != Histogram:
        return hist.to_dask_array(flow=False, dd=True)
    return hist


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
    storage: storage.Storage = storage.Double(),
    threads: int | None = None,
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
    histogram : dask_histogram.Histogram, optional
        If `dh.Histogram`, object based output is enabled.
    storage : boost_histogram.storage.Storage
        Define the storage used by the :py:class:`Histogram` object.
    threads : int, optional
        Ignored argument kept for compatibility with boost-histogram.
        We let Dask have complete control over threads.

    Returns
    -------
    tuple(dask.array.Array, dask.array.Array, dask.array.Array) or Histogram
        The default return is the style of
        :func:`dask.array.histogram2d`: An array of bin contents, an
        array of the x-edges, and an array of the y-edges. If the
        `histogram` argument is used then the return is a
        :obj:`dask_histogram.Histogram` object.

    See Also
    --------
    histogram
    histogramdd

    Examples
    --------
    Uniform distributions along each dimension with the array return style:

    >>> import dask_histogram.boost as dhb
    >>> import dask.array as da
    >>> x = da.random.uniform(0.0, 1.0, size=(1000,), chunks=200)
    >>> y = da.random.uniform(0.4, 0.6, size=(1000,), chunks=200)
    >>> h, edgesx, edgesy = dhb.histogram2d(x, y, bins=(12, 4), range=((0, 1), (0.4, 0.6)))

    Now with the object return style:

    >>> h = dhb.histogram2d(
    ...     x, y, bins=(12, 4), range=((0, 1), (0.4, 0.6)), histogram=dhb.Histogram
    ... )

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
    >>> h, e1, e2 = dhb.histogram2d(
    ...     x, y, bins=[binsx, binsy], weights=w
    ... ) #  doctest: +SKIP

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
        return hist.to_dask_array(flow=False, dd=False)
    return hist


def histogram(
    x: DaskCollection,
    bins: BinType = 10,
    range: RangeType = None,
    normed: bool | None = None,
    weights: DaskCollection | None = None,
    density: bool = False,
    *,
    histogram: Any | None = None,
    storage: storage.Storage = storage.Double(),
    threads: int | None = None,
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
    histogram : dask_histogram.Histogram, optional
        If `dh.Histogram`, object based output is enabled.
    storage : boost_histogram.storage.Storage
        Define the storage used by the :py:class:`Histogram` object.
    threads : int, optional
        Ignored argument kept for compatibility with boost-histogram.
        We let Dask have complete control over threads.

    Returns
    -------
    tuple(dask.array.Array, dask.array.Array) or Histogram
        The default return is the style of
        :func:`dask.array.histogram`: An array of bin contents and an
        array of bin edges. If the `histogram` argument is used then
        the return is a :obj:`dask_histogram.Histogram` object.

    See Also
    --------
    histogram2d
    histogramdd

    Examples
    --------
    Gaussian distribution with object return style and ``Weight`` storage:

    >>> import dask_histogram.boost as dhb
    >>> import dask.array as da
    >>> x = da.random.standard_normal(size=(1000,), chunks=(250,))
    >>> h = dhb.histogram(
    ...     x, bins=10, range=(-3, 3), histogram=dhb.Histogram, storage=dhb.storage.Weight()
    ... )

    Now with variable width bins and the array return style:

    >>> bins = [-3, -2.2, -1.0, -0.2, 0.2, 1.2, 2.2, 3.2]
    >>> h, edges = dhb.histogram(x, bins=bins)

    Now with weights and the object return style:

    >>> w = da.random.uniform(0.0, 1.0, size=x.shape[0], chunks=x.chunksize[0])
    >>> h = dhb.histogram(x, bins=bins, weights=w, histogram=dhb.Histogram)
    >>> h
    Histogram(Variable([-3, -2.2, -1, -0.2, 0.2, 1.2, 2.2, 3.2]), storage=Double()) # (has staged fills)

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
    )

    if histogram != Histogram:
        return hist.to_dask_array(flow=False, dd=False)
    return hist
