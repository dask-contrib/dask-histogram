"""Routines for staging histogram computations."""

from __future__ import annotations

import boost_histogram.axis as _axis
import boost_histogram.storage as _storage
import dask.array as da
from dask.base import is_dask_collection

from .bins import normalize_bins_range
from .boost import Histogram


def histogramdd(
    a,
    bins=10,
    range=None,
    normed=None,
    weights=None,
    density=False,
    *,
    histogram=None,
    storage=_storage.Double(),
    threads=None,
):
    """Histogram dask data in multiple dimensions.

    Parameters
    ----------
    a : dask.array.Array or tuple of dask collections
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
          be used in each dimension (this requires the ``range``
          argument to be defined).
        * A sequence of ints describing the total number of bins to be
          used in each dimension (this requires the ``range`` argument
          to be defined).

        When bins are described by arrays, the rightmost edge is
        included. Bins described by arrays also allows for non-uniform
        bin widths.
    range : sequence of pairs, optional
        A sequence of length D, each a (min, max) tuple giving the
        outer bin edges to be used if the edges are not given
        explicitly in ``bins``. If defined, this argument is required
        to have an entry for each dimension. Unlike
        :func:`numpy.histogramdd`, if `bins` does not define bin
        edges, this argument is required (this function will not
        automatically use the min and max of of the value in a given
        dimension because the input data may be lazy in dask).
    normed : bool, optional
        An unsupported argument that has been deprecated in the NumPy
        API (preserved to maintain calls dependent on argument order).
    weights : dask collection, optional
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

    # If the input data is a 2D matrix with one column for each
    # coordinate, we must transpose the array such that each row is a
    # coordinate. This maintains compatibility with NumPy's design.
    if isinstance(a, da.Array) and a.ndim > 1:
        a = a.T
    else:
        for entry in a:
            if not is_dask_collection(entry):
                raise ValueError(
                    "non-dask collection was passed; this function only supports dask "
                    "collections as input"
                )

    # Total number of dimensions is based on the structure of the
    # input data.
    ndim = len(a)
    bins, range = normalize_bins_range(ndim, bins, range)

    # Create the axes based on the bins and range values.
    axes = []
    for i, (b, r) in enumerate(zip(bins, range)):
        if r is None:
            axes.append(_axis.Variable(b))
        else:
            axes.append(_axis.Regular(bins=b, start=r[0], stop=r[1]))

    # Finally create and fill the histogram object.
    hist = Histogram(*axes, storage=storage).fill(*a, weight=weights)
    return hist


def histogram2d(
    x,
    y,
    bins=10,
    range=None,
    normed=None,
    weights=None,
    density=False,
    *,
    histogram=None,
    storage=_storage.Double(),
    threads=None,
):
    """Histogram data in two dimensions.

    Parameters
    ----------
    x : dask.array.Array
        Array representing the `x` coordinates of the points to the
        histogrammed.
    y : dask.array.Array
        Array representing the `y` coordinates of the points to the
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
    range : tuple(tuple(float, float))
        If integers are passed to the `bins` argument, `range` is
        required to define the min and max of each axis, that is:
        `((xmin, xmax), (ymin, ymax))`.
    normed : bool, optional
        An unsupported argument that has been deprecated in the NumPy
        API (preserved to maintain calls dependent on argument order).
    weights : dask collection, optional
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

    Examples
    --------
    FIXME: Add docs.

    """
    return histogramdd(
        (x, y),
        bins=bins,
        range=range,
        normed=normed,
        weights=weights,
        density=density,
        histogram=histogram,
        storage=storage,
        threads=threads,
    )


def histogram(
    a,
    bins=10,
    range=None,
    normed=None,
    weights=None,
    density=False,
    *,
    histogram=None,
    storage=None,
    threads=None,
):
    """Histogram dask data in one dimension.

    Parameters
    ----------
    x : dask.array.Array
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
    weights : dask collection, optional
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

    Examples
    --------
    FIXME: Add docs.

    """
    return histogramdd(
        (a,),
        bins=bins,
        range=range,
        normed=normed,
        weights=weights,
        density=density,
        histogram=histogram,
        storage=storage,
        threads=threads,
    )
