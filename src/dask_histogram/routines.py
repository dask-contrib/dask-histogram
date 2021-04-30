"""Routines for staging histogram computations."""

from __future__ import annotations

import boost_histogram.axis as _axis
import boost_histogram.storage as _storage
import dask.array as da
from dask.base import is_dask_collection

from .bins import BinsStyle, RangeStyle, bins_range_styles
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
    histogram : Any, optional
        If dh.Histogram, object based output is enabled.
    storage : FIXME: Add type.
        Define the storage used by the :py:class:`Histogram` object.
    threads : FIXME: Add type.
        Enable threading on :py:func:`Histogram.fill` calls.

    Raises
    ------
    ValueError
        FIXME: Add docs.
    KeyError
        FIXME: Add docs.

    Examples
    --------
    FIXME: Add docs.

    """
    if normed is not None:
        raise KeyError(
            "normed=True is deprecated in NumPy and not supported by dask-histogram."
        )
    if density and histogram is not None:
        raise KeyError(
            "dask-histogram does not support the density keyword when returning a "
            "dask-histogram object."
        )

    # In this case the input is a matrix where the each column is an
    # array representing one coordinate in the multidimensional
    # dataset; This is a NumPy design choice.
    if isinstance(a, da.Array):
        a = a.T
    else:
        for entry in a:
            if not is_dask_collection(entry):
                raise ValueError("non-dask collection was passed")

    D = len(a)

    b_style, r_style = bins_range_styles(D=D, bins=bins, range=range)
    if b_style == BinsStyle.SingleScalar:
        bins = (bins,) * D
    if r_style == RangeStyle.SinglePair:
        range = (range,) * D
    if b_style == BinsStyle.SingleSequence:
        bins = (bins,) * D

    axes = []
    for i, (b, r) in enumerate(zip(bins, range)):
        if r is None:
            axes.append(_axis.Variable(b))
        else:
            axes.append(_axis.Regular(bins=b, start=r[0], stop=r[1]))

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
    """Histogram data in two dimensions."""
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
    """Histogram data in one dimension."""
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
