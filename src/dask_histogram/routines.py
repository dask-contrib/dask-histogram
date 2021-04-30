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
    """Histogram data in multiple dimensions."""
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
            if not is_dask_collection(a):
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
