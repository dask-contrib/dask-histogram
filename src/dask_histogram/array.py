"""Dask compatible boost-histogram dask.array like API."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Sequence, Tuple, Type, Union

import boost_histogram.axis as _axis
import boost_histogram.storage as _storage
import dask.array as da
import numpy as np

from .boost import Histogram


class BinsStyle(Enum):
    """Styles for the bins argument in histogramming functions."""

    Undetermined = 0
    SingleScalar = 1
    MultiScalar = 2
    SingleSequence = 3
    MultiSequence = 4


class RangeStyle(Enum):
    """Styles for the range argument in histogramming functions."""

    Undetermined = 0
    IsNone = 1
    SingleTuple = 2
    MultiTuple = 3


def bins_and_range_styles(bins: Any, range: Any) -> Tuple[BinsStyle, RangeStyle]:
    """Determine the style of the bins and range arguments."""
    bins_style, range_style = BinsStyle.Undetermined, RangeStyle.Undetermined
    if isinstance(bins, int):
        bins_style = BinsStyle.SingleScalar
    elif isinstance(bins, (tuple, list)):
        if all(isinstance(b, int) for b in bins):
            bins_style = BinsStyle.MultiScalar
        else:
            bins_style = BinsStyle.MultiSequence
    elif isinstance(bins, np.ndarray):
        if bins.ndim == 1:
            bins_style = BinsStyle.SingleSequence

    return bins_style, range_style


def histogramdd(
    a: Tuple[Any, ...],
    bins: Union[int, Tuple[int, ...]] = 10,
    range: Optional[Sequence[Tuple[float, float]]] = None,
    normed: None = None,
    weights: Optional[Any] = None,
    density: bool = False,
    *,
    histogram: Optional[Type[Histogram]] = None,
    storage: _storage.Storage = _storage.Double(),
    threads: Optional[int] = None,
) -> Any:
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

    rank = len(a)

    scalar_bins = False
    if isinstance(bins, (list, tuple)):
        if len(bins) != rank:
            raise ValueError(
                "Total number of defined bins must be equal to the dimension of the sample."
            )
        scalar_bins = all(isinstance(b, int) for b in bins)
    if isinstance(bins, int):
        bins = (bins,) * rank
        scalar_bins = True

    if scalar_bins:
        if range is None:
            raise TypeError("Integer bins requires range definition(s) (non-None)")
        if len(range) != len(bins):
            raise TypeError(
                "bins and range must have the same number of definitions received; "
                f"{len(bins)} bins and {len(range)} ranges."
            )
    else:
        if len(bins) != rank:
            raise TypeError(
                f"Total number of binning definitions ({len(bins)}) is incompatible "
                f"with data dimensionality ({rank})."
            )

    axes = []
    for i, (b, r) in enumerate(zip(bins, range)):
        if r is None:
            axes.append(_axis.Variable(b))
        else:
            axes.append(_axis.Regular(bins=b, start=r[0], stop=r[1]))

    hist = Histogram(*axes, storage=storage).fill(*a, weight=weights)
    return hist


def histogram2d(
    x: Any,
    y: Any,
    bins: Union[int, Tuple[int, int]] = 10,
    range: Optional[Sequence[Tuple[float, float]]] = None,
    normed: None = None,
    weights: Optional[Any] = None,
    density: bool = False,
    *,
    histogram: Optional[Type[Histogram]] = None,
    storage: _storage.Storage = _storage.Double(),
    threads: Optional[int] = None,
) -> Any:
    """Histogram data in two dimensions."""
    pass


def histogram(
    a: Any,
    bins: int = 10,
    range: Optional[Tuple[float, float]] = None,
    normed: None = None,
    weights: Optional[Any] = None,
    density: bool = False,
    *,
    histogram: Optional[Type[Histogram]] = None,
    storage: Optional[_storage.Storage] = None,
    threads: Optional[int] = None,
) -> Any:
    """Histogram data in one dimension."""
    pass
