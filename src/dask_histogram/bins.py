"""Help determining bin definitions."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dask_histogram.typing import BinArg, BinType, RangeArg, RangeType


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
    SinglePair = 2
    MultiPair = 3


def bins_style(ndim: int, bins: BinArg) -> BinsStyle:
    """Determine bin style from a bins argument and histogram dimensions.

    Parameters
    ----------
    ndim : int
        Total dimensions of the eventual histogram.
    bins : BinType
        Raw bins argument.

    Returns
    -------
    BinsStyle
        The determined BinStyle.

    Raises
    ------
    ValueError
        If `bins` is not compatible with `ndim` or the style is
        undetermined.

    """
    if isinstance(bins, int):
        return BinsStyle.SingleScalar
    elif isinstance(bins, (tuple, list)):
        # all integers in the tuple of list
        if ndim == 1 and np.ndim(bins) == 1:
            return BinsStyle.SingleSequence
        if all(isinstance(b, int) for b in bins):
            if len(bins) != ndim and ndim != 1:
                raise ValueError(
                    "Total number of bins definitions must be equal to the "
                    "dimensionality of the histogram."
                )
            if ndim == 1:
                return BinsStyle.SingleSequence
            return BinsStyle.MultiScalar
        # sequence of sequences
        else:
            if len(bins) != ndim:
                raise ValueError(
                    "Total number of bins definitions must be equal to the "
                    "dimensionality of the histogram."
                )
            return BinsStyle.MultiSequence
    elif isinstance(bins, np.ndarray):
        if bins.ndim == 1:
            return BinsStyle.SingleSequence
        elif bins.ndim == 2:
            if len(bins) != ndim:
                raise ValueError(
                    "Total number of bins definitions must be equal to the "
                    "dimensionality of the histogram."
                )
            return BinsStyle.MultiSequence

    raise ValueError(f"Could not determine bin style from bins={bins}")


def bins_range_styles(
    ndim: int, bins: BinArg, range: RangeArg
) -> tuple[BinsStyle, RangeStyle]:
    """Determine the style of the bins and range arguments.

    Parameters
    ----------
    ndim : int
        The dimensionality of the histogram to be created by the bin
        and range defintions.
    bins : int, sequence if ints, array, or sequence of arrays
        Definition of the bins either by total number of bins in each
        dimension, or by the bin edges in each dimension.
    range : sequence of pairs, optional
        If bins are defined by the total number in each dimension, a
        range must be defined representing the left- and right-most
        edges of the axis. For a multidimensional histogram a single
        pair will represent a (min, max) in each dimension. For
        multiple pairs, the total number must be equal to the
        dimensionality of the histogram.

    Returns
    -------
    BinsStyle
        The style of the bins argument
    RangeStyle
        The style of the range argument

    """
    b_style = bins_style(ndim, bins)
    r_style = RangeStyle.Undetermined

    # If range is None we can return or raise if the bins are defined by scalars.
    if range is None:
        r_style = RangeStyle.IsNone
        if b_style in (BinsStyle.SingleSequence, BinsStyle.MultiSequence):
            return b_style, r_style
        else:
            raise ValueError(
                "range cannot be None when bins argument is a scalar or sequence of scalars."
            )

    if b_style is BinsStyle.SingleScalar:
        if len(range) != 2:
            raise ValueError(
                "For a single scalar bin definition, one range tuple must be defined."
            )
        if not isinstance(range[0], (int, float)) or not isinstance(
            range[1], (int, float)
        ):
            raise ValueError(
                "For a single scalar bin definition, one range tuple must be defined."
            )
        r_style = RangeStyle.SinglePair

    elif b_style is BinsStyle.MultiScalar:
        if len(range) != ndim:
            ValueError(
                "Total number of range pairs must be equal to the dimensionality of the histogram."
            )
        for entry in range:
            if len(entry) != 2:  # type: ignore
                raise ValueError("Each range definition must be a pair of numbers.")
        r_style = RangeStyle.MultiPair

    return b_style, r_style


def normalize_bins_range(
    ndim: int, bins: BinArg, range: RangeArg
) -> tuple[tuple[BinType, ...], tuple[RangeType, ...]]:
    """Normalize bins and range arguments to tuples.

    Parameters
    ----------
    ndim : int
        Total dimensions of the eventual histogram.
    bins : BinType
        Raw bins argument.
    range : RangeType
        Raw range argument

    Returns
    -------
    Tuple[BinType, RangeType]
        Normalized bins and range arguments.

    """
    b_style, r_style = bins_range_styles(ndim=ndim, bins=bins, range=range)

    if b_style is BinsStyle.SingleScalar:
        out_bins = (bins,) * ndim
    elif b_style is BinsStyle.SingleSequence:
        out_bins = (bins,) * ndim
    elif b_style is BinsStyle.MultiScalar:
        out_bins = tuple(bins)  # type: ignore
    elif b_style is BinsStyle.MultiSequence:
        out_bins = tuple(bins)  # type: ignore
    else:
        raise ValueError("incompatible bins argument")

    if r_style is RangeStyle.SinglePair:
        out_range = (range,) * ndim
    elif r_style is RangeStyle.IsNone:
        out_range = (None,) * ndim
    elif r_style is RangeStyle.MultiPair:
        out_range = tuple(range)  # type: ignore
    else:
        raise ValueError("incompatible range argument")

    if len(out_bins) != len(out_range):
        raise ValueError("bins and range arguments must be the same length")

    return out_bins, out_range  # type: ignore
