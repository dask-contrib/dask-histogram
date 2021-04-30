import numpy as np
import pytest

from dask_histogram.bins import bins_style, bins_range_styles
from dask_histogram.bins import BinsStyle, RangeStyle


def test_bins_styles_scalar():

    # Valid
    assert bins_style(D=1, bins=5) == BinsStyle.SingleScalar
    assert bins_style(D=2, bins=(2, 5)) == BinsStyle.MultiScalar
    assert bins_style(D=2, bins=[3, 4]) == BinsStyle.MultiScalar

    # Invalid
    with pytest.raises(
        ValueError,
        match="Total number of bins definitions must be equal to the dimensionality of the histogram.",
    ):
        bins_style(D=3, bins=[2, 3])
    with pytest.raises(
        ValueError,
        match="Total number of bins definitions must be equal to the dimensionality of the histogram.",
    ):
        bins_style(D=4, bins=[2, 3, 4, 7, 8])


def test_bins_styles_sequence():
    assert bins_style(D=1, bins=np.array([1, 2, 3])) == BinsStyle.SingleSequence
    assert bins_style(D=1, bins=[1, 2, 3]) == BinsStyle.SingleSequence
    assert bins_style(D=1, bins=(4, 5, 6)) == BinsStyle.SingleSequence
    assert bins_style(D=2, bins=[[1, 2, 3], [4, 5, 7]]) == BinsStyle.MultiSequence
    assert (
        bins_style(D=3, bins=[[1, 2], [1, 2, 3], [4, 5, 7]]) == BinsStyle.MultiSequence
    )
    assert BinsStyle.MultiSequence == bins_style(
        D=2,
        bins=(np.array([1.1, 2.2]), np.array([2.2, 4.4, 6.6])),
    )

    with pytest.raises(
        ValueError,
        match="Total number of bins definitions must be equal to the dimensionality of the histogram.",
    ):
        bins_style(D=1, bins=[[1, 2], [4, 5]])
    with pytest.raises(
        ValueError,
        match="Total number of bins definitions must be equal to the dimensionality of the histogram.",
    ):
        bins_style(D=3, bins=[[1, 2], [4, 5]])

    with pytest.raises(
        ValueError,
        match="Total number of bins definitions must be equal to the dimensionality of the histogram.",
    ):
        bins_style(
            D=3,
            bins=(np.array([1.1, 2.2]), np.array([2.2, 4.4, 6.6])),
        )


def test_bins_range_styles():
    bs, rs = bins_range_styles(D=2, bins=(3, 4), range=((0, 1), (0, 1)))
    assert bs == BinsStyle.MultiScalar
    assert rs == RangeStyle.MultiPair

    bs, rs = bins_range_styles(D=1, bins=10, range=(0, 1))
    assert bs == BinsStyle.SingleScalar
    assert rs == RangeStyle.SinglePair

    bs, rs = bins_range_styles(D=2, bins=[[1, 2, 3], [4, 5, 6]], range=None)
    assert bs == BinsStyle.MultiSequence
    assert rs == RangeStyle.IsNone

    bs, rs = bins_range_styles(D=1, bins=[1, 2, 3], range=None)
    assert bs == BinsStyle.SingleSequence
    assert rs == RangeStyle.IsNone

    with pytest.raises(
        ValueError,
        match="range cannot be None when bins argument is a scalar or sequence of scalars.",
    ):
        bins_range_styles(D=1, bins=3, range=None)

    with pytest.raises(
        ValueError,
        match="range cannot be None when bins argument is a scalar or sequence of scalars.",
    ):
        bins_range_styles(D=2, bins=3, range=None)

    with pytest.raises(
        ValueError,
        match="range cannot be None when bins argument is a scalar or sequence of scalars.",
    ):
        bins_range_styles(D=2, bins=(3, 8), range=None)

    with pytest.raises(
        ValueError,
        match="For a single scalar bin definition, one range tuple must be defined.",
    ):
        bins_range_styles(D=1, bins=5, range=((2, 3), (4, 5)))
