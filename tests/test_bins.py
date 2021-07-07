import numpy as np
import pytest

from dask_histogram.bins import (
    BinsStyle,
    RangeStyle,
    bins_range_styles,
    bins_style,
    normalize_bins_range,
)


def test_bins_styles_scalar():
    # Valid
    assert bins_style(ndim=1, bins=5) is BinsStyle.SingleScalar
    assert bins_style(ndim=2, bins=(2, 5)) is BinsStyle.MultiScalar
    assert bins_style(ndim=2, bins=[3, 4]) is BinsStyle.MultiScalar

    # Invalid
    with pytest.raises(
        ValueError,
        match="Total number of bins definitions must be equal to the dimensionality of the histogram.",
    ):
        bins_style(ndim=3, bins=[2, 3])
    with pytest.raises(
        ValueError,
        match="Total number of bins definitions must be equal to the dimensionality of the histogram.",
    ):
        bins_style(ndim=4, bins=[2, 3, 4, 7, 8])


def test_bins_styles_sequence():
    assert bins_style(ndim=1, bins=np.array([1, 2, 3])) is BinsStyle.SingleSequence
    assert bins_style(ndim=1, bins=[1, 2, 3]) is BinsStyle.SingleSequence
    assert bins_style(ndim=1, bins=(4, 5, 6)) is BinsStyle.SingleSequence
    assert bins_style(ndim=2, bins=[[1, 2, 3], [4, 5, 7]]) is BinsStyle.MultiSequence

    bins = [[1, 2, 6, 7], [1, 2, 3], [4, 7, 11, 12, 13]]
    assert bins_style(ndim=3, bins=bins) is BinsStyle.MultiSequence

    bins = (np.array([1.1, 2.2]), np.array([2.2, 4.4, 6.6]))
    assert BinsStyle.MultiSequence is bins_style(ndim=2, bins=bins)

    with pytest.raises(
        ValueError,
        match="Total number of bins definitions must be equal to the dimensionality of the histogram.",
    ):
        bins_style(ndim=1, bins=[[1, 2], [4, 5]])
    with pytest.raises(
        ValueError,
        match="Total number of bins definitions must be equal to the dimensionality of the histogram.",
    ):
        bins_style(ndim=3, bins=[[1, 2], [4, 5]])

    with pytest.raises(
        ValueError,
        match="Total number of bins definitions must be equal to the dimensionality of the histogram.",
    ):
        bins = (np.array([1.1, 2.2]), np.array([2.2, 4.4, 6.6]))
        bins_style(ndim=3, bins=bins)


def test_bins_style_cannot_determine():
    bins = 3.3
    with pytest.raises(ValueError, match="Could not determine bin style from bins=3.3"):
        bins_style(ndim=1, bins=bins)


def test_bins_range_styles():
    bs, rs = bins_range_styles(ndim=2, bins=(3, 4), range=((0, 1), (0, 1)))
    assert bs is BinsStyle.MultiScalar
    assert rs is RangeStyle.MultiPair

    bs, rs = bins_range_styles(ndim=1, bins=10, range=(0, 1))
    assert bs is BinsStyle.SingleScalar
    assert rs is RangeStyle.SinglePair

    bs, rs = bins_range_styles(ndim=2, bins=[[1, 2, 3], [4, 5, 6]], range=None)
    assert bs is BinsStyle.MultiSequence
    assert rs is RangeStyle.IsNone

    bs, rs = bins_range_styles(ndim=1, bins=[1, 2, 3], range=None)
    assert bs is BinsStyle.SingleSequence
    assert rs is RangeStyle.IsNone

    bins = np.array([[1, 2, 3], [2, 5, 6]])
    bs, rs = bins_range_styles(ndim=2, bins=bins, range=None)
    assert bs is BinsStyle.MultiSequence
    assert rs is RangeStyle.IsNone

    with pytest.raises(
        ValueError,
        match="range cannot be None when bins argument is a scalar or sequence of scalars.",
    ):
        bins_range_styles(ndim=1, bins=3, range=None)

    with pytest.raises(
        ValueError,
        match="range cannot be None when bins argument is a scalar or sequence of scalars.",
    ):
        bins_range_styles(ndim=2, bins=3, range=None)

    with pytest.raises(
        ValueError,
        match="range cannot be None when bins argument is a scalar or sequence of scalars.",
    ):
        bins_range_styles(ndim=2, bins=(3, 8), range=None)

    with pytest.raises(
        ValueError,
        match="For a single scalar bin definition, one range tuple must be defined.",
    ):
        bins_range_styles(ndim=1, bins=5, range=((2, 3), (4, 5)))


def test_normalize_bins_range():
    # 1D, scalar bins, single range
    ndim = 1
    bins, range = 5, (3, 3)
    bins, range = normalize_bins_range(ndim, bins, range)
    assert bins == (5,)
    assert range == ((3, 3),)

    # 1D, sequence bins, no range
    ndim = 1
    bins, range = [1, 2, 3], None
    bins, range = normalize_bins_range(ndim, bins, range)
    assert bins == ([1, 2, 3],)
    assert range == (None,)

    # 2D, singel scalar bins, single range
    ndim = 2
    bins, range = 5, (3, 3)
    bins, range = normalize_bins_range(ndim, bins, range)
    assert bins == (5, 5)
    assert range == ((3, 3), (3, 3))

    # 2D, sequence bins, no range
    ndim = 2
    bins, range = [[1, 2, 3], [4, 5, 6]], None
    bins, range = normalize_bins_range(ndim, bins, range)
    assert bins == [[1, 2, 3], [4, 5, 6]]
    assert range == (None, None)

    # 2D, numpy arrays as bins, no range
    ndim = 2
    bins, range = (np.array([1, 2, 3]), np.array([4, 5, 6])), None
    bins, range = normalize_bins_range(ndim, bins, range)
    assert len(bins) == 2
    np.testing.assert_array_equal(bins[0], np.array([1, 2, 3]))
    np.testing.assert_array_equal(bins[1], np.array([4, 5, 6]))

    # 3D, single multidim numpy array as bins, no range
    ndim = 3
    bins, range = np.array([[1, 2, 3], [4, 5, 6], [1, 5, 6]]), None
    bins, range = normalize_bins_range(ndim, bins, range)
    assert len(bins) == 3
    assert range == (None, None, None)
    np.testing.assert_array_equal(bins[0], np.array([1, 2, 3]))
    np.testing.assert_array_equal(bins[1], np.array([4, 5, 6]))
    np.testing.assert_array_equal(bins[2], np.array([1, 5, 6]))
    np.testing.assert_array_equal(bins, np.array([[1, 2, 3], [4, 5, 6], [1, 5, 6]]))

    # bad number of bins/range
    ndim = 2
    bins = (2, 2)
    range = ((0, 1),) * 3
    with pytest.raises(
        ValueError, match="bins and range arguments must be the same length"
    ):
        normalize_bins_range(ndim, bins, range)
