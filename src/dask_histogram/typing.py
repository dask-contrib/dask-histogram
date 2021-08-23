from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    import dask.array as da
    import dask.dataframe as dd
    from numpy.typing import ArrayLike

    BinType = Union[int, ArrayLike]
    BinArg = Union[BinType, Sequence[BinType]]

    RangeType = Optional[Tuple[float, float]]
    RangeArg = Optional[Union[RangeType, Sequence[RangeType]]]

    DaskCollection = Union[da.Array, dd.Series, dd.DataFrame]
else:
    DaskCollection = object
    ArrayLike = object
