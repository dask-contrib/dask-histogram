from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

from dask.array.core import Array
from dask.dataframe.core import DataFrame, Series
from numpy.typing import ArrayLike

BinType = Union[int, ArrayLike]
BinArg = Union[BinType, Sequence[BinType]]

RangeType = Optional[Tuple[float, float]]
RangeArg = Optional[Union[RangeType, Sequence[RangeType]]]

DaskCollection = Union[Array, Series, DataFrame]
