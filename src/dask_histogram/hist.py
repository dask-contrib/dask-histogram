"""Dask compatible hist API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union

import hist as _hist
import hist.storage as storage
import hist.axis as axis
import numpy as np

from dask.delayed import delayed
import dask_histogram

from .boost import Histogram

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from .boost import DaskCollection
else:
    DaskCollection = object


T = TypeVar("T", bound="Hist")


__all__ = (
    "Hist",
    "axis",
    "storage",
)


class Hist(_hist.BaseHist, family=dask_histogram):
    """Lazy fillable histogram."""

    def __init__(
        self,
        *args,
        storage: Optional[Union[storage.Storage, str]] = None,
        metadata: Any = None,
        data: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize a lazy-fillable Hist object."""
        super().__init__(*args, storage=storage, metadata=metadata, data=data)
        self._dhist = Histogram(*args, storage=storage)

    def fill(
        self: T,
        *args: DaskCollection,
        weight: Optional[Any] = None,
        sample: Optional[Any] = None,
        threads: Optional[int] = None,
        **kwargs: DaskCollection,
    ) -> T:
        """Stage fill with one or more Dask collections.

        Parameters
        ----------
        *args : one or more Dask collections
            Provide one or more Dask collection per dimension, or a
            single columnar Dask collection. see
            :py:func:`dask_histogram.Histogram.fill`.
        weight : dask.array.Array or dask.dataframe.Series, optional
            Weights associated with each sample. The weights must be
            chunked/partitioned in a way compatible with the dataset.
        sample : Any
            Unsupported argument from boost_histogram.Histogram.fill.
        threads : Any
            Unsupported argument from boost_histogram.Histogram.fill.
        **kwargs : Dask collection(s)
            Named dask collections

        Examples
        --------
        >>> import dask_histogram as dh

        """
        data_dict = {
            self._name_to_index(k) if isinstance(k, str) else k: v
            for k, v in kwargs.items()
        }

        if set(data_dict) != set(range(len(args), self.ndim)):
            raise TypeError("All axes must be accounted for in fill")

        data = (data_dict[i] for i in range(len(args), self.ndim))

        total_data = (*args, *data)
        self._dhist.fill(*total_data, weight=weight, sample=sample, threads=threads)

    def __repr__(self):
        """Text representation of Hist."""
        return self._dhist.__repr__()

    def concrete_fill(
        self: T,
        *args: ArrayLike,
        weight: Optional[ArrayLike] = None,
        sample: Optional[ArrayLike] = None,
        threads: Optional[int] = None,
        **kwargs,
    ) -> T:
        """Insert concrete data (non Dask collections) into histogram."""
        return super().fill(
            *args,
            weight=weight,
            sample=sample,
            threads=threads,
            **kwargs,
        )

    def compute(self) -> T:
        """Compute staged fills."""
        self[...] = self._dhist.compute().view(flow=True)
        return self
