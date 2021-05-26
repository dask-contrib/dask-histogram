"""Dask compatible hist API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union

import hist as _hist
from hist.storage import Storage
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


@delayed
def _stitch(h: T, result: Histogram):
    h[...] = result.view()
    return h


class Hist(_hist.Hist, family=dask_histogram):
    """Lazy fillable histogram."""

    def __init__(
        self,
        *args,
        storage: Optional[Union[Storage, str]] = None,
        metadata: Any = None,
        data: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize a lazy-fillable Hist object."""
        super().__init__(*args, storage=storage, metadata=metadata, data=data)
        self._dhist = Histogram(*args, storage=self.storage)

    def fill(
        self: T,
        *args: DaskCollection,
        weight: Optional[Any] = None,
        sample: Optional[Any] = None,
        threads: Optional[int] = None,
        **kwargs,
    ) -> T:
        """Stage fill with one or more Dask collections."""
        pass

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
        self._dhist.compute()
        self[...] = self._dhist.view()
        return self
