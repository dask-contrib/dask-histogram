"""Routines for staging histogram computations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import dask.array as da

if TYPE_CHECKING:
    import dask.dataframe as dd

    DaskCollection = Union[da.Array, dd.Series, dd.DataFrame]
else:
    DaskCollection = object

__all__ = ("histogramdd", "histogram2d", "histogram")


def hist(*args, **kwargs):
    pass


def histogram(*args, **kwargs):
    pass


def histogram2d(*args, **kwargs):
    pass


def histogramdd(*args, **kwargs):
    pass
