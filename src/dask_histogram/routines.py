"""Routines for staging histogram computations with a dask.array like API."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .typing import DaskCollection
else:
    DaskCollection = object

__all__ = ("histogramdd", "histogram2d", "histogram")


def histogram(*args, **kwargs):
    pass


def histogram2d(*args, **kwargs):
    pass


def histogramdd(*args, **kwargs):
    pass
