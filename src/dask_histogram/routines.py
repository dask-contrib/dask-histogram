"""Routines for staging histogram computations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .typing import DaskCollection
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
