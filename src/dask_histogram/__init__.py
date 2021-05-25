"""Histogramming with Dask collections."""

import boost_histogram.axis as _axis
import boost_histogram.storage as _storage

from .boost import Histogram
from .routines import histogram, histogram2d, histogramdd
from .version import version as __version__

version_info = tuple(__version__.split("."))


axis = _axis
"""
module: Alias to boost_histogram.axis for import simplicity.
"""

storage = _storage
"""
module: Alias to boost_histogram.storage for import simplicity.
"""


__all__ = (
    "__version__",
    "Histogram",
    "axis",
    "histogram",
    "histogram2d",
    "histogramdd",
    "storage",
)
