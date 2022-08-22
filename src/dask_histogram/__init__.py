"""Histogramming with Dask collections."""

import boost_histogram.axis as _axis
import boost_histogram.storage as _storage

from dask_histogram.core import AggHistogram, PartitionedHistogram, factory
from dask_histogram.routines import histogram, histogram2d, histogramdd

__version__ = "2022.8.0"
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
    "AggHistogram",
    "PartitionedHistogram",
    "axis",
    "factory",
    "histogram",
    "histogram2d",
    "histogramdd",
    "storage",
)
