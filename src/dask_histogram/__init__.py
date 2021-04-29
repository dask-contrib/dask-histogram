from .version import version as __version__
from .boost import Histogram
from .array import histogram, histogram2d, histogramdd

import boost_histogram.storage as storage
import boost_histogram.axis as axis

version_info = tuple(__version__.split("."))

__all__ = (
    "__version__",
    "Histogram",
    "storage",
    "axis",
    "histogram",
    "histogram2d",
    "histogramdd",
)
