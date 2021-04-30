import boost_histogram.axis as axis
import boost_histogram.storage as storage

from .array import histogram, histogram2d, histogramdd
from .boost import Histogram
from .version import version as __version__

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
