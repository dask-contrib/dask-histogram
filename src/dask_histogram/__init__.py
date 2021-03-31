from .version import version as __version__
from .boost import Histogram

version_info = tuple(__version__.split("."))

__all__ = ("__version__", "Histogram")
