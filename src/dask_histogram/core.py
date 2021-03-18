from dask.delayed import delayed
import boost_histogram as bh


@delayed
def histogram(x, bins) -> bh.Histogram:
    pass
