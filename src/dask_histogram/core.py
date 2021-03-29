# boost-histogram and dask
import boost_histogram as bh
from dask.delayed import delayed


@delayed
def blocked_fill_1d(data, hist, weight=None):
    hist_for_block = bh.Histogram(*hist.axes, storage=hist._storage_type())
    hist_for_block.fill(data, weight=weight)
    return hist_for_block


def fill_1d(data, hist, weight=None):
    d_data = data.to_delayed()
    if weight is None:
        d_histograms = [blocked_fill_1d(a, hist) for a in d_data]
    else:
        d_weight = weight.to_delayed()
        d_histograms = [blocked_fill_1d(a, hist, w) for a, w in zip(d_data, d_weight)]
    s = delayed(sum)(d_histograms)
    return s


@delayed
def blocked_fill_nd(*args, hist, weight=None):
    hist_for_block = bh.Histogram(*hist.axes, storage=hist._storage_type())
    hist_for_block.fill(*args, weight=weight)
    return hist_for_block


def fill_nd(*args, hist, weight=None):
    ## total number of dimensions
    D = len(args)
    ## each entry is data along a specific dimension
    stack_of_delayeds = [a.to_delayed() for a in args]
    ## we assume all dimensions are chunked identically
    npartitions = len(stack_of_delayeds[0])
    ## we need to create a data structure that will connect coordinate
    ## chunks we loop over the number of partitions and connect the
    ## ith chunk along each dimension (loop over j is the loop over
    ## the dimensions).
    partitioned_fused_coordinates = [  # <-- list of tuples
        tuple(stack_of_delayeds[j][i] for j in range(D)) for i in range(npartitions)
    ]

    if weight is None:
        d_histograms = [
            blocked_fill_nd(*d, hist=hist) for d in partitioned_fused_coordinates
        ]
    else:
        d_weight = weight.to_delayed()
        d_histograms = [
            blocked_fill_nd(*d, hist=hist, weight=w)
            for d, w in zip(partitioned_fused_coordinates, d_weight)
        ]
    s = delayed(sum)(d_histograms)
    return s
