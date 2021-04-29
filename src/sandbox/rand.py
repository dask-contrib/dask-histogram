import dask_histogram as dh
import dask_histogram.array as dha
import dask.array as da

x = da.random.standard_normal(size=(10_000,), chunks=2_000)
y = da.random.standard_normal(size=(10_000,), chunks=2_000)
w = da.random.uniform(start=0.1, stop=0.5, size=(10_000,), chunks=2_000)

h = dh.Histogram(
    dh.axis.Regular(10, -3, 3), dh.axis.Regular(10, -3, 3), storage=dh.storage.Weight()
)

h.fill(x, y)

h2 = dha.histogramdd((x, y), bins=(10, 10), range=((-3, 3), (-3, 3)), weights=w)
