Examples
--------

Simple Object Example
^^^^^^^^^^^^^^^^^^^^^

Filling a histogram with two dimensional Gaussian data (see the
`boost-histogram`_ docs for an introduction to the API).

.. code-block:: python

   >>> import dask_histogram as dh
   >>> import dask.array as da
   >>> x = da.random.standard_normal(size=(100_000_000, 2), chunks=(10_000_000, 2))
   >>> h = dh.Histogram(
   ...     dh.axis.Regular(10, -3, 3),
   ...     dh.axis.Regular(10, -3, 3),
   ...     storage=dh.storage.Double(),
   ... )
   >>> h.fill(*x.T)  # <-- no computation occurs
   Histogram(
     Regular(50, -3, 3),
     Regular(50, -3, 3),
     storage=Double()) # (has staged fills)
   >>> h.empty()
   True
   >>> h.compute()   # <-- trigger computation
   Histogram(
     Regular(50, -3, 3),
     Regular(50, -3, 3),
     storage=Double()) # Sum: 99459483.0 (100000000.0 with flow)
   >>> h.fill(*x.T)  # fill again; notice the repr tells us we have staged fills.
   Histogram(
     Regular(50, -3, 3),
     Regular(50, -3, 3),
     storage=Double()) # Sum: 99459483.0 (100000000.0 with flow) (has staged fills)
   >>> import dask
   >>> dask.compute(h.to_delayed())  # <-- convert to delayed and use dask.compute
   (Histogram(
     Regular(50, -3, 3),
     Regular(50, -3, 3),
     storage=Double()) # Sum: 198918966.0 (200000000.0 with flow),)

.. _boost-histogram: https://boost-histogram.readthedocs.io/en/latest/
.. _Dask: https://docs.dask.org/en/latest/
