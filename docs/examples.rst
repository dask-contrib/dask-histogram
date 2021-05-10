Examples
--------

You're encouraged to check out the documentation for boost-histogram_;
any example you see there should work in dask-histogram if the input
data is a Dask collection.

Object Example
^^^^^^^^^^^^^^

In this example we will fill a 2D histogram with Gaussian data in both
dimensions (notice that, for convenience, the ``boost_histogram.axis``
and ``boost_histogram.storage`` namespaces are brought in as
``dh.axis`` and ``dh.storage``):

.. code-block:: python

   >>> import dask_histogram as dh
   >>> import dask.array as da
   >>> x = da.random.standard_normal(size=(100_000_000, 2), chunks=(10_000_000, 2))
   >>> h = dh.Histogram(
   ...     dh.axis.Regular(10, -3, 3),
   ...     dh.axis.Regular(10, -3, 3),
   ...     storage=dh.storage.Double(),
   ... )
   >>> h.fill(x)  # <-- no computation occurs
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
   >>> h.fill(x)  # fill again; notice the repr tells us we have staged fills.
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

NumPy-like Examples
^^^^^^^^^^^^^^^^^^^

We can create the same histogram we created above via the function API
which mirrors the functions in the ``dask.array`` module. First, we
explictly ask for a :py:obj:`Histogram <dask_histogram.Histogram>`
object by using the `histogram` argument. The computation is still
lazy, notice the `(has staged fills)` line in the repr.

.. code-block:: python

   >>> h = dh.histogramdd(x, bins=(10, 10), range=((-3, 3), (-3, 3), histogram=dh.Histogram)
   >>> h
   Histogram(
     Regular(10, -3, 3),
     Regular(10, -3, 3),
     storage=Double()) # (has staged fills)

If the `histogram` argument is left as the default value we get the
return style of the ``dask.array`` module (which itself is supporting
a NumPy like API), but we're using the ``Histogram`` object in the
background; again, the computation is still lazy:

.. code-block:: python

   >>> h, edges = dh.histogramdd(x, bins=(10, 10), range=((-3, 3), (-3, 3))
   >>> type(h)
   <class 'dask.array.core.Array'>
   >>> len(edges)
   2
   >>> type(edges[0])
   <class 'dask.array.core.Array'>
   >>> h.compute() # <-- this is calling dask.array.Array.compute()
   <result will be a NumPy array>

.. _boost-histogram: https://boost-histogram.readthedocs.io/en/latest/
.. _Dask: https://docs.dask.org/en/latest/

.. note:: More examples are shown in the API Reference for each API entry point.
