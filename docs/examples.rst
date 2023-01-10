Examples
--------

Using the dask_histogram.factory function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:func:`dask_histogram.factory` function is the core piece of
the dask-histogram API; all other parts of the public API use it.

The function takes in two core inputs: the Dask data to be
histogrammed and the information that defines the histogram's
structure. The Dask data can be in Array, Series, or DataFrame form.
The histogram structure can be defined using the `axes` and
(optionally) `storage` arguments, or the `histref` argument can be
used.

Histogramming one dimensional data:

.. code-block:: python

   >>> import boost_histogram as bh
   >>> import dask.array as da
   >>> import dask_histogram as dh
   >>> x = da.random.uniform(size=(1000,), chunks=(250,))
   >>> h = dh.factory(x, axes=(bh.axis.Regular(10, 0, 1),))
   >>> h
   dask_histogram.AggHistogram<hist-aggregate, ndim=1, storage=Double()>
   >>> h.compute()
   Histogram(Regular(10, 0, 1), storage=Double()) # Sum: 1000.0

Using weights and a reference histogram:

.. code-block:: python

   >>> w = da.random.uniform(size=(1000,), chunks=(250,))
   >>> ref = bh.Histogram(bh.axis.Regular(10, 0, 1), storage=bh.storage.Weight())
   >>> h = dh.factory(x, weights=w, histref=ref)
   >>> h
   dask_histogram.AggHistogram<hist-aggregate, ndim=1, storage=Weight()>

dask.array/NumPy-like Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can create histograms via the API which mirrors the functions in
the ``dask.array`` module (of course, ``dask.array`` mirrors the
``numpy`` API).

First, we explictly ask for an :py:obj:`AggHistogram
<dask_histogram.AggHistogram>` object by using the `histogram`
argument.

.. code-block:: python

   >>> import dask.array as da
   >>> import dask_histogram as dh
   >>> x = da.random.standard_normal(size=(10000, 2), chunks=(2000, 2))
   >>> h = dh.histogramdd(x, bins=(10, 10), range=((-3, 3), (-3, 3)), histogram=True)
   >>> h
   dask_histogram.AggHistogram<hist-aggregate, ndim=2, storage=Double()>

If the `histogram` argument is left as the default value (``None``) we
get the return style of the ``dask.array`` module (which itself is
supporting a NumPy like API), but we're using the ``AggHistogram``
object in the background; again, the computation is still lazy:

.. code-block:: python

   >>> h, edges = dh.histogramdd(x, bins=(10, 10), range=((-3, 3), (-3, 3)))
   >>> type(h)
   <class 'dask.array.core.Array'>
   >>> len(edges)
   2
   >>> type(edges[0])
   <class 'dask.array.core.Array'>
   >>> h.compute() # doctest:+SKIP
   <result will be a NumPy array>

Let's consider a DataFrame called ``df`` with four columns: `a`, `b`,
`c`, and `w`:

.. code-block:: python

   >>> df  # doctest:+SKIP
   Dask DataFrame Structure:
                        a        b        c        w
   npartitions=5
   0              float64  float64  float64  float64
   200                ...      ...      ...      ...
   ...                ...      ...      ...      ...
   800                ...      ...      ...      ...
   999                ...      ...      ...      ...
   Dask Name: from_pandas, 5 tasks

First let's consider a one dimensional histogram of `a` with weights `w`:

.. code-block:: python

   >>> h, edges = dh.histogram(df["a"], bins=12, range=(-3, 3), weights=df["w"]) # doctest:+SKIP
   >>> h  # doctest:+SKIP
   dask.array<from-value, shape=(12,), dtype=float64, chunksize=(12,), chunktype=numpy.ndarray>
   >>> edges # doctest:+SKIP
   dask.array<array, shape=(13,), dtype=float64, chunksize=(13,), chunktype=numpy.ndarray>

Note that the same histogram can be created with
:py:func:`dask_histogram.factory` like so:

.. code-block:: python

   >>> h = dh.factory(df["a"], axes=(bh.axis.Regular(12, -3, 3),), weights=df["w"]) # doctest:+SKIP
   >>> h # doctest:+SKIP
   dask_histogram.AggHistogram<hist-aggregate, ndim=1, storage=Double()>

We can also grab multiple columns to histogram and return a
:py:obj:`Histogram <dask_histogram.AggHistogram>` object:

.. code-block:: python

   >>> h = dh.histogramdd(  # doctest:+SKIP
   ...     df[["a", "b", "c"]],
   ...     bins=(6, 7, 8),
   ...     range=((-3, 3),) * 3,
   ...     histogram=True,
   ... )
   >>> h # doctest: +SKIP
   dask_histogram.AggHistogram<hist-aggregate, ndim=3, storage=Double()>

With weights and variable width bins:

   >>> h = dh.histogramdd(  # doctest:+SKIP
   ...     df[["a", "c"]],
   ...     bins=[
   ...         [-3, -2, 0, 1, 2, 3],
   ...         [-2, -1, 1, 2],
   ...     ],
   ...     weights=df["w"],
   ...     storage=dh.storage.Weight(),
   ...     histogram=True,
   ... )
   >>> h # doctest:+SKIP
   dask_histogram.AggHistogram<hist-aggregate, ndim=2, storage=Weight()>

boost-histogram Inheriting Example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You're encouraged to check out the documentation for boost-histogram_;
any example you see there should work in dask-histogram if the input
data is a Dask collection.

In this example we will fill a 2D histogram with Gaussian data in both
dimensions (notice that, for convenience, the ``boost_histogram.axis``
and ``boost_histogram.storage`` namespaces are brought in as
``dh.axis`` and ``dh.storage``):

.. code-block:: python

   >>> import dask_histogram.boost as dhb
   >>> import dask.array as da
   >>> x = da.random.standard_normal(size=(100_000_000, 2), chunks=(10_000_000, 2))
   >>> h = dhb.Histogram(
   ...     dh.axis.Regular(10, -3, 3),
   ...     dh.axis.Regular(10, -3, 3),
   ...     storage=dh.storage.Double(),
   ... )
   >>> h.fill(x)  # <-- no computation occurs
   Histogram(
     Regular(10, -3, 3),
     Regular(10, -3, 3),
     storage=Double()) # (has staged fills)
   >>> h.empty()
   True
   >>> h.compute() # doctest:+SKIP
   Histogram(
     Regular(50, -3, 3),
     Regular(50, -3, 3),
     storage=Double()) # Sum: 99459483.0 (100000000.0 with flow)
   >>> import dask
   >>> dask.compute(h.to_delayed())  # doctest:+SKIP
   (Histogram(
     Regular(50, -3, 3),
     Regular(50, -3, 3),
     storage=Double()) # Sum: 99459483.0 (100000000.0 with flow),)


.. note:: More examples are shown in the API Reference.


.. _boost-histogram: https://boost-histogram.readthedocs.io/en/latest/
.. _Dask: https://docs.dask.org/en/latest/
