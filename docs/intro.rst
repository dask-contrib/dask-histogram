Getting Started
---------------

Installation
^^^^^^^^^^^^

The only dependencies are Dask_ and boost-histogram_.

Install dask-histogram with pip_:

.. code-block::

   pip install dask-histogram

Or with conda_ via the conda-forge_ channel:

.. code-block::

   conda install dask-histogram -c conda-forge

We test dask-histogram on GNU/Linux, macOS, and Windows.

Overview
^^^^^^^^

Dask-histogram provides a new `collection type
<https://docs.dask.org/en/latest/custom-collections.html>`_ for lazily
constructing histogram objects. The API provided by boost-histogram_
is leveraged to calculate histograms on chunked/partitioned data from
the core Dask Array and DataFrame collections.

The main component is the :class:`dask_histogram.AggHistogram` class.
Users will typically create ``AggHistogram`` objects via the
:py:func:`dask_histogram.factory` function, or the
NumPy/dask.array-like functions in the
:py:mod:`dask_histogram.routines` module. Another histogram class
exists in the :py:mod:`dask_histogram.boost` module
(:py:obj:`dask_histogram.boost.Histogram`) which inherits from
:class:`boost_histogram.Histogram` and overrides the ``fill`` function
such that it is aware of chunked/partitioned Dask collections. This
class is backed by :py:obj:`dask_histogram.AggHistogram.`

.. _boost-histogram: https://boost-histogram.readthedocs.io/en/latest/
.. _Dask: https://docs.dask.org/en/latest/
.. _conda-forge: https://conda-forge.org/
.. _pip: https://pip.pypa.io/en/stable/
.. _conda: https://docs.conda.io/en/latest/
