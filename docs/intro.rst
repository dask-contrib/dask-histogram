Getting Started
---------------

Installation
^^^^^^^^^^^^

The only dependencies are Dask_ and boost-histogram_.

Install dask-histogram with pip:

.. code-block::

   pip install dask-histogram

Or with conda-forge_:

.. code-block::

   conda install dask-histogram -c conda-forge

Overview
^^^^^^^^

Dask-histogram aims to reproduce the API provided by boost-histogram_,
but with support for :doc:`dask collections <dask:user-interfaces>` as
input data. The documentation assumes that you have some familiarity
with both Dask and boost-histogram.

The core component is the :class:`dask_histogram.Histogram` class,
which inherits from :class:`boost_histogram.Histogram` and overrides
the ``fill`` function such that it is aware of chunked/partitioned
Dask collections.

Additional components include the NumPy-like
:func:`dask_histogram.histogram`, :func:`dask_histogram.histogram2d`,
and :func:`dask_histogram.histogramdd` functions. These functions
exist to mirror what is provided by the
:py:mod:`boost_histogram.numpy` module. The behavior of these routines
(by default) mirrors the :py:func:`dask.array.histogram`,
:py:func:`dask.array.histogram2d`, and
:py:func:`dask.array.histogramdd` functions, respectively, while
taking advantage of the :class:`Histogram <dask_histogram.Histogram>`
object in the internal implementation.

.. _boost-histogram: https://boost-histogram.readthedocs.io/en/latest/
.. _Dask: https://docs.dask.org/en/latest/
.. _conda-forge: https://conda-forge.org/
