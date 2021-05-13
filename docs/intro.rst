Getting Started
---------------

Installation
^^^^^^^^^^^^

The only dependencies are Dask_ and boost-histogram_.

Install dask-histogram with pip:

.. code-block::

   pip install dask-histogram

Or with ``conda`` via the conda-forge_ channel:

.. code-block::

   conda install dask-histogram -c conda-forge

We test dask-histogram on GNU/Linux, macOS, and Windows.

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

We say Dask collections instead of only Dask arrays because
dask-histogram supports :py:obj:`dask.dataframe.DataFrame` as input
data anywhere that columnar arrays are supported, and
:py:obj:`dask.dataframe.Series` anywhere that 1D arrays are supported.
When mixing collections care must be taken to ensure the partitioning
of :py:obj:`DataFrame <dask.dataframe.DataFrame>` and :py:obj:`Series
<dask.dataframe.Series>` inputs are compatible with the chunking of
:py:obj:`Array <dask.array.Array>` inputs.

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
