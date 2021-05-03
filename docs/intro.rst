Introduction
------------

Installation
^^^^^^^^^^^^

Install dask-histogram with pip:

.. code-block::

   pip install dask-histogram

The only dependencies are Dask_ and boost-histogram_.

Overview
^^^^^^^^

Dask-histogram aims to reproduce the API provided by boost-histogram_,
but with support for :doc:`dask collections <dask:user-interfaces>` as
input data. The core component is the
:class:`dask_histogram.Histogram` class. Additional components include
the NumPy-like :func:`dask_histogram.histogram`,
:func:`dask_histogram.histogram2d`, and
:func:`dask_histogram.histogramdd` functions.

The :class:`dask_histogram.Histogram` class inherits from
:class:`boost_histogram.Histogram`, overriding the ``fill`` function
such that it is aware of chunked Dask collections. The NumPy-like API
providing histogramming functions exists to mirror what is provided by
the :py:mod:`boost_histogram.numpy`.

.. _boost-histogram: https://boost-histogram.readthedocs.io/en/latest/
.. _Dask: https://docs.dask.org/en/latest/
