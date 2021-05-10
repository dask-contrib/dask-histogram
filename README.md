# dask-histogram

> Scale up histogramming with [Dask](https://dask.org).

[![Tests](https://github.com/douglasdavis/dask-histogram/actions/workflows/ci.yml/badge.svg)](https://github.com/douglasdavis/dask-histogram/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/dask-histogram/badge/?version=latest)](https://dask-histogram.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/dask-histogram.svg?colorB=486b87&style=flat)](https://pypi.org/project/dask-histogram/)
[![Python Version](https://img.shields.io/pypi/pyversions/dask-histogram)](https://pypi.org/project/dask-histogram/)

The [boost-histogram](https://github.com/scikit-hep/boost-histogram)
library provides a performant object oriented API for histogramming in
Python. This library adds support for lazy calculations on Dask
collections, in the style of boost-histogram.

**The library is still in development but usable for testing.**
