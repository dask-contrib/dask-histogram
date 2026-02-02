Contributing
------------

Development
^^^^^^^^^^^

The recommended way to setup development of a feature branch is an
editable install of a fork of the git repository. First make sure that
you are working in a ``conda`` environment or your favorite style of
virtual environment.

.. code-block::

   $ git clone git@github.com:<username>/dask-histogram.git
   $ cd dask-histogram
   $ git remote add upstream https://github.com/dask-contrib/dask-histogram.git
   $ git checkout -b my-feature upstream/main
   $ pip install -e .[dev]

The use of ``[dev]`` ensures that you install the dependencies
for testing and building the documentation. You can also use
``[test]`` or ``[docs]`` to install only the dependencies for running
tests or building documentation, respectively.

After running the tests on your new feature, push your branch to your
fork and create a pull request.

Testing, etc.
^^^^^^^^^^^^^

We use ``pytest`` for testing; after installing with the
``[dev]`` option you can run (from the top level of the
repository):

.. code-block:: bash

   $ python -m pytest

We use ``black`` for formatting:

.. code-block:: bash

   $ black src tests

And ``ruff`` for linting:

.. code-block:: bash

   $ python -m ruff .

Type hints are encouraged; we use ``mypy`` for static type checking:

.. code-block:: bash

   $ mypy

Documentation
^^^^^^^^^^^^^

We use Sphinx_ to build the documentation.

.. code-block:: bash

   $ cd docs
   $ make html
   # Open _build/html/index.html in a web browser

Install ``sphinx-autobuild`` to get a live updated instance of the
documentation, and run it from the project root directory.

.. code-block:: bash

   $ pip install sphinx-autobuild
   $ sphinx-autobuild docs docs/_build/html
   # Open a web browser at http://127.0.0.1:8000/


.. _Sphinx: https://www.sphinx-doc.org/en/master/
