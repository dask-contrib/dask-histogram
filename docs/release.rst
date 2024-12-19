Releasing
=========

Tagging a version
-----------------

We use calendar versioning (CalVer) with the format: ``YYYY.MM.X``
where ``X`` is incremented depending on how many releases have already
occurred in the same year and month. For example, if the most recent
release is the first release from March of 2023 it would be
``2023.3.0``, the next release (on any day in that month) would be
``2023.3.1``.

Check the latest tag with git (or just visit the GitHub repository
tags list):

.. code-block::

   $ git fetch --all --tags
   $ git describe --tags $(git rev-list --tags --max-count=1)
   2023.3.0

Create a new tag that follows our CalVer convention (using
``2023.3.0`` example above, we write the next tag accordingly):


.. code-block::

   $ git tag -a -m "2023.3.1" 2023.3.1

Push the tag to GitHub (assuming ``origin`` points to the
``dask-contrib/dask-histogram`` remote):

.. code-block::

   $ git push origin 2023.3.1

Making the release
------------------

After pushing the tag, GitHub actions will take care of uploading the
wheel and sdist to PyPI. Please go to the Releases page on the GitHub
repository and Draft a new release associated with the tag.
