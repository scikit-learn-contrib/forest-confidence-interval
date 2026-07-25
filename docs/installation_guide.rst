.. _installation_guide:

Installation Guide
==================

To install `forestci` and its dependencies:

.. code-block:: bash

   python -m pip install forestci

If you wish to install from the source code (available `here <https://github.com/scikit-learn-contrib/forest-confidence-interval>`_),
change your working directory to the top-level directory and install the
package in editable mode. Development dependencies are declared in
``pyproject.toml``:

.. code-block:: bash

   python -m pip install -e ".[dev]"
