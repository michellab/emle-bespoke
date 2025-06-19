Installation Guide
==================

Requirements
-----------

EMLE-Bespoke requires:

* Python 3.8 or later
* PyTorch 2.0 or later
* NumPy
* OpenMM
* Loguru

Installation Methods
------------------

From PyPI
~~~~~~~~~

The easiest way to install EMLE-Bespoke is via pip:

.. code-block:: bash

   pip install emle-bespoke

From Source
~~~~~~~~~~

To install EMLE-Bespoke from source:

.. code-block:: bash

   git clone https://github.com/your-username/emle-bespoke.git
   cd emle-bespoke
   pip install -e .

Development Installation
~~~~~~~~~~~~~~~~~~~~~~

For development, install additional dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

This will install additional packages needed for development:

* pytest
* black
* isort
* flake8
* sphinx
* sphinx-rtd-theme

Verifying Installation
--------------------

To verify your installation:

.. code-block:: python

    import emle_bespoke

    print(emle_bespoke.__version__)
