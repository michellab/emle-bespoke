Welcome to EMLE-Bespoke's documentation!
=====================================

EMLE-Bespoke is a Python package for training and using bespoke EMLE (Embedded Machine Learning Environment) models.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api/index
   examples/index

Installation
-----------

To install EMLE-Bespoke, run:

.. code-block:: bash

   pip install emle-bespoke

Quick Start
----------

Here's a simple example of using EMLE-Bespoke:

.. code-block:: python

   from emle_bespoke import EMLETrainer

   # Initialize trainer
   trainer = EMLETrainer()

   # Train model
   trainer.train(
       z=atomic_numbers,
       xyz=coordinates,
       s=s_values,
       q_core=q_core_values,
       q_val=q_val_values,
       alpha=alpha_values
   )

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 