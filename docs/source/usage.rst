Usage Guide
===========

This guide will walk you through the main features of EMLE-Bespoke.

Basic Usage
----------

Training a Model
~~~~~~~~~~~~~~

Here's how to train a basic EMLE model:

.. code-block:: python

    from emle_bespoke import EMLETrainer

    # Initialize trainer
    trainer = EMLETrainer()

    # Train the model
    trainer.train(
        z=atomic_numbers,
        xyz=coordinates,
        s=s_values,
        q_core=q_core_values,
        q_val=q_val_values,
        alpha=alpha_values,
        train_mask=None,
        model_filename="model.mat",
        sigma=0.001,
        ivm_thr=0.05,
        epochs=500
    )

Patching a Model
~~~~~~~~~~~~~~

To patch an existing model:

.. code-block:: python

    # Patch the model
    trainer.patch(
        opt_param_names=["a_QEq", "chi_ref"],
        e_static_target=e_static,
        e_ind_target=e_ind,
        atomic_numbers=z,
        charges_mm=charges,
        xyz_qm=xyz_qm,
        xyz_mm=xyz_mm,
        q_core=q_core,
        q_val=q_val,
        s=s,
        alpha=alpha
    )

Command Line Interface
-------------------

Training from Command Line
~~~~~~~~~~~~~~~~~~~~~~~

You can also train models using the command line interface:

.. code-block:: bash

    emle-bespoke-train \
        --reference-data data.pkl \
        --filename-prefix model \
        --sigma 0.001 \
        --epochs 500 \
        --device cuda

Advanced Features
---------------

Using Different Samplers
~~~~~~~~~~~~~~~~~~~~~

EMLE-Bespoke provides different samplers for various use cases:

.. code-block:: python

    from emle_bespoke.samplers import OpenMMSampler, MDSampler

    # Using OpenMM sampler
    sampler = OpenMMSampler(system, context, integrator)
    
    # Using MD sampler
    md_sampler = MDSampler(system, context, integrator)

Customizing Training
~~~~~~~~~~~~~~~~~

You can customize various aspects of training:

.. code-block:: python

    trainer.train(
        # ... basic parameters ...
        lr_qeq=0.05,
        lr_thole=0.05,
        lr_sqrtk=0.05,
        print_every=10,
        device="cuda",
        dtype="float64"
    )

Tips and Best Practices
--------------------

1. Always validate your input data before training
2. Use appropriate learning rates for your specific case
3. Monitor training progress using the print_every parameter
4. Save models regularly during training
5. Use GPU acceleration when available 