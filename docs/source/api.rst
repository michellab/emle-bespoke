API Reference
============

This section provides a comprehensive reference for the emle-bespoke package.

Code Structure
-------------

The emle-bespoke package is organized into the following modules:

.. graphviz::
   :align: center

   digraph package_structure {
      rankdir=LR;
      node [shape=box];
      
      "emle_bespoke" -> "train";
      "emle_bespoke" -> "models";
      "emle_bespoke" -> "utils";
      
      "train" -> "_loss";
      
      "models" -> "_emle";
      
      "utils" -> "_constants";
   }

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/emle_bespoke
   api/emle_bespoke.train
   api/emle_bespoke.models
   api/emle_bespoke.utils 