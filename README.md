# emle-bespoke

[![Documentation Status](https://github.com/michellab/emle-bespoke/actions/workflows/docs.yml/badge.svg)](https://michellab.github.io/emle-bespoke/)

A package for training and patching bespoke EMLE models. Also includes routines for fitting and deriving EMLE-compatible Lennard-Jones parameters.

## Table of Contents

1. [Documentation](#documentation)
1. [Installation](#installation)
1. [Usage](#usage)
1. [Log Level](#log-level)

## Documentation

For detailed documentation, including API reference, tutorials, and examples, visit the [documentation website](https://michellab.github.io/emle-bespoke/).

## Installation

First, create a conda environment with all of the required dependencies:

```bash
conda env create -f environment.yaml
conda activate emle-bespoke
```

Finally, install `emle-bespoke` in interactive mode within the activated environment:

```bash
pip install -e .
```

## Usage

### Command-Line Interface (CLI)

To generate reference data and train a bespoke EMLE model for a specific solute (e.g., benzene), run the following command:

```bash
emle-bespoke --solute c1ccccc1
```

This command will automatically solvate benzene in a 1000-molecule cubic water box and create a simulation using the default settings.

To sample the reference configurations using a hybrid mechanically-embedded ML/MM simulation with the ANI-2x model, run:

```bash
emle-bespoke --solute c1ccccc1 --ml_model ani2x
```

For more information on usage and available options, run:

```bash
emle-bespoke --help
```

## Logging Settings

By default, emle-bespoke logs messages at the INFO level. This means you will see informative messages about the overall progress but not necessarily detailed debugging information. You can control the verbosity of the logging output by setting the `EMLE_BESPOKE_LOG_LEVEL` environment variable:

```bash
export EMLE_BESPOKE_LOG_LEVEL="DEBUG"
```

If you want to include log messages from packages other than emle-bespke, set the `EMLE_BESPOKE_FILTER_LOGGERS` variable to 0:

```bash
export EMLE_BESPOKE_FILTER_LOGGERS=0
```

By default, this variable is set to 1, meaning only log messages coming from emle-bespoke are displayed.
