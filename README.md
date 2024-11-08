# emle-bespoke

A package for training and patching EMLE models, with routines for fitting LJ parameters.

## Table of Contents

1. [Installation](#installation)
1. [Log Level](#log-level)

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