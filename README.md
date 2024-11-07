# Region Grower

Synthesize neuronal morphologies in a given spatial context provided by an atlas.


## Introduction

This package provides a general tools to synthesize cells in a given spatial context (a brain Atlas)
with embarrassingly parallel computation (with distributed computing via dask-mpi).
This package is difficult to use on its own, as it requires several specific inputs from atlas and
synthesis. It is advised to use it via [synthesis-workflow](https://github.com/BlueBrain/synthesis-workflow),
which is a workflow generating most of the inputs of region-grower.


## Installation

This package should be installed using pip:

```bash
pip install region-grower
```


## Usage

This package provides one main command which provides several tools.
Two tools to generate input parameters and input distributions and another to synthesize the cells.

### Generate distributions

Generate the TMD distribution file:

```bash
region-grower generate-distributions --help
```

#### Input data

The command ``region-grower generate-distributions`` needs the following inputs:

* a folder containing the cells which are used to generate the parameters (see the ``input_folder`` and ``--ext`` parameters).
* a ``dat`` file (see the ``dat_file`` parameter) with 3 columns:
	* morphology names,
	* any integer value (this column is not used by ``region-grower``)
	* mtypes.
* optionally a ``JSON`` file containing the specific parameters used for diametrization (see the ``--diametrizer-config`` parameter).

#### Output data

The command ``region-grower generate-distributions`` will create the following outputs:

* a ``JSON`` file containing the formatted distributions (see the ``--parameter-filename`` parameter).

### Generate parameters

Generate the TMD parameter file:

```bash
region-grower generate-parameters --help
```

#### Input data

The command ``region-grower generate-parameters`` needs the following inputs:

* a folder containing the cells which are used to generate the parameters (see the ``input_folder`` and ``--ext`` parameters).
* a ``dat`` file (see the ``dat_file`` parameter) with 3 columns:
	* morphology names,
	* any integer value (this column is not used by ``region-grower``)
	* mtypes.
* optionally a ``JSON`` file containing the specific parameters used for diametrization (see the ``--diametrizer-config`` parameter).

#### Output data

The command ``region-grower generate-parameters`` will create the following outputs:

* a ``JSON`` file containing the formatted parameters (see the ``--parameter-filename`` parameter).

### Synthesize cells

Synthesize morphologies into an given atlas according to the given TMD parameters and distributions:

```bash
region-grower synthesize-morphologies --help
```

#### Input data

The command ``region-grower synthesize-morphologies`` needs the following inputs:

* a ``sonata`` file containing the positions of the cells that must be synthesized.
* a ``JSON`` file containing the parameters used to synthesize the cells (see the ``--tmd-parameters`` parameter). This file should follow the schema given in :ref:`Parameters`.
* a ``JSON`` file containing the distributions used to synthesize the cells (see the ``--tmd-distributions`` parameter). This file should follow the schema given in :ref:`Parameters`.
* a ``TSV`` file giving which morphology should be used for axon grafting and the optional scaling factor (see the ``--morph-axon`` parameter). The morphologies referenced in this file should exist in the directory given with the ``--base-morph-dir`` parameter.
* a directory containing an Atlas.

#### Output data

The command ``region-grower synthesize-morphologies`` will create the following outputs:

* a ``sonata`` file containing all the positions and orientations of the synthesized cells (see ``--out-cells`` parameter).
* a directory containing all the synthesized morphologies (see ``--out-morph-dir`` and ``--out-morph-ext`` parameters).
* a ``YAML`` file containing the apical point positions (see ``--out-apical`` parameter).
* a ``YAML`` file containing the Neuron IDs of the sections containing the apical points (see ``--out-apical-nrn-sections`` parameter).

## Funding & Acknowledgment

The development of this software was supported by funding to the Blue Brain Project,
a research center of the École polytechnique fédérale de Lausanne (EPFL),
from the Swiss government's ETH Board of the Swiss Federal Institutes of Technology.

Copyright (c) 2022-2024 Blue Brain Project/EPFL
