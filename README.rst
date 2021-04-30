Region Grower
=============


Introduction
------------

This package provides tools to synthesize cells in a given spatial context (a brain Atlas).
It is an implementation of the algorithm for picking morphologies for given cell positions,
which aims to match a set of constraints prescribed by *placement rules*.


Installation
------------

This package should be installed using pip:

.. code-block:: bash

    pip install region-grower


Usage
-----

This package provides one main command which provides several tools.
Two tools to generate input parameters and input distributions and another to synthesize the cells.

Generate distributions
~~~~~~~~~~~~~~~~~~~~~~

Generate the TMD distribution file:

.. code-block:: bash

	region-grower generate-distributions --help

Input data
^^^^^^^^^^

The command ``region-grower generate-distributions`` needs the following inputs:

* a folder containing the cells which are used to generate the parameters (see the ``input_folder`` and ``--ext`` parameters).
* a ``dat`` file (see the ``dat_file`` parameter) with 3 columns:
	* morphology names,
	* any integer value (this column is not used by ``region-grower``)
	* mtypes.
* optionally a ``JSON`` file containing the specific parameters used for diametrization (see the ``--diametrizer-config`` parameter).

Output data
^^^^^^^^^^^

The command ``region-grower generate-distributions`` will create the following outputs:

* a ``JSON`` file containing the formatted distributions (see the ``--parameter-filename`` parameter).

Generate parameters
~~~~~~~~~~~~~~~~~~~

Generate the TMD parameter file:

.. code-block:: bash

	region-grower generate-parameters --help

Input data
^^^^^^^^^^

The command ``region-grower generate-parameters`` needs the following inputs:

* a folder containing the cells which are used to generate the parameters (see the ``input_folder`` and ``--ext`` parameters).
* a ``dat`` file (see the ``dat_file`` parameter) with 3 columns:
	* morphology names,
	* any integer value (this column is not used by ``region-grower``)
	* mtypes.
* optionally a ``JSON`` file containing the specific parameters used for diametrization (see the ``--diametrizer-config`` parameter).

Output data
^^^^^^^^^^^

The command ``region-grower generate-parameters`` will create the following outputs:

* a ``JSON`` file containing the formatted parameters (see the ``--parameter-filename`` parameter).

Synthesize cells
~~~~~~~~~~~~~~~~

Synthesize morphologies into an given atlas according to the given TMD parameters and distributions:

.. code-block:: bash

	region-grower synthesize-morphologies --help

Input data
^^^^^^^^^^

The command ``region-grower synthesize-morphologies`` needs the following inputs:

* a ``MVD3`` file containing the positions of the cells that must be synthesized.
* a ``JSON`` file containing the parameters used to synthesize the cells (see the ``--tmd-parameters`` parameter). This file should follow the schema given in :ref:`Parameters`.
* a ``JSON`` file containing the distributions used to synthesize the cells (see the ``--tmd-distributions`` parameter). This file should follow the schema given in :ref:`Parameters`.
* a ``TSV`` file giving which morphology should be used for axon grafting and the optional scaling factor (see the ``--morph-axon`` parameter). The morphologies referenced in this file should exist in the directory given with the ``--base-morph-dir`` parameter.
* a directory containing an Atlas.

Output data
^^^^^^^^^^^

The command ``region-grower synthesize-morphologies`` will create the following outputs:

* a ``MVD3`` file containing all the positions and orientations of the synthesized cells (see ``--out-cells`` parameter).
* a directory containing all the synthesized morphologies (see ``--out-morph-dir`` and ``--out-morph-ext`` parameters).
* a ``YAML`` file containing the apical point positions (see ``--out-apical`` parameter).
* a ``YAML`` file containing the Neuron IDs of the sections containing the apical points (see ``--out-apical-nrn-sections`` parameter).


Acknowledgments
---------------

region-grower is a generalization of the approach originally proposed by `Michael Reimann <mailto:michael.reimann@epfl.ch>`_ and `Eilif Muller <mailto:eilif.mueller@epfl.ch>`_ for hexagonal mosaic circuits.


Reporting issues
----------------

``region-grower`` is maintained by BlueBrain Cells team at the moment.

Should you face any issue with using it, please submit a ticket to our `issue tracker <https://bbpteam.epfl.ch/project/issues/browse/CELLS>`_; or drop us an `email <mailto: bbp-ou-cells@groupes.epfl.ch>`_.
