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

This package provides two commands:

* region-grower
* synthesize-morphologies


The region-grower command
~~~~~~~~~~~~~~~~~~~~~~~~~

This command provides two tools to generate input parameters and input distributions:

.. code-block:: bash

	region-grower generate-distributions --help  => Create the TMD distribution file
	region-grower generate-parameters --help     => Generate the TMD parameter file


The synthesize-morphologies command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This command chooses morphologies with 'placement hints'.

.. code-block:: bash

	synthesize-morphologies --help

Acknowledgments
---------------

region-grower is a generalization of the approach originally proposed by `Michael Reimann <mailto:michael.reimann@epfl.ch>`_ and `Eilif Muller <mailto:eilif.mueller@epfl.ch>`_ for hexagonal mosaic circuits.


Reporting issues
----------------

region-grower is maintained by BlueBrain Cells team at the moment.

Should you face any issue with using it, please submit a ticket to our `issue tracker <https://bbpteam.epfl.ch/project/issues/browse/CELLS>`_; or drop us an `email <mailto: bbp-ou-cells@groupes.epfl.ch>`_.
