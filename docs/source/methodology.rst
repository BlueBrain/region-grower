.. |name| replace:: ``region-grower``

.. _NeuroTS: https://NeuroTS.readthedocs.io
.. _diameter-synthesis: https://diameter-synthesis.readthedocs.io

Methodology
===========

The methodology used to synthesize the cells in a given spatial context (an Atlas) is divideed into the following steps:

1. get the positions in the context from the MVD3 where the new cells must be synthesized.
2. get the following local properties of the cells at each of these positions: orientation, cell depth and layer depths.
3. get the TMD parameters and TMD distributions that will be used for synthesis (see the details of these parameters and distributions in the `NeuroTS`_ and `diameter-synthesis`_ documentations and in :ref:`Parameters`).
4. compute the cortical depth, which is a global properties of the context, from the TMD distributions.
5. load morphologies used for axon grafting (only if requested).
6. compute the target and hard limits for each cell.
7. call `NeuroTS`_ and `diameter-synthesis`_ with these data to synthesize each cell.
8. graft the given axons (only if requested).
9. save the morphology files (can be 'swc', 'asc' or 'h5' file).
10. save the new MVD3 file containing the position and orientation of each cell.
11. save the apical points and apical sections as YAML files.


Spatial context: the Atlas
--------------------------

The spatial context is given by an Atlas which is a set of volumetric datasets.

[PH]y
~~~~~

Position along brain region principal axis (for cortical regions that is the direction towards pia).


[PH]<layer>
~~~~~~~~~~~

For each `layer`, the corresponding volumetric dataset stores two numbers per voxel: lower and upper layer boundary along brain region principal axis.
Effectively, this allows to bind atlas-agnostic placement rules to a particular atlas space.

For instance, if we use `L1` to `L6` layer names in the placement rules, the atlas should have the following datasets ``[PH]y``, ``[PH]L1``, ``[PH]L2``, ``[PH]L3``, ``[PH]L4``, ``[PH]L5``, ``[PH]L6``.

``[PH]`` prefix stands for "placement hints" which is a historical way to address the approach used in |name|.


Orientation
~~~~~~~~~~~

For each voxcel, this dataset gives the local "principal direction" :math:`Y` (for instance, for cortical regions it is the direction towards pia).


Lookups
~~~~~~~

The local properties are looked up from the atlas.
For each cell in MVD3 we obtain its position :math:`y` along its "principal direction" :math:`Y` as well as the orientation and all layer boundaries along :math:`Y`.
This gives us the cell position `profiles`.


Scaling computation
-------------------

From the given Atlas we compute three kinds of scaling factors and limits:

* the target extent (only used for apicals): it is used inside `NeuroTS`_ to rescale the barcodes in order to obtain a size close to the one desired.
* the target thickness (always used for basals and used for apical if the fit is not given for the target extent): it is also used inside `NeuroTS`_ as a rescale factor for the barcodes but it is less accurate than the target extent because it is only base on the cortical depth.
* the hard limits: they are used to rescale the results of `NeuroTS`_ if it is needed.

Target extent
~~~~~~~~~~~~~

The given target extents should be computed as a linear fit (slope and intercept values) of the :math:`Y` extent as a function of path length. This is due to how `NeuroTS`_ works because it is not aware of the :math:`Y` extent of the synthesized cell, it is only aware of its path length.
These slope and intercept values are thus used to compute the path length required for `NeuroTS`_ to synthesize a morphology with a :math:`Y` extent close to the one desired. This factor is finally used inside `NeuroTS`_ to rescale the barcodes.

Note that this scaling factor can only be used with apicals.

In order to use this feature, the parameters should contain the following entries:

.. code-block:: python

    {
        "<mtype>": {
            "context_constraints": {
                "apical": {
                    "extent_to_target": {
                        "slope": 0.5,
                        "intercept": 1,
                        "layer": 1,
                        "fraction": 0.5
                    }
                }
            }
        }
    }

Where the ``"layer"`` and ``"fraction"`` entries stand for the target depth of the highest point of the morphology, and ``"slope"`` and ``"intercept"`` stand for the linear fit properties.

Target thickness
~~~~~~~~~~~~~~~~

The target thickness is a simple scaling computed from the ratio of the cortical thickness over of the current layer thickness (where the soma of the current cell is located).
This factor is also used inside `NeuroTS`_ to rescale the barcodes.

This feature is mandatory, thus the distributions should always contain the following entry:

.. code-block:: python

    {
        "metadata": {
            "cortical_thickness": [
                100,
                100,
                200,
                100,
                100,
                200
            ]
        }
    }

Hard limits
~~~~~~~~~~~

The previous target scaling factors do not ensure the actual size of the synthesized morphology.
This can lead to some issues like morphologies going slightly further to L1 for example.
In order to fix this issue, hard limits are added to resize the neurites so they can accurately fit to the given target.

In order to use this feature, the parameters should contain the following entries:

.. code-block:: python

    {
        "<mtype>": {
            "context_constraints": {
                "neurite type": {
                    "hard_limit_max": {
                        "layer": 1,
                        "fraction": 0.5
                    },
                    "hard_limit_min": {
                        "layer": 1,
                        "fraction": 0.5
                    }
                }
            }
        }
    }

Where ``"hard_limit_min"`` stand for the lower limit and ``"hard_limit_max"`` stand for the upper limit.
A fraction equal to 0 points to the bottom of the given layer and 1 points to its top.


Usage
=====

|name| is distributed via BBP Spack packages, and is available at BBP systems as |name| module.

.. code-block:: console

    module load region-grower

To pin module version, please consider using some specific `BBP archive S/W release <https://bbpteam.epfl.ch/project/spaces/display/BBPHPC/BBP+ARCHIVE+SOFTWARE+MODULES#BBPARCHIVESOFTWAREMODULES-TousetheSpackarchivemodules>`_.

This module brings one command:

.. code-block:: console

    region-grower --help

.. tip::

    Under the hood |name| is a Python package.

    Those willing to experiment with development versions can thus install it from BBP devpi server:

    .. code-block:: console

        $ pip install -i https://bbpteam.epfl.ch/repository/devpi/simple/ region-grower[mpi]

    Please note though that it requires ``mpi4py`` which can be non-trivial to install.
