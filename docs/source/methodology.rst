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


Basic atlas information
-----------------------

The spatial context is provided by an Atlas which is a set of volumetric datasets. It must contains the following for synthesis to work correctly.

[PH]y
~~~~~

Position along brain region principal axis (for cortical regions that is the direction towards pia, otherwise, understood as the y direction).


[PH]<layer>
~~~~~~~~~~~

For each `layer`, the corresponding volumetric dataset stores two numbers per voxel: lower and upper layer boundary along brain region principal axis.
Effectively, this allows to bind atlas-agnostic placement rules to a particular atlas space.

For instance, if we use `L1` to `L6` layer names in the placement rules, the atlas should have the following datasets ``[PH]y``, ``[PH]L1``, ``[PH]L2``, ``[PH]L3``, ``[PH]L4``, ``[PH]L5``, ``[PH]L6``.

``[PH]`` prefix stands for "placement hints" which is a historical way to address the approach used in |name|.


Orientation
~~~~~~~~~~~

For each voxcel, this dataset gives the local "principal direction" :math:`Y` (for instance, for cortical regions it is the direction towards pia, or y direction).


Region structure
~~~~~~~~~~~~~~~~

In addition to a working atlas folder, one needs additional information to run synthesis in this atlas, which we encode in ``region_structure.yaml`` file.
An example of this file for an single column, considered are a region names ``O0`` is:

.. code-block:: yaml

  O0:
    layers:
    - 1
    - 2
    - 3
    - 4
    - 5
    - 6
    names:
        1: layer 1
        2: layer 2
        3: layer 3
        4: layer 4
        5: layer 5
        6: layer 6
    region_queries:
        1: '@.*1$'
        2: '@.*2[a|b]?$'
        3: '@.*3$'
        4: '@.*4$'
        5: '@.*5$'
        6: '@.*6[a|b]?$'
    thicknesses:
        1: 165
        2: 149
        3: 353
        4: 190
        5: 525
        6: 700

The entry ``layers`` contains the name of layers (not int values, but str in general) that corresponds to ``[PH][layers].nrrd``, ordered by depth, from top to bottom. The next entries are dictionaries where keys are the layers in ``layers`` entry.
The ``names`` entry contains human readable names that can be used for plotting, it is optional, mostly used for legend of collage plots.
The entry ``region_queries`` contains regexes for querying the atlas ``hierarchy.json`` to find ids or layers present in ``brain_region.nrrd``.
Finally, the entry ``thicknesses`` contains expected thicknesses of synthesis in vacuum which will be used to apply the rescaling algorithm. If the ``thicknesses`` entry is absent, no scaling rule ``extent_to_target`` will be applied, even if the rule is present, see below.


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

Advanced: Boundary and direction constraints
--------------------------------------------

We describe here two more advances usage of insitu synthesis, for controlling the spatial growth, from atlas cues, such as direction vector field, or additional meshes.

Direction constraints
~~~~~~~~~~~~~~~~~~~~~~

Under a region block of ``region_structure.yaml``, one can add a ``directions`` block to control the growing directions  of sections during synthesis via atlas orientation field.
Here is an example:

.. code-block:: yaml

  directions:
    - mtypes:
      - L1_HAC
      - L1_SAC
      neurite_types:
        - axon
      processes:
        - major
      params:
        direction: [0, 1, 0]
        power : 2.0
        mode: perpendicular
        layers: [1, 2]

This block contains a list of rules, with the following entries.  
* ``mtypes`` is the list of mtypes to apply this rule, 
* ``neurite_types`` is the list of neurite_types to apply this rule. 
* ``processes`` is optional and is the list of type of sections in NeuroTS (``major`` or ``secondary``) to differentiate between trunk (``major``) and obliques or collaterals (``secondary``).
* ``params`` is a dictionary to parametrize the rule. 
    * First, we specify the ``direction`` with a 3-vector, where ``[0, 1, 0]`` is the pia direction and ``[0, -1, 0]`` is opposite. For non-cortical regions, pia generalises to ``y`` coordinate of the orientation vector in ``orientation.nrrd``.
    * The ``mode`` selects between ``parallel`` (default if omitted) to follow the direction, and ``perpendicular`` to follow the perpendicular directions, hence a plane.
    * The optional ``power`` value is to set how strong the direction constraint is. The underlying algorithm converts the angle between the next point to grow and the direction into a probability function. If ``power=1`` (default) the relation is linear, otherwise it is a power of it (see ``get_directions`` in ``region-grower/region_grower/context.py``).
* Finally, this rule can be applied into only specific layers, via the list in ``layers`` entry (default to all layers).

Boundary constraints
~~~~~~~~~~~~~~~~~~~~

Under a region block of ``region_structure.yaml``, one can add a ``boundaries`` block to control the growing directions of trunks and sections during synthesis via atlas based meshes.
Here is an example:

.. code-block:: yaml

  boundaries:
    - mtypes:
      - L2_TPC:A
      neurite_types:
        - apical_dendrite
        - basal_dendrite
        - axon
      params_section:
        d_min: 5
        d_max: 50
      params_trunk:
        d_min: 5.0
        d_max: 1000
        power: 3.0
      mode: repulsive
      path: pia_mesh.obj

This block contains a list of rules for boundary constraints, similar to the direction for ``mtypes`` and ``neurite_types`` entries.
Each rule contains the following:
* a ``path`` entry to a mesh (readabe by https://github.com/mikedh/trimesh) in either voxel id or coordinates. If the path is relative, it will be interpreted as relative to the location of ``region_structure.yaml`` file. If the ``path`` is a folder, then it must contain mesh files which will be used for this rule. 
* ``mesh_type`` entry can be used with value ``voxel`` (default) for voxel ids or ``spatial`` for coordinates of the mesh.
* For a folder of meshes, the way the mesh are selected to act as boundary depends on the rule parametrized by ``multimesh_mode``, which can be set to 
    * ``closest`` (default) for selecting the closest (in euclidiean morm) mesh to the soma as the unique mesh, 
    * ``closest_y`` as closst along the y direction only,
    * ``inside`` to select the mesh surrounding the soma (used for barrel cortext for example),
    * ``territories``, specific for olfactory bulb glomeruli (see code for details, it assumes specific form of input data)
* There are two main modes for these rules, parametrized by ``modes``. 
    * ``repulsive`` (default) where the mesh will act as a repulsive wall/boundary, 
    * ``attractive`` where the mesh will attract the growing sections (more experimental, used for glomeruli spherical meshes for example).
* This rule can then be applied to either the section growing with ``params_section`` or trunk placements with ``params_trunk`` (only if the non-default trunk angle method is selected, see above), with following entries:
    * ``d_min``: distance under which probability of accept is 0
    * ``d_max``: distance over which probability of accepct is 1
    * ``power``: linearity of the probability as a function of distance (same as for direction entry).

The algorithm uses ray tracing to compute the distance to the mesh in the direction of the growth, and convert it to a probability function. The probability will be ``0`` below a distance of ``d_min``, and ``1`` above the distance of ``d_max``. This distance is from the previous point (soma for trunk), and the direction is to the next point (first neurite point for trunk). The ``power`` argument is as above, to have a nonlinear function of distance.
If ``d_min`` is close negative, there will be a probability of going though the mesh, hence making it leaky.
The mesh are considered as non-oriented, hence there is no notion of side, so is a branch passes through, it will have no effect, unless the growing turns back and hit the mesh again from the other side.
For more details of the probability functions for the various cases, we refer the reader to the corresponding part of the code, in ``context.py``.
See also examples in ``examples`` folder.
