.. |name| replace:: ``region-grower``

Methodology
===========

Candidate pool
--------------

For each cell in MVD3 we obtain its position :math:`y` along its "principal direction" :math:`Y` (for instance, for cortical regions it is the direction towards pia); as well as all layer boundaries along :math:`Y`.
This gives us cell position `profile`.
Please refer to :ref:`Atlas <ref-data-atlas>` section for the details where do these numbers come from.

To reduce computation, we coarsen these profiles, specifying their `resolution`.
Note there is trade-off between performance and precision; 10 um resolution works fine in practice.

Cells are then grouped by (`layer`, `mtype`, `etype`, `profile`), and joined to the morphology database using (`layer`, `mtype`, `etype`) as the join key.

Each (`morphology`, `profile`) pair from the resulting `candidate pool` is then given a score using the algorithm described in the detail in the next section.

Once each (`morphology`, `profile`) pair is scored, we group them *by profile*. Using scores as probability weights, we pick a morphology for every cell from the corresponding profile group (sampling with replacement). If no morphology gets a positive score at the given profile, all the corresponding cell positions are dropped.

The choice of morphologies can be tuned with :math:`\alpha` parameter, which specifies the exponential factor for each score. I.e., instead of using score :math:`S` as probability weights, one can use :math:`S^\alpha`. Using :math:`\alpha > 1` thus gives more preference to high scorers.


Calculating a placement score
-----------------------------

For each location we assign the morphology a score that reflects to what degree the applicable placement rules are fulfilled, if the morphology was placed at that location. This score is a real number from :math:`[0.0, 1.0]`, with :math:`0.0` indicating that a placement is impossible and :math:`1.0` indicating that all restrictions are fully met.

The set of rules applicable for each type is defined in :ref:`placement rules <ref-data-rules>` file.

Each morphology is :ref:`annotated <ref-data-annotations>` accordingly to pre-calculate Y-intervals for each region of interest (apical tuft, for instance).

Scores are first calculated for each separate rule and then combined to a total score.
If annotation corresponding to the rule is missing in morphology annotations, this rule is ignored when calculating the scores.

We distinguish two types of rules: *strict* ones and *optional* ones.
We aggregate scores for those differently, penalizing low *strict* score heavier than low *optional* score (see below).

In the following descriptions we will denote :math:`Y`-interval for a given morphology :math:`M` at a given position :math:`p` according to morphology annotation with :math:`(a^\uparrow, a^\downarrow)`; and :math:`Y`-interval prescribed by a placement rule with :math:`(r^\uparrow, r^\downarrow)`.

Strict rules
~~~~~~~~~~~~

As of now we have a single *strict* rule type named ``below``.
It prescribes that morphology should stay below certain Y-limit :math:`r^\uparrow`.
Thus we also address these rules as *hard limit* rules.

below
^^^^^

Despite the name "hard limit", we allow a small error margin: a base score of :math:`1.0` is reduced for each :math:`\mu` um exceeding the limit until reaching `0.0` for :math:`\mu=30` um.

.. math::

    L = \max\left(\min\left(\frac{r^\uparrow - a^\uparrow + 30}{30}, 1\right),0\right)

In placement rules file these rules are encoded with ``<rule type=below>`` elements:

.. code-block:: xml

    <rule id="L1_hard_limit" type="below" segment_type="dendrite" y_layer="1" y_fraction="1.0"/>

- ``y_layer``, ``y_fraction`` specify layer ID (string) and relative position in the layer (:math:`0.0` to :math:`1.0`) corresponding to the upper limit :math:`r^\uparrow`
- ``segment_type`` attribute is not used at the moment

Optional rules
~~~~~~~~~~~~~~

As of now we have two rules of these type: ``region_target`` and ``region_occupy``.

These are rules of the type where an interval in the layer structure (for example upper half of layer 5) has to be aligned with an (vertical) interval in the structure of the morphology (for example: the apical tuft). Thus we also address these rules as *interval overlap* rules.

region_target
^^^^^^^^^^^^^

Assuming :math:`(a^\uparrow, a^\downarrow)` is :math:`Y`-interval for a given morphology :math:`M` at a given position :math:`p` according to morphology annotation; and :math:`(r^\uparrow, r^\downarrow)` is :math:`Y`-interval prescribed by a placement rule, we calculate the overlap between the two:

.. math::

    I = \max{\left(\frac{\min\left(a^\uparrow, r^\uparrow\right) - \max\left(a^\downarrow, r^\downarrow\right)}{\min\left(a^\uparrow - a^\downarrow, r^\uparrow - r^\downarrow\right)}, 0\right)}

:math:`I` varies from :math:`0.0` (no overlap) to :math:`1.0` (max possible overlap, i.e. one of the intervals contains another).

In placement rules file these rules are encoded with ``<rule type=region_target>`` elements:

.. code-block:: xml

    <rule id="dendrite, Layer_1"  type="region_target" segment_type="dendrite" y_min_layer="1" y_min_fraction="0.00" y_max_layer="1" y_max_fraction="1.00" />

- ``y_min_layer``, ``y_min_fraction`` specify layer ID and relative position in the layer corresponding to the lower limit :math:`r^\downarrow`
- ``y_max_layer``, ``y_max_fraction`` specify layer ID and relative position in the layer corresponding to the upper limit :math:`r^\uparrow`
- ``segment_type`` attribute is not used at the moment


region_occupy
^^^^^^^^^^^^^

This rule is similar to ``region_target`` but instead of checking if one interval is *within* the other, we are striving for *exact* match.

.. math::

    I = \max{\left(\frac{\min\left(a^\uparrow, r^\uparrow\right) - \max\left(a^\downarrow, r^\downarrow\right)}{\max\left(a^\uparrow - a^\downarrow, r^\uparrow - r^\downarrow\right)}, 0\right)}

I.e., we achieve optimal score :math:`1.0` if and only if two intervals coincide.

In placement rules file these rules are encoded with ``<rule type=region_occupy>`` elements:

.. code-block:: xml

    <rule id="dendrite, Layer_1"  type="region_occupy" segment_type="dendrite" y_min_layer="1" y_min_fraction="0.00" y_max_layer="1" y_max_fraction="1.00" />

Rule attributes are analogous to those used with ``region_target`` rule.

Combining the scores
~~~~~~~~~~~~~~~~~~~~

We aggregate strict scores :math:`L_k` with :math:`\min` function:

.. math::

    \hat{L} = {\min\limits_{k} L_k}

If there are no strict scores, :math:`\hat{L} = 1`.

By contrast, we aggregate optional scores :math:`I_j` in a slightly more "relaxed" way, with a harmonic mean.
That allows us to penalize low score for a particular rule heavier than a simple mean, but still "give it a chance" if other interval scores are high:

.. math::

    \hat{I} = \left(\frac{\sum\limits_{j} I_j^{-1}}{n}\right)^{-1}

Please note that if some optional score is close to zero (<0.001); the aggregated optional score would be zero, same as with strict scores.

If there are no optional scores or if optional scores are ignored, :math:`\hat{I} = 1`.

The final score :math:`\hat{S}` is a product of aggregated strict and optional scores:

.. math::

    \hat{S} = \hat{I} \cdot \hat{L}


Usage
=====

|name| is distributed via BBP Spack packages, and is available at BBP systems as |name| module.

.. code-block::console

    $ module load region-grower

To pin module version, please consider using some specific `BBP archive S/W release <https://bbpteam.epfl.ch/project/spaces/display/BBPHPC/BBP+ARCHIVE+SOFTWARE+MODULES#BBPARCHIVESOFTWAREMODULES-TousetheSpackarchivemodules>`_.

This module brings several commands, some of them to be used for circuit building; and others as auxiliary tools for debugging placement algorithm itself.
We will briefly list them below.

.. tip::

    Under the hood |name| is a Python package.

    Those willing to experiment with development versions can thus install it from BBP devpi server:

    .. code-block:: console

        $ pip install -i https://bbpteam.epfl.ch/repository/devpi/simple/ region-grower[mpi]

    Please note though that it requires ``mpi4py`` which can be non-trivial to install.


synthesize-morphologies
-----------------------

Synthesize a set of morphologies.

Parameters
~~~~~~~~~~

    --mvd3                      Path to input MVD3 file [deprecated: use --cells-path instead]
    --cells-path                Path to a file storing cells collection [required]
    --tmd-parameters            Path to JSON with TMD parameters [required]
    --tmd-distributions         Path to JSON with TMD distributions [required]
    --morph-axon                TSV file with axon morphology list (for grafting)
    --base-morph-dir            Path to base morphology release folder
    --atlas                     Atlas URL [required]
    --atlas-cache               Atlas cache folder [optional]
    --seed                      Random number generator seed [optional, default:0]
    --out-mvd3                  Path to output MVD3 file [deprecated: use --out-cells-path instead]
    --out-cells-path            Path to output cells file [required]
    --out-apical                Path to output YAML apical file containing the coordinates where apical dendrites are tufting [required]
    --out-apical-NRN-sections   Path to output YAML apical file containing the neuron section ids where apical dendrites are tufting [required]
    --out-morph-dir             Path to output morphology folder [optional, default: out]
    --out-morph-ext             Morphology export format(s) [choices: ['swc', 'asc', 'h5'], default: ['swc']]
    --max-files-per-dir         Maximum files per level for morphology output folder [optional]
    --overwrite                 Overwrite output morphology folder [optional]
    --max-drop-ratio            Max drop ratio for any mtype [optional, default: 0]
    --scaling-jitter-std        Apply scaling jitter to all axon sections with the given std [optional]
    --rotational-jitter-std     Apply rotational jitter to all axon sections with the given std [optional]
    --no-mpi                    Do not use MPI and run everything on a single core [optional]


Input Data
==========

.. _ref-data-atlas:

Atlas
-----

`synthesize-morphologies` relies on a set of volumetric datasets being provided by the atlas.

[PH]y
~~~~~

Position along brain region principal axis (for cortical regions that is the direction towards pia).

[PH]<layer>
~~~~~~~~~~~

For each `layer` used in the placement rules (see below), the corresponding volumetric dataset stores two numbers per voxel: lower and upper layer boundary along brain region principal axis.
Effectively, this allows to bind atlas-agnostic placement rules to a particular atlas space.

For instance, if we use `L1` to `L6` layer names in the placement rules, the atlas should have the following datasets ``[PH]y``, ``[PH]L1``, ``[PH]L2``, ``[PH]L3``, ``[PH]L4``, ``[PH]L5``, ``[PH]L6``.

``[PH]`` prefix stands for "placement hints" which is a historical way to address the approach used in |name|.


.. _ref-data-rules:

Placement rules
---------------

XML file defining a set of rules.

Root element ``<placement_rules>`` (no attributes) contains a collection of ``<rule>`` elements encoding rules described above.
Each ``<rule>`` has required ``id``, ``type`` attributes, plus additional attributes depending on the rule type (please refer to the rules description above for the details).
Rules are grouped into *rule sets*: `global`, which are applied to all the morphologies; and `mtype`-specific, applied solely to morphologies of the corresponding mtype.

This XML file might also specify additional random rotation applied to all the cells or specific mtypes.

Global rules
~~~~~~~~~~~~

Defined in ``<global_rule_set>`` element (no attributes), which can appear only once in XML file.

Usually global rules are hard limit rules.

Rule IDs should be unique.

Mtype rules
~~~~~~~~~~~

Defined in ``<mtype_rule_set>`` elements, which can appear multiple times in XML file.
Each element should have ``mtype`` attribute with the associated mtype (or `|`-separated list of mtypes).
No mtype can appear in more than one ``<mtype_rule_set>``.

Usually mtype rules are interval overlap rules.

Rule IDs should be unique within mtype rule set, and should not overlap with global rule IDs.


Global rotation
~~~~~~~~~~~~~~~

.. warning::

  | This functionality is temporarily not available; random rotation around Y-axis is used indiscriminately for all cells.
  | Please contact NSE team if you need fine control over rotation angles.

Defined in ``<global_rotation>`` element (no attributes), which can appear no more than once in XML file.
It specifies rotation for *all* the cells, for which there are no mtype-specific rotation rules (see below).

Contains one or several ``<rotation>`` element(s), each one specifying rotation axis and random distribution to draw angles from (in radians). Please refer to `this page <https://bbpteam.epfl.ch/project/spaces/display/BBPNSE/Defining+distributions+in+config+files>`_ for instructions how to specify distribution.

.. code-block:: xml

    <!-- uniform random rotation around Y-axis -->
    <rotation axis="y" distr='["uniform", {"low": -3.14159, "high": 3.14159}]' />

Rotations are applied sequentially as they appear in XML file.


Mtype rotations
~~~~~~~~~~~~~~~

.. warning::

  | This functionality is temporarily not available; random rotation around Y-axis is used indiscriminately for all cells.
  | Please contact NSE team if you need fine control over rotation angles.

Defined in ``<mtype_rotation>`` elements, which can appear multiple times in XML file.
Each element should have ``mtype`` attribute with the associated mtype (or `|`-separated list of mtypes).
No mtype can appear in more than one ``<mtype_rotation>``.

The content of each element is analogous to ``<global_rotation>``.

Mtype-specific rotations *override* global ones (not combined with those).


Example
~~~~~~~

.. code-block:: xml

    <placement_rules>

      <global_rule_set>
        <rule id="L1_hard_limit" type="below" segment_type="dendrite" y_layer="1" y_fraction="1.0"/>
        <rule id="L1_axon_hard_limit" type="below" segment_type="axon" y_layer="1" y_fraction="1.0"/>
      </global_rule_set>

      <mtype_rule_set mtype="L5_TPC:A|L5_TPC:B">
        <rule id="dendrite, Layer_1"  type="region_target" segment_type="dendrite" y_min_layer="1" y_min_fraction="0.00" y_max_layer="1" y_max_fraction="1.00" />
        <rule id="axon, Layer_1" type="region_target" segment_type="axon" y_min_layer="1" y_min_fraction="0.00" y_max_layer="1" y_max_fraction="1.00" />
      </mtype_rule_set>

      <global_rotation>
        <!-- uniform random rotation around Y-axis -->
        <rotation axis="y" distr='["uniform", {"a": -3.14159, "b": 3.14159}]' />
      </global_rotation>

      <mtype_rotation mtype="L1_SAC">
        <!-- suppress random rotation -->
      </mtype_rotation>


    </placement_rules>

.. _ref-data-annotations:

Annotations
-----------

XML file which maps certain regions of the morphology (for instance, apical tuft) to corresponding placement rules.

Root element ``<annotations>`` (with single ``morphology`` attribute) contains a collection of ``<placement>`` elements.

Each ``<placement>`` element contains as attributes:

  * ``rule``: one of rule IDs defined by placement rules XML
  * ``y_min``, ``y_max``: :math:`Y`-range of morphology region, assuming morphology center is at :math:`y=0`

Example
~~~~~~~

.. code-block:: xml

    <annotations morphology="C030796A-P3">
      <placement rule="L1_hard_limit" y_max="1268.106" y_min="-323.641" />
      <placement rule="L1_axon_hard_limit" y_max="1186.089" y_min="-657.869" />
      <placement rule="dendrite, Layer_1" y_max="1270.0" y_min="1150.0" />
      <placement rule="axon, Layer_1" y_max="1230.0" y_min="1100.0" />
    </annotations>

For efficiency purpose, when collection of annotation files is used for ``choose-morphologies``, it is packed into a single JSON file with the following command delivered by |name| module:

.. code-block:: bash

    $ compact-annotations -o <OUTPUT> <ANNOTATION_DIR>

The result is a JSON file like:

::

  {
    "morph-1": {
      "L1_hard_limit": {
        "y_max": "96.4037744144",
        "y_min": "-224.580195025"
      },
    },
    "morph-2": {
      "L1_hard_limit": {
        "y_max": "350.432",
        "y_min": "-183.648"
      },
      "L4_UPC, dendrite, Layer_2 - Layer_1": {
        "y_max": "350.292",
        "y_min": "228.707"
      },
    },
    ...
  }

To choose only a subset of morphologies from a given annotation folder, one can provide an optional ``--morphdb`` argument with path to MorphDB file:

.. code-block:: bash

    $ compact-annotations --morphdb <MORPHDB> -o <OUTPUT> <ANNOTATION_DIR>
