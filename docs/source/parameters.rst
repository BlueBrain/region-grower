.. _Parameters:

Parameters
==========

This page describes the format of the expected parameters used by region-grower.

.. note:: These schemas only contain the entries used in ``region-grower``, not the ones used by its dependencies.

Parameters
----------

The parameters are given as a JSON file with one entry per mtype (the key must be equal to the mtype name).
Each of these entries must follow the following schema:

.. jsonschema:: ../../region_grower/schemas/parameters.json
    :lift_definitions:
    :auto_reference:
    :auto_target:

Distributions
-------------

The distributions are given as a JSON file which must follow the following schema:

.. jsonschema:: ../../region_grower/schemas/distributions.json
    :lift_definitions:
    :auto_reference:
    :auto_target:
