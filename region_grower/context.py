"""Use spatial properties to grow a cell.

The objective of this module is to provide an interface between
synthesis tools (here TNS) and the circuit building pipeline.

TLDR: SpaceContext.synthesized() is being called by
the placement_algorithm package to synthesize circuit morphologies.
"""

import json
from collections import namedtuple
from copy import deepcopy

import attr
import morphio
from diameter_synthesis import build_diameters
from tns import NeuronGrower
import numpy as np

from voxcell.cell_collection import CellCollection
from voxcell.nexus.voxelbrain import Atlas

from region_grower import RegionGrowerError, modify
from region_grower.atlas_helper import AtlasHelper

SpacePos = namedtuple(
    "SpacePos", ["position", "depth", "orientation", "thickness_layers"]
)


@attr.s
class SynthesisResult:
    """
    The object returned by SpaceContext.synthesized()
    """

    #: The grown morphology
    neuron = attr.ib(type=morphio.mut.Morphology)  # pylint: disable=no-member

    #: The apical points
    apical_points = attr.ib(type=[])


class SpaceContext(object):
    """Loads spatial information and provides
    basic functionality to query spatial properties
    required for neuronal synthesis.
    """

    def __init__(
        self, atlas: Atlas, tmd_distributions_path, tmd_parameters_path, recenter=True
    ):
        """Initialization with an atlas (of a BBP circuit)"""
        self.recenter = recenter
        self.atlas = AtlasHelper(atlas)

        with open(tmd_distributions_path, "r") as f:
            self.tmd_distributions = json.load(f)

        with open(tmd_parameters_path, "r") as f:
            self.tmd_parameters = json.load(f)

        self.cortical_depths = np.cumsum(self.tmd_distributions["metadata"]["cortical_thickness"])

    def verify(self, mtypes):
        """Check that context has distributions / parameters for all given mtypes."""
        for mtype in mtypes:
            if mtype not in self.tmd_distributions["mtypes"]:
                raise RegionGrowerError("Missing distributions for mtype: '%s'" % mtype)
            if mtype not in self.tmd_parameters:
                raise RegionGrowerError("Missing parameters for mtype: '%s'" % mtype)

    def synthesize(self, position, mtype) -> SynthesisResult:
        """Synthesize a cell based on the position and mtype."""
        par = self._correct_position_orientation_scaling(
            self.tmd_parameters[mtype],
            position,
        )

        # Today we don't use the atlas during the synthesis (we just use it to
        # generate the parameters)so we can
        # grow the cell as if it was in [0, 0, 0]
        # But the day we use it during the actual growth, we will need to grow the cell at its
        # absolute position and translate to [0, 0, 0] after the growth
        if self.recenter:
            par["origin"] = [0, 0, 0]

        if self.tmd_parameters[mtype]["diameter_params"]["method"] == "external":

            def external_diametrizer(neuron, model, neurite_type):
                return build_diameters.build(
                    neuron,
                    model,
                    [neurite_type],
                    self.tmd_parameters[mtype]["diameter_params"],
                )

        else:
            external_diametrizer = None

        grower = NeuronGrower(
            input_parameters=par,
            input_distributions=self.tmd_distributions["mtypes"][mtype],
            external_diametrizer=external_diametrizer,
        )
        grower.grow()

        return SynthesisResult(grower.neuron, grower.apical_points or [])

    def _correct_position_orientation_scaling(
        self, params, position
    ):
        """Return a copy of the passed parameter with the correct orientation and
        recentered at [0,0,0]"""
        result = deepcopy(params)
        result["origin"] = position

        for neurite_type in params["grow_types"]:
            if isinstance(params[neurite_type]["orientation"], list):
                result[neurite_type]["orientation"] = [
                    self.atlas.lookup_orientation(position, orient)
                    for orient in params[neurite_type]["orientation"]
                ]

        target, reference = self.atlas.lookup_target_reference_depths(position,
                                                                      self.cortical_depths)
        return modify.input_scaling(result, reference, target)


class CellHelper(object):
    """Loads spatial information and provides
       basic functionality to query spatial properties
       required for neuronal synthesis. In addition to
       SpatialContext also loads the cell information to be used.
    """

    def __init__(self, cells_file):
        """
        """
        self.cells = CellCollection.load_mvd3(cells_file)

    def positions(self, mtype):
        """Return a generator of mtype cell positions"""
        return (self.cells.positions[gid] for gid in self._filter_by_mtype(mtype))

    def _filter_by_mtype(self, mtype):
        """Returns ids of cell with the given mtype"""
        return self.cells.properties.index[
            self.cells.properties.mtype.str.contains(mtype)
        ]
