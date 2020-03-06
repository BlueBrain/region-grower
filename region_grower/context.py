"""Use spatial properties to grow a cell.

The objective of this module is to provide an interface between
synthesis tools (here TNS) and the circuit building pipeline.

TLDR: SpaceContext.synthesized() is being called by
the placement_algorithm package to synthesize circuit morphologies.
"""

from collections import namedtuple
from copy import deepcopy
import json

import attr
import morphio
import numpy as np
from voxcell import OrientationField
from voxcell.cell_collection import CellCollection

from tns import NeuronGrower
from diameter_synthesis import build_diameters

from region_grower import modify, RegionGrowerError

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
        self, atlas, tmd_distributions_path, tmd_parameters_path, recenter=True
    ):
        """Initialization with an atlas (of a BBP circuit)"""
        self.brain_regions = atlas.load_data("brain_regions")
        self.depths = atlas.load_data("depth")
        self.orientations = atlas.load_data("orientation", cls=OrientationField)
        self.L1 = atlas.load_data("thickness:L1")
        self.L2 = atlas.load_data("thickness:L2")
        self.L3 = atlas.load_data("thickness:L3")
        self.L4 = atlas.load_data("thickness:L4")
        self.L5 = atlas.load_data("thickness:L5")
        self.L6 = atlas.load_data("thickness:L6")

        self.recenter = recenter

        with open(tmd_distributions_path, "r") as f:
            self.tmd_distributions = json.load(f)

        with open(tmd_parameters_path, "r") as f:
            self.tmd_parameters = json.load(f)

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
            self.tmd_distributions["metadata"]["cortical_thickness"],
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

    def _cumulative_thicknesses(self, position):
        """cumulative thicknesses starting at layer 1"""
        return np.cumsum(
            [
                self.L1.lookup(position),
                self.L2.lookup(position),
                self.L3.lookup(position),
                self.L4.lookup(position),
                self.L5.lookup(position),
                self.L6.lookup(position),
            ]
        )

    def _get_layer(self, position, cumulative_thickness):
        """Find out in which layer position is"""
        depth = self.depths.lookup(position)
        for indice, thickness in enumerate(cumulative_thickness):
            if thickness > depth:
                return indice

        return len(cumulative_thickness)

    def _correct_position_orientation_scaling(
        self, params, cortical_thickness, position
    ):
        """Return a copy of the passed parameter with the correct orientation and
        recentered at [0,0,0]"""
        result = deepcopy(params)
        result["origin"] = position

        for neurite_type in params["grow_types"]:
            if isinstance(params[neurite_type]["orientation"], list):
                result[neurite_type]["orientation"] = [
                    self._get_orientation(position, orient)
                    for orient in params[neurite_type]["orientation"]
                ]

        cumulative_thickness = self._cumulative_thicknesses(position)
        layer = self._get_layer(position, cumulative_thickness)
        target_thickness = cumulative_thickness[layer - 1]

        if target_thickness < 1e-8:
            raise RegionGrowerError(
                "Zero thickness in space. This will not generate cell with ID: "
            )

        reference_thickness = np.cumsum(cortical_thickness)[layer - 1]

        return modify.input_scaling(result, reference_thickness, target_thickness)

    def _get_orientation(self, position, vector=None):
        """Returns the orientation for the selected cell with the corresponding
           input position, as extracted from spatial properties.
        """
        if vector is None:
            vector = [0, 1, 0]  # assume direction towards the pia.

        return np.dot(self.orientations.lookup(position), vector)[0]


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
