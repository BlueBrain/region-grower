"""Use spatial properties to grow a cell.

The objective of this module is to provide an interface between
synthesis tools (here TNS) and the circuit building pipeline.

TLDR: SpaceContext.synthesized() is being called by
the placement_algorithm package to synthesize circuit morphologies.
"""

import json
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Union

import attr
import morphio
import numpy as np
from diameter_synthesis import build_diameters
from morphio import SectionType
from neuroc.scale import ScaleParameters, scale_section
from tns import NeuronGrower
from tns.validator import validate_neuron_params, validate_neuron_distribs
from voxcell.cell_collection import CellCollection
from voxcell.nexus.voxelbrain import Atlas

from region_grower import RegionGrowerError, modify
from region_grower.atlas_helper import AtlasHelper

Point = Union[List[float], np.array]

TYPE_TO_STR = {
    SectionType.basal_dendrite: "basal",
    SectionType.apical_dendrite: "apical",
    SectionType.axon: "axon",
}


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
        self,
        atlas: Atlas,
        tmd_distributions_path: str,
        tmd_parameters_path: str,
        recenter: bool = True,
    ) -> None:
        """Initialization with an atlas (of a BBP circuit)"""
        self.recenter = recenter
        self.atlas = AtlasHelper(atlas)

        with open(tmd_distributions_path, "r") as f:
            self.tmd_distributions = json.load(f)

        with open(tmd_parameters_path, "r") as f:
            self.tmd_parameters = json.load(f)

        self.cortical_depths = np.cumsum(self.tmd_distributions["metadata"]["cortical_thickness"])

        self.position = None
        self.depths = None
        self.current_depth = None
        self.current_orientations = None

    def _set_current_position(self, position: float) -> None:
        '''Lookup atlas informations at current neuron position.'''
        self.position = position
        self.depths = self.atlas.get_layer_boundary_depths(position)
        self.current_depth = self.atlas.depths.lookup(position)
        self.current_orientations = self.atlas.orientations.lookup(position)

    def _distance_to_constraint(self, constraint: Dict) -> Optional[float]:
        """Returns the distance from the current position and the constraint

        Args:
            constraint: a dict containing a 'layer' key and a 'fraction' keys.

        """
        if not constraint:
            return None

        constraint_position = self._layer_fraction_to_position(
            constraint['layer'],
            constraint['fraction']
        )
        return self.current_depth - constraint_position

    def _lookup_orientation(self, vector: Optional[Point]) -> np.array:
        """Returns the looked-up orientation for the given position.

        If orientation is None, the direction is assumed towards the pia"""
        return np.dot(self.current_orientations, vector)[0] if vector else np.array([0, 1, 0])

    def _lookup_target_reference_depths(self) -> np.array:
        """Returns the target and the reference depth for a given neuron position.

        First item is the depth of the lower (the further away from the pia) boundary
        of the layer in which is located 'position'.

        Second one is the equivalent value for the same layer but in the cortical column.
        """
        for depth, cortical_depth in zip(self.depths[1:], self.cortical_depths):
            if self.current_depth <= depth:
                return depth, cortical_depth

        raise RegionGrowerError(f"Current depth ({self.current_depth}) for position "
                                f"({self.position}) is outside of circuit boundaries")

    def _layer_fraction_to_position(self, layer: int, layer_fraction: float) -> float:
        """Returns an absolute position from a layer and a fraction of the layer

        Args:
            layer: a layer
            layer_fraction: relative position within the layer (from 0 at
                the bottom of the layer to 1 at the top)

        Returns: target depth
        """
        layer_thickness = self.depths[layer] - self.depths[layer - 1]
        return self.depths[layer - 1] + (1.0 - layer_fraction) * layer_thickness

    def _correct_position_orientation_scaling(self, params_orig: Dict) -> Dict:
        """Return a copy of the parameter with the correct orientation and scaling."""
        params = deepcopy(params_orig)

        for neurite_type in params["grow_types"]:
            if isinstance(params[neurite_type]["orientation"], list):
                params[neurite_type]["orientation"] = [
                    self._lookup_orientation(orient).tolist()
                    for orient in params[neurite_type]["orientation"]
                ]

        target, reference = self._lookup_target_reference_depths()

        apical_target = params.get("context_constraints", {}).get("apical", {}).get(
            "extent_to_target")
        modify.input_scaling(params, reference, target,
                             apical_target_extent=self._distance_to_constraint(apical_target))

        return params

    def verify(self, mtypes: Sequence[str]) -> None:
        """Check that context has distributions / parameters for all given mtypes."""
        for mtype in mtypes:
            if mtype not in self.tmd_distributions["mtypes"]:
                raise RegionGrowerError("Missing distributions for mtype: '%s'" % mtype)
            if mtype not in self.tmd_parameters:
                raise RegionGrowerError("Missing parameters for mtype: '%s'" % mtype)

            validate_neuron_distribs(self.tmd_distributions["mtypes"][mtype])
            validate_neuron_params(self.tmd_parameters[mtype])

    def _post_growth_rescaling(self, neuron: morphio.Morphology, params: Dict) -> None:
        """Scale all neurites so that their extents are compatible with the min and
        max hard limits rules."""

        for root_section in neuron.root_sections:
            constraints = params.get("context_constraints", {}).get(
                TYPE_TO_STR[root_section.type], {})

            scale = modify.output_scaling(
                root_section,
                target_min_length=self._distance_to_constraint(constraints.get("min")),
                target_max_length=self._distance_to_constraint(constraints.get("max"))
            )

            scale_section(root_section, ScaleParameters(mean=scale), recursive=True)

    def synthesize(self, position: Point, mtype: str) -> SynthesisResult:
        """Synthesize a cell based on the position and mtype.

        The steps are the following:
        1) Modify the input params so that the cell growth is compatible with the layer
        thickness at the given position
        2) Perform the growth and the diametrization
        3) Rescale the neurites so that they are compatible with the hard limits (if
        the neurite goes after the max hard limit, it is downscaled. And vice-versa if it is
        smaller than the min hard limit)
        """
        self._set_current_position(position)
        params = self._correct_position_orientation_scaling(self.tmd_parameters[mtype])

        # Today we don't use the atlas during the synthesis (we just use it to
        # generate the parameters)so we can
        # grow the cell as if it was in [0, 0, 0]
        # But the day we use it during the actual growth, we will need to grow the cell at its
        # absolute position and translate to [0, 0, 0] after the growth
        if self.recenter:
            params["origin"] = [0, 0, 0]

        if self.tmd_parameters[mtype]["diameter_params"]["method"] == "external":
            def external_diametrizer(neuron, model, neurite_type):
                return build_diameters.build(
                    neuron, model, [neurite_type], self.tmd_parameters[mtype]["diameter_params"]
                )
        else:
            external_diametrizer = None

        grower = NeuronGrower(
            input_parameters=params,
            input_distributions=self.tmd_distributions["mtypes"][mtype],
            external_diametrizer=external_diametrizer,
        )
        grower.grow()

        self._post_growth_rescaling(grower.neuron, params)
        return SynthesisResult(grower.neuron, grower.apical_points or [])


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
