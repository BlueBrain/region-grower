"""Use spatial properties to grow a cell.

The objective of this module is to provide an interface between
synthesis tools (here NeutoTS) and the circuit building pipeline.

TLDR: SpaceContext.synthesized() is being called by
the placement_algorithm package to synthesize circuit morphologies.
"""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of region-grower.
# See https://github.com/BlueBrain/region-grower for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import os
from collections import defaultdict
from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")  # suppress NEURON warning
# This environment variable must be set before 'NEURON' is loaded in 'morph_tool.nrnhines'.
# Note that if 'NEURON' was imported before it will not consider this environment variable.

import attr  # noqa: E402 ; pylint: disable=C0413
import morphio  # noqa: E402 ; pylint: disable=C0413
import numpy as np  # noqa: E402 ; pylint: disable=C0413
from diameter_synthesis import build_diameters  # noqa: E402 ; pylint: disable=C0413
from morph_tool import nrnhines  # noqa: E402 ; pylint: disable=C0413
from morph_tool import transform as mt  # noqa: E402 ; pylint: disable=C0413
from morph_tool.graft import graft_axon  # noqa: E402 ; pylint: disable=C0413
from neuroc.scale import RotationParameters  # noqa: E402 ; pylint: disable=C0413
from neuroc.scale import ScaleParameters  # noqa: E402 ; pylint: disable=C0413
from neuroc.scale import rotational_jitter  # noqa: E402 ; pylint: disable=C0413
from neuroc.scale import scale_morphology  # noqa: E402 ; pylint: disable=C0413
from neuroc.scale import scale_section  # noqa: E402 ; pylint: disable=C0413
from neurots import NeuronGrower  # noqa: E402 ; pylint: disable=C0413
from neurots import NeuroTSError  # noqa: E402 ; pylint: disable=C0413
from neurots.utils import PIA_DIRECTION  # noqa: E402 ; pylint: disable=C0413
from voxcell.cell_collection import CellCollection  # noqa: E402 ; pylint: disable=C0413

from region_grower import RegionGrowerError  # noqa: E402 ; pylint: disable=C0413
from region_grower import SkipSynthesisError  # noqa: E402 ; pylint: disable=C0413
from region_grower import modify  # noqa: E402 ; pylint: disable=C0413
from region_grower.morph_io import MorphLoader  # noqa: E402 ; pylint: disable=C0413
from region_grower.morph_io import MorphWriter  # noqa: E402 ; pylint: disable=C0413
from region_grower.utils import random_rotation_y  # noqa: E402 ; pylint: disable=C0413

Point = Union[List[float], np.array]
Matrix = Union[List[List[float]], np.array]


@attr.s(auto_attribs=True)
class SynthesisResult:
    """The object returned by SpaceWorker.synthesize()."""

    #: The grown morphology
    neuron: morphio.mut.Morphology  # pylint: disable=no-member

    #: The apical sections
    apical_sections: list

    #: The apical points (coordinates where the apical tufts are starting)
    apical_points: list


@attr.s(auto_attribs=True)
class CellState:
    """The container class for the current cell state."""

    position: Point
    orientation: Matrix
    mtype: str
    depth: float

    def lookup_orientation(self, vector: Optional[Point] = None) -> np.array:
        """Returns the looked-up orientation for the given position.

        If orientation is None, the direction is assumed towards the pia.
        """
        return np.dot(self.orientation, vector)[0] if vector else PIA_DIRECTION


@attr.s(auto_attribs=True)
class SpaceContext:
    """The container class for the current space context state."""

    layer_depths: List
    cortical_depths: List

    def layer_fraction_to_position(self, layer: int, layer_fraction: float) -> float:
        """Returns an absolute position from a layer and a fraction of the layer.

        Args:
            layer: a layer
            layer_fraction: relative position within the layer (from 0 at
                the bottom of the layer to 1 at the top)

        Returns: target depth
        """
        layer_thickness = self.layer_depths[layer] - self.layer_depths[layer - 1]
        return self.layer_depths[layer - 1] + (1.0 - layer_fraction) * layer_thickness

    def lookup_target_reference_depths(self, depth: float) -> np.array:
        """Returns the target and the reference depth for a given neuron position.

        First item is the depth of the lower (the further away from the pia) boundary
        of the layer in which is located 'position'.

        Second one is the equivalent value for the same layer but in the cortical column.
        """
        if self.cortical_depths is None:
            return 1, 1
        for layer_depth, cortical_depth in zip(self.layer_depths[1:], self.cortical_depths):
            # we round to avoid being outside due to numerical precision
            if np.round(depth, 3) <= np.round(layer_depth, 3):
                return layer_depth, cortical_depth

        raise RegionGrowerError(f"Current depth ({depth}) is outside of circuit boundaries")

    def distance_to_constraint(self, depth: float, constraint: Dict) -> Optional[float]:
        """Returns the distance from a given depth to a given constraint.

        Args:
            depth: the given depth.
            constraint: a dict containing a 'layer' key and a 'fraction' keys.

        """
        if not constraint or not self.layer_depths or np.isnan(self.layer_depths).all():
            return None

        constraint_position = self.layer_fraction_to_position(
            constraint["layer"], constraint["fraction"]
        )
        return depth - constraint_position


@attr.s(auto_attribs=True)
class SynthesisParameters:
    """The container class for the current synthesis parameters."""

    tmd_distributions: dict
    tmd_parameters: dict
    axon_morph_name: str = None
    axon_morph_scale: Optional[float] = None
    rotational_jitter_std: float = None
    scaling_jitter_std: float = None
    recenter: bool = True
    seed: int = 0
    min_hard_scale: float = 0


@attr.s(auto_attribs=True)
class ComputationParameters:
    """The container class for the current computation parameters."""

    morph_writer: Optional[MorphWriter] = None
    morph_loader: Optional[MorphLoader] = None
    with_NRN_sections: bool = False
    retries: int = 1
    debug_data: bool = False


def _to_be_isolated(morphology_path, point):  # pragma: no cover
    """Internal function to isolate Neuron."""
    cell = nrnhines.get_NRN_cell(morphology_path)
    return nrnhines.point_to_section_end(cell.icell.apical, point)


class SpaceWorker:
    """Synthesize cells in a given spatial context."""

    def __init__(
        self,
        cell_state: CellState,
        space_context: SpaceContext,
        synthesis_parameters: SynthesisParameters,
        computation_parameters: ComputationParameters,
    ) -> None:
        """Initialization with all required information for synthesis."""
        self.cell = cell_state
        self.context = space_context
        self.params = synthesis_parameters
        self.internals = computation_parameters
        self.debug_infos = defaultdict(dict)

    def _correct_position_orientation_scaling(self, params_orig: Dict) -> Dict:
        """Return a copy of the parameter with the correct orientation and scaling."""
        params = deepcopy(params_orig)

        for neurite_type in params["grow_types"]:
            if isinstance(params[neurite_type]["orientation"], dict):
                params["pia_direction"] = self.cell.lookup_orientation(PIA_DIRECTION).tolist()
            if isinstance(params[neurite_type]["orientation"], list):
                params[neurite_type]["orientation"] = [
                    self.cell.lookup_orientation(orient).tolist()
                    for orient in params[neurite_type]["orientation"]
                ]

        target, reference = self.context.lookup_target_reference_depths(self.cell.depth)

        apical_target = (
            params.get("context_constraints", {}).get("apical_dendrite", {}).get("extent_to_target")
        )
        modify.input_scaling(
            params,
            reference,
            target,
            apical_target_extent=self.context.distance_to_constraint(
                self.cell.depth, apical_target
            ),
            debug_info=self.debug_infos["input_scaling"] if self.internals.debug_data else None,
        )

        basal_target = (
            params.get("context_constraints", {}).get("basal_dendrite", {}).get("extent_to_target")
        )
        if basal_target is not None:
            modify.input_scaling(
                params,
                reference,
                target,
                basal_target_extent=self.context.distance_to_constraint(
                    self.cell.depth, basal_target
                ),
                debug_info=self.debug_infos["input_scaling"] if self.internals.debug_data else None,
            )

        return params

    def _post_growth_rescaling(self, grower: NeuronGrower, params: Dict) -> None:
        """Scale all neurites so that their extents stay between the min and max hard limits."""
        for root_section in grower.neuron.root_sections:
            constraints = params.get("context_constraints", {}).get(root_section.type.name, {})

            target_min_length = self.context.distance_to_constraint(
                self.cell.depth, constraints.get("hard_limit_min")
            )
            target_max_length = self.context.distance_to_constraint(
                self.cell.depth, constraints.get("hard_limit_max")
            )

            scale = modify.output_scaling(
                root_section,
                self.cell.orientation.dot(PIA_DIRECTION)[0],
                target_min_length=target_min_length,
                target_max_length=target_max_length,
            )

            if scale > self.params.min_hard_scale:
                scale_section(root_section, ScaleParameters(mean=scale), recursive=True)
                is_deleted = False
            else:
                if root_section.type.name == "apical_dendrite":
                    raise RegionGrowerError(f"Apical is removed because rescale = {scale}")

                grower.neuron.delete_section(root_section, recursive=True)
                is_deleted = True

            if self.internals.debug_data and scale != 1.0:
                self.debug_infos["neurite_hard_limit_rescaling"].update(
                    {
                        root_section.id: {
                            "neurite_type": root_section.type.name,
                            "scale": scale,
                            "target_min_length": target_min_length,
                            "target_max_length": target_max_length,
                            "deleted": is_deleted,
                        }
                    }
                )

    def synthesize(self) -> SynthesisResult:
        """Synthesize a cell based on the position and mtype.

        The steps are the following:
        1) Modify the input params so that the cell growth is compatible with the layer
        thickness at the given position
        2) Perform the growth and the diametrization
        3) Rescale the neurites so that they are compatible with the hard limits (if
        the neurite goes after the max hard limit, it is downscaled. And vice-versa if it is
        smaller than the min hard limit)
        """
        rng = np.random.default_rng(self.params.seed)

        for _ in range(self.internals.retries):
            try:
                return self._synthesize_once(rng)
            except NeuroTSError:
                pass

        raise SkipSynthesisError(
            "Too many attempts at synthesizing cell with NeuroTS"
        )  # pragma: no cover

    def completion(self, synthesized):
        """Write the given morphology and compute section IDs for Neuron."""
        morph_name, ext_paths = self.internals.morph_writer(
            synthesized.neuron, seed=self.params.seed
        )

        apical_NRN_sections = None
        if self.internals.with_NRN_sections and not self.internals.morph_writer.skip_write:
            # Get the first .asc or .swc path so neuron can load it
            morph_path = next(filter(lambda x: x.suffix in [".asc", ".swc"], ext_paths))
            apical_NRN_sections = self._convert_apical_sections_to_NRN_sections(
                synthesized.apical_points, morph_path
            )

        return {
            "name": morph_name,
            "apical_points": synthesized.apical_points,
            "apical_sections": synthesized.apical_sections,
            "apical_NRN_sections": apical_NRN_sections,
        }

    def _synthesize_once(self, rng) -> SynthesisResult:
        """One try to synthesize the cell."""
        params = self._correct_position_orientation_scaling(self.params.tmd_parameters)

        # Today we don't use the atlas during the synthesis (we just use it to
        # generate the parameters) so we can
        # grow the cell as if it was in [0, 0, 0]
        # But the day we use it during the actual growth, we will need to grow the cell at its
        # absolute position and translate to [0, 0, 0] after the growth
        if self.params.recenter:
            params["origin"] = [0, 0, 0]

        if self.params.tmd_parameters["diameter_params"]["method"] == "external":
            external_diametrizer = build_diameters.build
        else:
            external_diametrizer = None

        if self.params.axon_morph_name is not None:
            axon_morph = self.internals.morph_loader.get(self.params.axon_morph_name)
        else:
            axon_morph = None

        grower = NeuronGrower(
            input_parameters=params,
            input_distributions=self.params.tmd_distributions,
            external_diametrizer=external_diametrizer,
            skip_preprocessing=True,
            context={"debug_data": self.debug_infos["input_scaling"]},
            rng_or_seed=rng,
        )
        grower.grow()

        self._post_growth_rescaling(grower, params)

        if axon_morph is not None:
            self.transform_morphology(
                axon_morph, self.cell.orientation, self.params.axon_morph_scale, rng=rng
            )
            graft_axon(grower.neuron, axon_morph, rng=rng)

        apical_points = None
        apical_points = self._convert_apical_sections_to_apical_points(
            grower.neuron, grower.apical_sections
        )

        return SynthesisResult(grower.neuron, grower.apical_sections or [], apical_points)

    def transform_morphology(self, morph, orientation, scale=None, rng=np.random) -> None:
        """Transform the morphology.

        The morphology is scaled, rotated around Y and aligned according to the orientation field.
        If jitter parameters are provided, jitter process is also applied to the morphology.
        """
        transform = np.identity(4)
        transform[:3, :3] = np.matmul(orientation[0], random_rotation_y(n=1, rng=rng)[0])
        if scale is not None:
            transform = scale * transform
        mt.transform(morph, transform)

        if self.params.rotational_jitter_std is not None:
            rotational_jitter(
                morph, RotationParameters(std_angle=self.params.rotational_jitter_std), rng=rng
            )
        if self.params.scaling_jitter_std is not None:
            scale_morphology(
                morph, section_scaling=ScaleParameters(std=self.params.scaling_jitter_std), rng=rng
            )

    @staticmethod
    def _convert_apical_sections_to_apical_points(neuron, apical_sections):
        """Convert apical point sections to positions."""
        return [neuron.sections[apical_section].points[-1] for apical_section in apical_sections]

    @staticmethod
    def _convert_apical_sections_to_NRN_sections(apical_points, morph_path):
        """Convert apical point sections to neuron sections."""
        return [
            nrnhines.isolate(_to_be_isolated)(morph_path, apical_point)
            for apical_point in apical_points
        ]


class CellHelper:  # pragma: no cover
    """Loads spatial information and provides basic functionality to query spatial properties."""

    def __init__(self, cells_file):
        """The CellHelper constructor."""
        self.cells = CellCollection.load_mvd3(cells_file)

    def positions(self, mtype):
        """Return a generator of mtype cell positions."""
        return (self.cells.positions[gid] for gid in self._filter_by_mtype(mtype))

    def _filter_by_mtype(self, mtype):
        """Returns ids of cell with the given mtype."""
        return self.cells.properties.index[self.cells.properties.mtype.str.contains(mtype)]
