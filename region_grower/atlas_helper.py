"""An atlas helper to lookup the depths and orientations from an atlas.

This helper allows simple lookups without having to reason in term of [PH][1-6] and [PH]y.
"""
import operator
from pathlib import Path
from typing import List
from typing import Union

import numpy as np
import yaml
from voxcell import OrientationField
from voxcell import VoxelData
from voxcell.nexus.voxelbrain import Atlas

Point = Union[List[float], np.array]


class AtlasHelper:
    """Atlas helper for region grower."""

    def __init__(self, atlas: Atlas, region_structure_path: str):
        """The AtlasHelper constructor.

        Args:
            atlas: the atlas
            region_structure_path: path to region structure yaml file
        """
        self.atlas = atlas
        if region_structure_path is not None and Path(region_structure_path).exists():
            with open(region_structure_path) as region_file:
                self.region_structure = yaml.safe_load(region_file)
        else:
            raise ValueError("Please provide an existing region_structure file.")
        self.layers = self.region_structure["layers"]
        self.thicknesses = [self.layer_thickness(layer) for layer in self.layers]
        self.depths = VoxelData.reduce(operator.sub, [self.pia_coord, atlas.load_data("[PH]y")])
        self.brain_regions = atlas.load_data("brain_regions")
        self.orientations = atlas.load_data("orientation", cls=OrientationField)

    def layer_thickness(self, layer: int) -> Atlas:
        """Returns an atlas of the layer thickness."""
        layer_bounds = self.atlas.load_data(f"[PH]{layer}")
        return layer_bounds.with_data(layer_bounds.raw[..., 1] - layer_bounds.raw[..., 0])

    @property
    def pia_coord(self) -> Atlas:
        """Returns an atlas of the pia coordinate along the principal axis."""
        top_layer = self.atlas.load_data(f"[PH]{self.layers[0]}")
        return top_layer.with_data(top_layer.raw[..., 1])

    def get_layer_boundary_depths(self, position: Point) -> np.array:
        """Return layer depths at the given position.

        Args:
            position: the position of a neuron in the atlas
        """
        pos = np.array(position, ndmin=2)
        result = np.zeros((len(self.thicknesses) + 1, pos.shape[0]))
        all_thicknesses = [thickness.lookup(pos) for thickness in self.thicknesses]
        result[1:, :] = np.cumsum(all_thicknesses, axis=0)
        return result
