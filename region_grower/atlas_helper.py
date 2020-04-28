'''An atlas helper to lookup the depths and orientations from an atlas
without have to reason in term of [PH][1-6] and [PH]y'''
import operator
from itertools import accumulate
from typing import List, Optional, Tuple, Union

import numpy as np
from voxcell import OrientationField, VoxelData
from voxcell.nexus.voxelbrain import Atlas

from region_grower import RegionGrowerError

Point = Union[List[float], np.array]


class AtlasHelper:
    '''Atlas helper provides two lookup functions for region grower:
    - lookup_target_reference_depths
    - lookup_orientation
    '''
    def __init__(self, atlas: Atlas):
        '''AtlasHelper constructor

        Args:
            atlas: the atlas
        '''
        self.atlas = atlas
        self.thicknesses = [self.layer_thickness(layer) for layer in range(1, 7)]
        self.depths = VoxelData.reduce(operator.sub, [self.pia_coord, atlas.load_data("[PH]y")])
        self.brain_regions = atlas.load_data("brain_regions")
        self.orientations = atlas.load_data("orientation", cls=OrientationField)

    def layer_thickness(self, layer: int) -> Atlas:
        '''Returns an atlas of the layer thickness'''
        layer_bounds = self.atlas.load_data(f"[PH]{layer}")
        return layer_bounds.with_data(layer_bounds.raw[..., 1] - layer_bounds.raw[..., 0])

    @property
    def pia_coord(self) -> Atlas:
        '''Returns an atlas of the pia coordinate along the principal axis'''
        layer_1 = self.atlas.load_data("[PH]1")
        return layer_1.with_data(layer_1.raw[..., 1])

    def lookup_orientation(
            self,
            position: Point,
            vector: Optional[Point] = None
    ) -> np.array:
        """Returns the looked-up orientation for the given position."""
        if vector is None:
            vector = [0, 1, 0]  # assume direction towards the pia.

        return np.dot(self.orientations.lookup(position), vector)[0]

    def lookup_target_reference_depths(
            self,
            position: Point,
            cortical_depths: List[float]
    ) -> Tuple[np.array, np.array]:
        '''Returns the target and the reference depth for a given neuron position.

        Args:
            position: the position of a neuron in the atlas
            cortical_depths: the depths of the 6 layers in the cortical column

        First item is the depth of the lower (the further away from the pia) boundary
        of the layer in which is located 'position'.

        Second one is the equivalent value for the same layer but in the cortical column.
        '''
        current_depth = self.depths.lookup(position)
        depths = accumulate(thickness.lookup(position) for thickness in self.thicknesses)
        for depth, cortical_depth in zip(depths, cortical_depths):
            if current_depth <= depth:
                return depth, cortical_depth

        raise RegionGrowerError(f"Current depth ({current_depth}) for position ({position}) is"
                                " outside of circuit boundaries")

    def get_layer_boundary_depths(self, position: Point) -> np.array:
        """Return layer depths at the given position.

        Args:
            position: the position of a neuron in the atlas
        """
        return np.cumsum([0] + [thickness.lookup(position) for thickness in self.thicknesses])
