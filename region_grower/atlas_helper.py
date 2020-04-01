'''An atlas helper to lookup the depths and orientations from an atlas
without have to reason in term of [PH][1-6] and [PH]y'''
import operator
from itertools import accumulate, chain
from typing import List, Optional, Tuple, Union

import numpy as np

from voxcell import OrientationField, VoxelData
from voxcell.nexus.voxelbrain import Atlas

from region_grower import RegionGrowerError
from region_grower.utils import pairwise

Point = Union[List[float], np.array]


class AtlasHelper:
    '''Atlas helper provides two lookup functions for region grower:
    - lookup_depths
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

    def lookup_depths(
            self,
            position: Point,
            cortical_thicknesses: List[float]
    ) -> Tuple[np.array, np.array]:
        '''Returns a tuple of 2 depths (reminder: the pia has depth 0)

        Args:
            position: the position whose depths will be returned
            cortical_thicknesses: the thicknesses of the 6 layers in the cortical column

        First item is the depth of the upper (the closest from the pia) boundary of the layer in
        which is located 'position'.

        Second one is the equivalent value for the same layer but in the cortical column.
        '''
        current_depth = self.depths.lookup(position)

        thicknesses = (thickness.lookup(position) for thickness in self.thicknesses)
        layer_depth_bounds = pairwise(accumulate(chain([0], thicknesses)))

        cortical_depths = accumulate(chain([0], cortical_thicknesses))

        for (depth_start, depth_end), cortical_depth in zip(layer_depth_bounds, cortical_depths):
            if depth_end > current_depth:
                return depth_start, cortical_depth

        raise RegionGrowerError(f"Position ({position}) outside of circuit boundaries")
