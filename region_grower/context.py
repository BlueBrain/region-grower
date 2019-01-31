'''Use spatial properties to grow a cell'''

from collections import namedtuple

import numpy as np
from voxcell import OrientationField

SpacePos = namedtuple(
    'SpacePos', ['position', 'depth', 'orientation', 'thickness_layers'])


class SpaceContext(object):
    """Loads spatial information and provides
    basic functionality to query spatial properties
    required for neuronal synthesis.
    """

    def __init__(self, atlas):
        """Initialization with an atlas (of a BBP circuit)"""
        self.atlas = atlas
        self.hierarchy = self.atlas.load_hierarchy()
        self.brain_regions = self.atlas.load_data('brain_regions')
        self.depths = self.atlas.load_data('depth')
        self.orientations = self.atlas.load_data(
            'orientation', cls=OrientationField)
        self.L1 = self.atlas.load_data('thickness:L1')
        self.L2 = self.atlas.load_data('thickness:L2')
        self.L3 = self.atlas.load_data('thickness:L3')
        self.L4 = self.atlas.load_data('thickness:L4')
        self.L5 = self.atlas.load_data('thickness:L5')
        self.L6 = self.atlas.load_data('thickness:L6')

    def get_corrected_params(self, params, position):
        '''Return a copy of the passed parameter with the correct orientation and
        recentered at [0,0,0]'''
        result = dict(params)
        result['origin'] = [0, 0, 0]

        if 'apical' in params['grow_types']:
            result['apical']['orientation'] = [
                self.get_orientation(position, i)
                for i in params['apical']['orientation']
            ]

        return result

    def get_brain_region_name(self, position):
        """Returns the brain region for the selected cell with ID=n that corresponds
           to the position extracted from get_cell_position
        """
        return self.hierarchy.collect('id', self.brain_regions.lookup(position), 'name')

    def get_orientation(self, position, vector=None):
        """Returns the orientation for the selected cell with ID=n that corresponds
           to the position extracted from get_cell_position
        """
        if vector is None:
            vector = [0, 1, 0]  # assume direction towards the pia.

        return np.dot(self.orientations.lookup(position), vector)[0]

    def get_depth(self, position):
        """Returns the depth for the selected cell with ID=n that corresponds
           to the position extracted from get_cell_position
        """
        return self.depths.lookup(position)

    def get_thickness_layers(self, position):
        """Returns the thickness of layers for the selected cell with ID=n that corresponds
           to the position extracted from get_cell_position
        """
        return [self.L1.lookup(position),
                self.L2.lookup(position),
                self.L3.lookup(position),
                self.L4.lookup(position),
                self.L5.lookup(position),
                self.L6.lookup(position)]

    def get_data(self, position):
        '''Get data'''
        depth = self.get_depth(position)
        orientation = self.get_orientation(position)
        thickness_layers = self.get_thickness_layers(position)
        return SpacePos(position, depth, orientation, thickness_layers)
