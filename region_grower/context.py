'''Use spatial properties to grow a cell'''

from collections import namedtuple

import numpy as np
from voxcell import OrientationField
from voxcell.nexus.voxelbrain import Atlas
from voxcell.cell_collection import CellCollection

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

    def correct_position_orientation(self, params, position):
        '''Return a copy of the passed parameter with the correct orientation and
        recentered at [0,0,0]'''
        result = dict(params)
        result['origin'] = position

        for neurite_type in params['grow_types']:
            if isinstance(params[neurite_type]['orientation'], list):
                result[neurite_type]['orientation'] = [
                    self.get_orientation(position, orient)
                    for orient in params[neurite_type]['orientation']
                ]

        return result

    def get_orientation(self, position, vector=None):
        """Returns the orientation for the selected cell with the corresponding
           input position, as extracted from spatial properties.
        """
        if vector is None:
            vector = [0, 1, 0]  # assume direction towards the pia.

        return np.dot(self.orientations.lookup(position), vector)[0]

    def get_depth(self, position):
        """Returns the depth for the selected cell with the corresponding
           input position, as extracted from spatial properties.
        """
        return self.depths.lookup(position)

    def get_thickness_layers(self, position):
        """Returns the thickness of layers for the selected cell with the corresponding
           input position, as extracted from spatial properties.
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


class CellsSpaceContext(SpaceContext):
    """Loads spatial information and provides
       basic functionality to query spatial properties
       required for neuronal synthesis. In addition to
       SpatialContext also loads the cell information to be used.
    """

    def __init__(self, atlas_file, cells_file):
        """
        Basic loading of an Atlas and Cells
        using voxcell
        """
        atlas = Atlas.open(atlas_file)
        super(CellsSpaceContext, self).__init__(atlas)
        self.cells = CellCollection.load(cells_file)

    def get_cell_position(self, n=1):
        """Returns the position of the cell with ID=n
        """
        return self.cells.positions[n]

    def get_mtype(self, n=1):
        """Returns the mtype of the selected cell with ID=n
        """
        return self.cells.properties.mtype[n]

    def filter_by_mtype(self, mtype):
        '''Returns ids of cell with the given mtype'''
        return self.cells.properties.index[self.cells.properties.mtype.str.contains(mtype)]
