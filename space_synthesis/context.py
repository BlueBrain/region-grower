# Use spatial properties to grow a cell

from voxcell import OrientationField
from voxcell.nexus.voxelbrain import Atlas
#from voxcell.nexus.voxelbrain import CellCollection
from voxcell import CellCollection
import numpy as np

from collections import namedtuple
SpacePos = namedtuple('SpacePos', ['position','depth', 'orientation', 'thickness_layers'])

class SpaceContext(object):
    """Loads spatial information and provides
    basic functionality to query spatial properties
    required for neuronal synthesis.
    """
    def __init__(self, atlas_file, cells_file):
        """Initialization with an atlas (of a BBP circuit)
        and the corresponding cell types
        """
        self.atlas = Atlas.open(atlas_file)
        self.cells = CellCollection.load(cells_file)
        self.hierarchy = self.atlas.load_hierarchy()
        self.brain_regions = self.atlas.load_data('brain_regions')
        self.depths = self.atlas.load_data('depth')
        self.orientations = self.atlas.load_data('orientation', cls=OrientationField)
        self.L1 = self.atlas.load_data('thickness:L1')
        self.L2 = self.atlas.load_data('thickness:L2')
        self.L3 = self.atlas.load_data('thickness:L3')
        self.L4 = self.atlas.load_data('thickness:L4')
        self.L5 = self.atlas.load_data('thickness:L5')
        self.L6 = self.atlas.load_data('thickness:L6')

    def get_cell_position(self, n=23):
        """Returns the position of the cell with ID=n
        """
        return self.cells.positions[n]

    def get_mtype(self, n=23):
        """Returns the mtype of the selected cell with ID=n
        """
        return self.cells.properties.mtype[n]

    def get_brain_region(self, n=23):
        """Returns the brain region of the selected cell with ID=n
        """
        return self.cells.properties.region[n]

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
            vector = [0, 1, 0] # assume direction towards the pia.

        return np.dot(self.orientations.lookup(position), vector)[0]

    def filter_by_mtype(self, mtype):
        '''Returns ids of cell with the given mtype'''
        return self.cells.properties.index[self.cells.properties.mtype.str.contains(mtype)]

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
        depth = self.get_depth(position)
        orientation = self.get_orientation(position)
        thickness_layers = self.get_thickness_layers(position)
        return SpacePos(position, depth, orientation, thickness_layers)
