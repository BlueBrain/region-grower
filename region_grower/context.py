'''Use spatial properties to grow a cell'''

from collections import namedtuple
import json

import numpy as np
from voxcell import OrientationField
from voxcell.cell_collection import CellCollection

from tns import NeuronGrower

from region_grower import modify, RegionGrowerError


SpacePos = namedtuple(
    'SpacePos', ['position', 'depth', 'orientation', 'thickness_layers'])


class SpaceContext(object):
    """Loads spatial information and provides
    basic functionality to query spatial properties
    required for neuronal synthesis.
    """

    def __init__(self, atlas, tmd_distributions_path, tmd_parameters_path):
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

        with open(tmd_distributions_path, 'r') as f:
            self.tmd_distributions = json.load(f)

        with open(tmd_parameters_path, 'r') as f:
            self.tmd_parameters = json.load(f)

    def synthesize(self, position, mtype):
        '''Synthesize a cell based on the position and mtype'''
        par = self._correct_position_orientation_scaling(
            self.tmd_parameters[mtype],
            self.tmd_distributions['metadata']['cortical_thickness'],
            position)

        N = NeuronGrower(
            input_parameters=par, input_distributions=self.tmd_distributions['mtypes'][mtype])
        return N.grow()

    def _cumulative_thicknesses(self, position):
        '''cumulative thicknesses starting at layer 1'''
        return np.cumsum(
            [self.L1.lookup(position),
             self.L2.lookup(position),
             self.L3.lookup(position),
             self.L4.lookup(position),
             self.L5.lookup(position),
             self.L6.lookup(position)])

    def _get_layer(self, position, cumulative_thickness):
        '''Find out in which layer position is'''
        depth = self.depths.lookup(position)
        for indice, thickness in enumerate(cumulative_thickness):
            if thickness > depth:
                return indice

        return len(cumulative_thickness)

    def _correct_position_orientation_scaling(self, params, cortical_thickness, position):
        '''Return a copy of the passed parameter with the correct orientation and
        recentered at [0,0,0]'''
        result = dict(params)
        result['origin'] = position

        for neurite_type in params['grow_types']:
            if isinstance(params[neurite_type]['orientation'], list):
                result[neurite_type]['orientation'] = [
                    self._get_orientation(position, orient)
                    for orient in params[neurite_type]['orientation']
                ]

        cumulative_thickness = self._cumulative_thicknesses(position)
        layer = self._get_layer(position, cumulative_thickness)
        target_thickness = cumulative_thickness[layer - 1]

        if target_thickness < 1e-8:
            raise RegionGrowerError(
                'Zero thickness in space. This will not generate cell with ID: ')
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
        self.cells = CellCollection.load(cells_file)

    def positions(self, mtype):
        '''Return a generator of mtype cell positions'''
        return (self.cells.positions[gid] for gid in self._filter_by_mtype(mtype))

    def _filter_by_mtype(self, mtype):
        '''Returns ids of cell with the given mtype'''
        return self.cells.properties.index[self.cells.properties.mtype.str.contains(mtype)]
