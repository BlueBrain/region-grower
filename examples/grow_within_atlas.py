''' Example on how to use the TMD synthesis in space'''
import json
import os

import numpy as np

import tns
from region_grower import context, modify

# Parameters of growth
mtype = 'L2_TPC:A'
number_of_cells = 100
atlas_file = './Atlas2019/20190118/'
cells_file = './Atlas2019/circuit.mvd3.metypes'
template_distr = 'mouse_distributions.json'
template_params = 'tmd_parameters.json'
output_path = '../../Column_Synthesized/'

# Get positions of a defined m-type from the Atlas
mySpace = context.CellsSpaceContext(atlas_file, cells_file)
selectedIDs = mySpace.filter_by_mtype(mtype)
somata = [mySpace.get_cell_position(i) for i in selectedIDs[:number_of_cells]]

# Get distributions from a sample population of cells
with open(template_distr, 'r') as F:
    distributions = json.load(F)

with open(template_params, 'r') as F:
    parameters = json.load(F)

distr = distributions[mtype]
params = parameters[mtype]

# MOUSE layer thickness from DeFelipe data
# RAT layer thickness from Cell, 2015
thickness= {'mouse': [118.3,  # L1
                      93.01,  # L2
                      169.5,  # L3
                      178.6,  # L4
                      349.2,  # L5
                      420.5  # L6
                     ],
            'rat': [165,  # L1
                    149,  # L2
                    353,  # L3
                    190,  # L4
                    525,  # L5
                    700  # L6
                   ]
             }

# RAT layer thickness TBD (not CORRECT)

def create_population(distr, params, space, somata, num_cells=100, output_path='./Results',
                       output_name='L4_TPC', formats=None, species='mouse', layer=4):
    '''Generates N cells according to input distributions
       and saves them in the selected files formats (default: h5, swc, asc).
       thickness_ref: initial thickness corresponding to input cell
                      taken from species, layer combination
       layer: defines the layer of the somata
       num_cells: number of cells to grow
    '''
    if formats is None:
        formats = ['swc','h5','asc']

    # Creates directories to save selected formats
    for f in formats:
        if not os.path.isdir(output_path + f):
            os.mkdir(output_path + f)

    for i in range(num_cells):
        # Correct initial position and orientation
        position = somata[i]
        par = space.correct_position_orientation(params, position)

        # Correct barcode scaling according to available space
        target_thickness = np.float(np.sum(space.get_thickness_layers(position)[:layer]))
        if target_thickness != 0:
            reference_thickness = np.float(np.sum(thickness[species][:layer]))

            par = modify.input_scaling(par, reference_thickness, target_thickness)

            # Grow neuron according to specified parameters and distributions
            N = tns.NeuronGrower(input_parameters=par, input_distributions=distr)
            neuron = N.grow()

            for f in formats:
                neuron.write(output_path + '/' + f + '/' + output_name + '_' + str(i+1) + '.' + f)
        else:
            print('Zero thickness in space. This will not generate cell with ID: ', i)


if __name__ == '__main__':
    # Create a sample population of L2_TPC cells within layer 2
    create_population(distr, params, mySpace, somata, num_cells=number_of_cells,
                      output_path=output_path, output_name=mtype, species='mouse', layer=2)
