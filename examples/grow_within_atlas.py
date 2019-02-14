''' Example on how to use the TMD synthesis in space'''
import json
import os
from itertools import islice

import tns
from region_grower.context import SpaceContext, CellHelper

from voxcell.nexus.voxelbrain import Atlas

def create_population(context, cell_path, mtype, number_of_cells, output_path,
                       output_name, formats=None):
    '''Generates N cells according to input distributions
       and saves them in the selected files formats (default: h5, swc, asc).
       thickness_ref: initial thickness corresponding to input cell
                      taken from species, layer combination
       layer: defines the layer of the somata
       num_cells: number of cells to grow
    '''

    somata = CellHelper(cell_path).positions(mtype)

    if formats is None:
        formats = ['swc', 'h5', 'asc']

    # Creates directories to save selected formats
    for f in formats:
        if not os.path.isdir(output_path + f):
            os.mkdir(output_path + f)

    for i, position in enumerate(islice(somata, number_of_cells)):
        neuron = context.synthesize(position, mtype)
        for f in formats:
            neuron.write(output_path + '/' + f + '/' + output_name + '_' + str(i+1) + '.' + f)


if __name__ == '__main__':
    CONTEXT = SpaceContext(
        atlas=Atlas.open('/gpfs/bbp.cscs.ch/project/proj68/entities/dev/atlas/ccf_2017-25um/20190118'),
        tmd_distributions_path='/gpfs/bbp.cscs.ch/project/proj68/home/kanari/SynthInput/mouse_distributions.json',
        tmd_parameters_path='/gpfs/bbp.cscs.ch/project/proj68/home/kanari/SynthInput/tmd_parameters.json')


    create_population(CONTEXT,
                      '/gpfs/bbp.cscs.ch/project/proj68/circuits/COLUMN/20190211.dev/circuit.mvd3.metypes',
                      mtype='L2_TPC:A',
                      number_of_cells=100,
                      output_path='.',
                      output_name='L2_TPC:A')
