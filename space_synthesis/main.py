import os
# Import space
from space_synthesis.context import SpaceContext
import numpy as np
# Check generated data
import tmd
from tmd import view
import mayavi
from mayavi import mlab
import tns
from tns import NeuronGrower

def correct_params(context, params, position):
    result = dict(params)
    result['apical'].update({
        'branching_method': 'bio_oriented',
        'randomness': 0.18,
        'targeting': 0.18,
        'apical_distance': 200,
        'radius': 0.7,
    })
    result['apical']['orientation'] = [
        context.get_orientation(position, i) for i in params['apical']['orientation']
    ]

    return result

def reset_origin(context, params, cell_id):
    '''Return a copy of the params with the origin updated '''
    result = dict(params)
    result['origin'] = context.get_cell_position(cell_id)
    return result

def use_space4input(context, all_ids, training_folder):
    cell_id = np.random.choice(all_ids)
    position = context.get_cell_position(cell_id)
    distributions = tns.extract_input.distributions(training_folder)
    original_params = tns.extract_input.parameters(origin=position, method='tmd',
                                                   neurite_types=['basal', 'apical'])
    modified_params = correct_params(context, original_params, position)

    return distributions, modified_params


def validate(output):
    # 1. Check somata positions
    pop = tmd.io.load_population(output)
    somata = np.array([n.soma.get_center() for n in pop.neurons])
    somataR = np.array([n.soma.get_diameter() for n in pop.neurons])

    # mlab.points3d(somata[:,0], somata[:,1], somata[:,2], somataR,
    #               colormap='Reds', scale_factor=2.)

    # 2. Check subset of cells in space (2d)
    view.view.population(pop, title='')
    import matplotlib.pyplot as plt
    plt.show()

def run(circuit_config, training_folder, output_folder, mtype, num_cells):
    context = SpaceContext(circuit_config)

    ids_good_mtype = context.filter_by_mtype(mtype)

    distr, general_params = use_space4input(context=context,
                                            all_ids=ids_good_mtype,
                                            training_folder=training_folder)

    cell_ids = np.random.choice(ids_good_mtype, size=num_cells)

    for i, cell_id in enumerate(cell_ids):
        cell_params = reset_origin(context, general_params, cell_id)
        neuron = tns.NeuronGrower(input_parameters=cell_params,
                                  input_distributions=distr).grow()
        neuron.write(os.path.join(output_folder, 'ID_{}.h5'.format(i+1)))


if __name__=='__main__':
    OUTPUT_FOLDER = './outputs/test-L3_TPC/'
    PATH = os.path.dirname(os.path.abspath(__file__))
    run(circuit_config=os.path.join(PATH, '../templates/CircuitConfig'),
        training_folder='/home/bcoste/workspace/morphology/space_synthesis/inputs/L3_TPC/',
        output_folder=OUTPUT_FOLDER,
        mtype='L3_TPC',
        num_cells=100)

    # validate(OUTPUT_FOLDER)
