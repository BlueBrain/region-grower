import json
import numpy as np
from tns import extract_input

data = {'mouse': '/gpfs/bbp.cscs.ch/project/proj49/LidaSYN/BioInput/Mouse/',
        'rat': '/gpfs/bbp.cscs.ch/project/proj49/LidaSYN/BioInput/Rat/'}
output_file = 'tmd_distributions.json'

# Get all m-types within current circuit
# mtypes_all = np.unique(test.cells.properties.mtype.values)
mtypes_all = np.array([u'L1_DAC', u'L1_HAC', u'L1_LAC', u'L1_NGC-DA', u'L1_NGC-SA',
                       u'L1_SAC', u'L23_BP', u'L23_BTC', u'L23_CHC', u'L23_DBC',
                       u'L23_LBC', u'L23_MC', u'L23_NBC', u'L23_NGC', u'L23_SBC',
                       u'L2_IPC', u'L2_TPC', u'L3_TPC',
                       u'L4_BP', u'L4_BTC', u'L4_CHC', u'L4_DBC', u'L4_LBC', u'L4_MC',
                       u'L4_NBC', u'L4_NGC', u'L4_SBC', u'L4_SSC', u'L4_TPC', u'L4_UPC',
                       u'L5_BP', u'L5_BTC', u'L5_CHC', u'L5_DBC', u'L5_LBC', u'L5_MC',
                       u'L5_NBC', u'L5_SBC', u'L5_TPC', u'L5_UPC',
                       u'L6_BPC', u'L6_BTC', u'L6_CHC',
                       u'L6_DBC', u'L6_HPC', u'L6_IPC', u'L6_LBC', u'L6_MC', u'L6_NBC',
                       u'L6_NGC', u'L6_SBC', u'L6_TPC', u'L6_UPC'])


def run(species_select, mtypes, feature='path_distances_2'):
    # Create new dictionary for all mtypes
    mdict = {}

    # Fill in dictionary with distributions for each m-type
    for m in mtypes:
        mdict[m] = extract_input.distributions(data.get(species_select) + m + '/',
                                               neurite_types=['basal', 'apical'],
                                               feature=feature)
    # Hack for missing mtypes
    mdict['L5_NGC'] = mdict['L6_NGC']
    mdict['L6_BP'] = mdict['L5_BP']

    # Save the updated parameters for all m-types
    with open(output_file, 'w') as F:
        json.dump(mdict, F, sort_keys=True)


if __name__ == '__main__':
    run(species_select='mouse', mtypes=mtypes_all)
