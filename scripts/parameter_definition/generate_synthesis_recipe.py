import json
import numpy as np
import copy

def run():
    # Get all m-types within current circuit
    # mtypes_all = np.unique(test.cells.properties.mtype.values)
    mtypes_all = np.array([u'L1_DAC', u'L1_HAC', u'L1_LAC', u'L1_NGC-DA', u'L1_NGC-SA',
                           u'L1_SAC', u'L23_BP', u'L23_BTC', u'L23_CHC', u'L23_DBC',
                           u'L23_LBC', u'L23_MC', u'L23_NBC', u'L23_NGC', u'L23_SBC',
                           u'L2_IPC', u'L2_TPC:A', u'L2_TPC:B', u'L3_TPC:A', u'L3_TPC:B',
                           u'L4_BP', u'L4_BTC', u'L4_CHC', u'L4_DBC', u'L4_LBC', u'L4_MC',
                           u'L4_NBC', u'L4_NGC', u'L4_SBC', u'L4_SSC', u'L4_TPC', u'L4_UPC',
                           u'L5_BP', u'L5_BTC', u'L5_CHC', u'L5_DBC', u'L5_LBC', u'L5_MC',
                           u'L5_NBC', u'L5_NGC', u'L5_SBC', u'L5_TPC:A', u'L5_TPC:B',
                           u'L5_TPC:C', u'L5_UPC', u'L6_BP', u'L6_BPC', u'L6_BTC', u'L6_CHC',
                           u'L6_DBC', u'L6_HPC', u'L6_IPC', u'L6_LBC', u'L6_MC', u'L6_NBC',
                           u'L6_NGC', u'L6_SBC', u'L6_TPC:A', u'L6_TPC:C', u'L6_UPC'])

    # Get data from json saved files
    with open('pc_in_types.json', 'r') as F:
        pc_in = json.load(F)

    with open('defaults.json', 'r') as F:
        defaults = json.load(F)

    with open('pc_specific.json', 'r') as F:
        pc_specific = json.load(F)
    # Create new dictionary for all mtypes
    mdict = {}

    # Fill in dictionary with parameters for each m-type
    for m in mtypes_all:
        mdict[m] = copy.deepcopy(defaults[pc_in[m]])
        # Redefine pc-specific data
        if m in pc_specific:
            mdict[m]['apical'].update(pc_specific[m])

    # Save the updated parameters for all m-types
    with open('tmd_parameters.json', 'w') as F:
        json.dump(mdict, F, indent=4, sort_keys=True)

if __name__ == '__main__':
    run()
