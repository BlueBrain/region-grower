"""Script to generate a synthesis recipe."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of region-grower.
# See https://github.com/BlueBrain/region-grower for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import copy
import json

import numpy as np


def run():
    """Get all m-types within current circuit."""
    # mtypes_all = np.unique(test.cells.properties.mtype.values)
    mtypes_all = np.array(
        [
            "L1_DAC",
            "L1_HAC",
            "L1_LAC",
            "L1_NGC-DA",
            "L1_NGC-SA",
            "L1_SAC",
            "L23_BP",
            "L23_BTC",
            "L23_CHC",
            "L23_DBC",
            "L23_LBC",
            "L23_MC",
            "L23_NBC",
            "L23_NGC",
            "L23_SBC",
            "L2_IPC",
            "L2_TPC:A",
            "L2_TPC:B",
            "L3_TPC:A",
            "L3_TPC:B",
            "L4_BP",
            "L4_BTC",
            "L4_CHC",
            "L4_DBC",
            "L4_LBC",
            "L4_MC",
            "L4_NBC",
            "L4_NGC",
            "L4_SBC",
            "L4_SSC",
            "L4_TPC",
            "L4_UPC",
            "L5_BP",
            "L5_BTC",
            "L5_CHC",
            "L5_DBC",
            "L5_LBC",
            "L5_MC",
            "L5_NBC",
            "L5_NGC",
            "L5_SBC",
            "L5_TPC:A",
            "L5_TPC:B",
            "L5_TPC:C",
            "L5_UPC",
            "L6_BP",
            "L6_BPC",
            "L6_BTC",
            "L6_CHC",
            "L6_DBC",
            "L6_HPC",
            "L6_IPC",
            "L6_LBC",
            "L6_MC",
            "L6_NBC",
            "L6_NGC",
            "L6_SBC",
            "L6_TPC:A",
            "L6_TPC:C",
            "L6_UPC",
        ]
    )

    # Get data from json saved files
    with open("pc_in_types.json", "r") as F:
        pc_in = json.load(F)

    with open("defaults.json", "r") as F:
        defaults = json.load(F)

    with open("pc_specific.json", "r") as F:
        pc_specific = json.load(F)
    # Create new dictionary for all mtypes
    mdict = {}

    # Fill in dictionary with parameters for each m-type
    for m in mtypes_all:
        mdict[m] = copy.deepcopy(defaults[pc_in[m]])
        # Redefine pc-specific data
        if m in pc_specific:
            mdict[m]["apical"].update(pc_specific[m])

    # Save the updated parameters for all m-types
    with open("tmd_parameters.json", "w") as F:
        json.dump(mdict, F, indent=4, sort_keys=True)


if __name__ == "__main__":
    run()
