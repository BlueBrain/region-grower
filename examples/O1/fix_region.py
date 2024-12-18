# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of region-grower.
# See https://github.com/BlueBrain/region-grower for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Fix regions."""

import json

import pandas as pd
from voxcell.cell_collection import CellCollection

if __name__ == "__main__":
    params = json.load(open("tmd_parameters.json"))
    distr = json.load(open("tmd_distributions.json"))
    try:
        params = {"O0": params["Isocortex"]}
        distr = {"O0": distr["Isocortex"]}
        json.dump(params, open("tmd_parameters.json", "w"), indent=4)
        json.dump(distr, open("tmd_distributions.json", "w"), indent=4)
    except:  # noqa: E722
        pass

    cells_df = CellCollection.load("nodes.h5").as_dataframe()
    cells_df["region"] = "O0"

    # zoom into center of region to avoid boundary effects
    cells_df = cells_df[(400 > cells_df.x) & (cells_df.x > -400)]
    cells_df = cells_df[(2040 > cells_df.y)]
    cells_df = cells_df[(400 > cells_df.z) & (cells_df.z > -400)]
    dfs = []
    for mtype in cells_df.mtype.unique():
        dfs.append(cells_df[cells_df.mtype == mtype].sample(10))
    cells_df = pd.concat(dfs).reset_index(drop=True)
    print(len(cells_df.index), "cells to synthesize")
    cells_df.index += 1
    CellCollection.from_dataframe(cells_df).save("nodes.h5")
