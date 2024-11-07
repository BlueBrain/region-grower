"""Generate atlas for tests."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of region-grower.
# See https://github.com/BlueBrain/region-grower for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=missing-function-docstring
import json
import os
from itertools import cycle
from itertools import islice
from itertools import repeat

import numpy as np
import pandas as pd
from voxcell import CellCollection

DF_SIZE = 12


def generate_small_O1(directory):
    """Dump a small O1 atlas in folder path."""
    # fmt: off
    os.system(
        " ".join(
            [
                "brainbuilder", "atlases",
                "-n", "6,5,4,3,2,1",
                "-t", "200,100,100,100,100,200",
                "-d", "100",
                "-o", str(directory),
                "column",
                "-a", "1000",
            ]
        )
    )
    # fmt: on
    return str(directory)


def generate_cells_df():
    """Raw data for the cell collection."""
    x = [200] * DF_SIZE
    y = [200] * DF_SIZE
    z = [200] * DF_SIZE
    df = pd.DataFrame(
        {
            "mtype": list(repeat("L2_TPC:A", DF_SIZE)),
            "region": list(repeat("mc0;2", DF_SIZE)),
            "morphology": list(
                islice(
                    cycle(
                        [
                            (
                                "dend-C250500A-P3_axon-C190898A-P2_-"
                                "_Scale_x1.000_y1.025_z1.000_-_Clone_2"
                            ),
                            "C240300C1_-_Scale_x1.000_y0.975_z1.000_-_Clone_55",
                            "dend-Fluo15_right_axon-Fluo2_right_-_Clone_37",
                        ]
                    ),
                    DF_SIZE,
                )
            ),
            "x": x,
            "y": y,
            "z": z,
        }
    )
    # add a cell from another region without region_structure information
    df.loc[12] = df.loc[11]
    df.loc[12, "region"] = "other"
    df.loc[13] = df.loc[11]
    df.loc[13, "region"] = "other without layer info"
    df.index += 1
    return df


def generate_cell_collection(cells_df):
    """The cell collection."""
    return CellCollection.from_dataframe(cells_df)


def input_cells_path(tmpdir):
    """The path to the MVD3 file containing the cell collection."""
    return tmpdir / "input_cells.mvd3"


def generate_input_cells(cell_collection, tmpdir):
    """The path to the MVD3 file containing the cell collection."""
    filename = input_cells_path(tmpdir)
    cell_collection.save_mvd3(filename)
    return filename


def generate_axon_morph_tsv(tmpdir):
    """The path to the TSV file containing the axon morphologies."""
    df = pd.DataFrame(
        {
            "morphology": list(
                islice(
                    cycle(
                        [
                            "C170797A-P1",
                            "UNKNOWN",
                            None,
                        ]
                    ),
                    DF_SIZE,
                )
            ),
            "scale": np.repeat([0.5, 1, None], np.ceil(DF_SIZE // 3))[:DF_SIZE],
        }
    )
    df.loc[12, "morphology"] = "C170797A-P1"
    df.loc[12, "scale"] = 1
    df.loc[13, "morphology"] = "C170797A-P1"
    df.loc[13, "scale"] = 1
    filename = tmpdir / "axon_morphs.tsv"
    df.to_csv(filename, sep="\t", na_rep="N/A")
    return filename


def get_tmd_parameters(filename):
    """The TMD parameters."""
    with open(filename, "r", encoding="utf-8") as f:
        tmd_parameters = json.load(f)
    return tmd_parameters


def get_tmd_distributions(filename):
    """The TMD distributions."""
    with open(filename, "r", encoding="utf-8") as f:
        tmd_distributions = json.load(f)
    return tmd_distributions


def get_cell_position():
    """The cell position."""
    return [0, 500, 0]


def get_cell_mtype():
    """The cell mtype."""
    return "L2_TPC:A"


def get_cell_orientation():
    """The cell orientation."""
    return np.eye(3).reshape(1, 3, 3)
