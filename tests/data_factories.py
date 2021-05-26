"""Generate atlas for tests"""
# pylint: disable=missing-function-docstring
import json
from itertools import cycle
from itertools import islice
from itertools import repeat
from os import devnull
from subprocess import call

import numpy as np
import pandas as pd
from voxcell import CellCollection

DF_SIZE = 12


def generate_small_O1(directory):
    """Dump a small O1 atlas in folder path"""
    # fmt: off
    with open(devnull, "w") as f:
        call(
            [
                "brainbuilder", "atlases",
                "-n", "6,5,4,3,2,1",
                "-t", "200,100,100,100,100,200",
                "-d", "100",
                "-o", str(directory),
                "column",
                "-a", "1000",
            ],
            stdout=f,
            stderr=f,
        )
    # fmt: on
    return str(directory)


def generate_cells_df():
    """Raw data for the cell collection."""
    x = [200] * 12
    y = [200] * 12
    z = [200] * 12
    df = pd.DataFrame(
        {
            "mtype": list(repeat("L2_TPC:A", DF_SIZE)),
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
    df.index += 1
    return df


def generate_cell_collection(cells_df):
    """The cell collection."""
    return CellCollection.from_dataframe(cells_df)


def input_cells_path(tmpdir):
    return tmpdir / "input_cells.mvd3"


def generate_input_cells(cell_collection, tmpdir):
    """The path to the MVD3 file containing the cell collection."""
    filename = input_cells_path(tmpdir)
    cell_collection.save_mvd3(filename)
    return filename


def generate_axon_morph_tsv(tmpdir):
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
    filename = tmpdir / "axon_morphs.tsv"
    df.to_csv(filename, sep="\t", na_rep="N/A")
    return filename


def get_tmd_parameters(filename):
    with open(filename, "r") as f:
        tmd_parameters = json.load(f)
    return tmd_parameters


def get_tmd_distributions(filename):
    with open(filename, "r") as f:
        tmd_distributions = json.load(f)
    return tmd_distributions


def get_cell_position():
    return [0, 500, 0]


def get_cell_mtype():
    return "L2_TPC:A"


def get_cell_orientation():
    return np.eye(3).reshape(1, 3, 3)