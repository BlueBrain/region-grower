"""Generate atlas for tests"""
# pylint: disable=missing-function-docstring
from itertools import cycle
from itertools import islice
from itertools import repeat
from os import devnull
from subprocess import call

import pandas as pd
from mock import MagicMock


def small_O1(folder_path):
    """Dump a small O1 atlas in folder path"""
    # fmt: off
    with open(devnull, "w") as f:
        call(
            [
                "brainbuilder", "atlases",
                "-n", "6,5,4,3,2,1",
                "-t", "200,100,100,100,100,200",
                "-d", "100",
                "-o", str(folder_path),
                "column",
                "-a", "1000",
            ],
            stdout=f,
            stderr=f,
        )
    # fmt: on


def small_O1_placement_algo(folder_path):
    """Dump a small O1 atlas in folder path"""

    # fmt: off
    call(
        [
            "brainbuilder", "atlases",
            "-n", "1,2,3,4,5,6",
            "-t", "200,100,100,100,100,200",
            "-d", "100",
            "-o", str(folder_path),
            "column",
            "-a", "1000",
        ]
    )
    # fmt: on


class CellCollectionMock(MagicMock):
    """Mock a CellCollection"""

    size = 12

    def load_mvd3(self):
        pass

    @property
    def positions(self):
        mock = MagicMock()
        mock.__getitem__ = MagicMock(return_value=[200, 200, 200])
        return mock

    @property
    def properties(self):
        return pd.DataFrame(
            {
                "mtype": list(repeat("L2_TPC:A", self.size)),
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
                        self.size,
                    )
                ),
            }
        )
