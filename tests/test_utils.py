"""Test the region_grower.utils module."""
# pylint: disable=missing-function-docstring
# pylint: disable=no-self-use
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from voxcell import CellCollection

from region_grower import utils

DATA = Path(__file__).parent / "data"


def test_numpy_encoder():
    tested = {
        "regular types": {
            "string type": "string data",
            "list type": [1, 2],
            "int type": 1,
            "float type": 1.5,
        },
        "numpy types": {
            "string type": "string data",
            "list type": np.array([1, 2]),
            "int type": np.int32(1),
            "float type": np.float32(1.5),
        },
    }

    res = json.dumps(tested, cls=utils.NumpyEncoder)
    assert res == (
        '{"regular types": {"string type": "string data", "list type": [1, 2], "int type": 1, '
        '"float type": 1.5}, "numpy types": {"string type": "string data", "list type": [1, 2], '
        '"int type": 1, "float type": 1.5}}'
    )

    with pytest.raises(TypeError):
        json.dumps(tested)


def test_create_morphologies_dict():
    res = utils.create_morphologies_dict(DATA / "input-cells" / "neurondb.dat", "/test/path")
    assert list(res.items()) == [
        (
            "L2_TPC:A",
            [
                "/test/path/C170797A-P1.asc",
                "/test/path/C280199C-P3.asc",
                "/test/path/C280998A-P3.asc",
            ],
        )
    ]


def test_random_rotation_y():
    np.random.seed(0)
    assert_array_equal(utils.random_rotation_y(0), np.empty((0, 3, 3)))
    res_1 = utils.random_rotation_y(1)
    res_2 = utils.random_rotation_y(2)

    assert res_1.shape == (1, 3, 3)
    assert res_2.shape == (2, 3, 3)

    expected_res_1 = np.array(
        [
            [
                [0.9533337806844938, 0.0, 0.30191837076569117],
                [0.0, 1.0, 0.0],
                [-0.30191837076569117, 0.0, 0.9533337806844938],
            ]
        ]
    )
    assert_array_equal(res_1, expected_res_1)


class TestLoadMorphologyList:
    """Test the load_morphology_list function."""

    @pytest.mark.parametrize("with_scale", [True, False])
    @pytest.mark.parametrize("gids", [None, [], [0, 1, 2]])
    def test_default(self, tmpdir, with_scale, gids):
        filepath = tmpdir / "morphs.tsv"

        df = pd.DataFrame(
            {
                "morphology": [
                    "C170797A-P1",
                    "C280199C-P3",
                    np.nan,
                ],
            }
        )
        if with_scale:
            df["scale"] = [0.5, 1, 1.5]
        df.to_csv(filepath, sep="\t", na_rep="N/A")

        if gids != []:
            res = utils.load_morphology_list(filepath, gids)

            df.loc[df["morphology"].isnull(), "morphology"] = None
            if not with_scale:
                df["scale"] = None
            assert res.equals(df)
        else:
            with pytest.raises(RuntimeError, match="Morphology list GIDs mismatch"):
                utils.load_morphology_list(filepath, gids)


class TestCheckNaMorphologies:
    """Test the check_na_morphologies function."""

    @pytest.fixture
    def mtypes(self, cell_mtype):
        return pd.Series([cell_mtype] * 5)

    @pytest.mark.parametrize("threshold", [None, 0.25, 0.75])
    def test_default(self, mtypes, threshold, caplog):
        df = pd.DataFrame(
            {
                "morphology": [
                    "C170797A-P1",
                    "C280199C-P3",
                    None,
                    None,
                    "C280998A-P3",
                ],
            }
        )
        caplog.clear()
        caplog.set_level(logging.DEBUG)
        if threshold != 0.25:
            utils.check_na_morphologies(df, mtypes, threshold=threshold)
        else:
            with pytest.raises(
                RuntimeError,
                match=r"Max N/A ratio \(25.0%\) exceeded for mtype\(s\): L2_TPC:A",
            ):
                utils.check_na_morphologies(df, mtypes, threshold=threshold)
        assert caplog.record_tuples == [
            ("region_grower.utils", 30, "N/A morphologies for 2 position(s)"),
            (
                "region_grower.utils",
                20,
                (
                    "N/A ratio by mtypes:\n          N/A  out of  ratio, "
                    "%\nL2_TPC:A    2       5      40.0"
                ),
            ),
        ]

    @pytest.mark.parametrize("threshold", [None, 0.25, 0.75])
    def test_no_missing_value(self, mtypes, threshold, caplog):
        df = pd.DataFrame(
            {
                "morphology": [
                    "C170797A-P1",
                    "C280199C-P3",
                    "C280998A-P3",
                ],
            }
        )
        caplog.clear()
        caplog.set_level(logging.DEBUG)
        utils.check_na_morphologies(df, mtypes, threshold=threshold)
        assert caplog.record_tuples == []


class TestAssignMorphologies:
    """Test the assign_morphologies function."""

    @pytest.mark.parametrize("with_missing", [True, False])
    def test_assign_morphologies(self, with_missing):
        cells = CellCollection.from_dataframe(pd.DataFrame(index=[1, 2, 3]))
        morphs = {
            1: "a",
            2: "b",
        }
        if not with_missing:
            morphs[3] = "c"

        utils.assign_morphologies(cells, morphs)

        expected = pd.DataFrame({"morphology": ["a", "b", "c"]}, index=[1, 2, 3])

        if with_missing:
            cells.as_dataframe().equals(expected.drop(3))
        else:
            cells.as_dataframe().equals(expected.drop(3))