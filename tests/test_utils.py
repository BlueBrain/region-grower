"""Test the region_grower.utils module."""
# pylint: disable=missing-function-docstring
import json
import logging
import os
from copy import deepcopy
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from voxcell import CellCollection

from region_grower import utils
from region_grower.utils import MissingMpiError
from region_grower.utils import close_parallel_client
from region_grower.utils import initialize_parallel_client
from region_grower.utils import setup_logger

DATA = Path(__file__).parent / "data"


def test_numpy_encoder():
    """Test the JSON encoder for numpy objects."""
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
    """Test the create_morphologies_dict function."""
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
    """Test the random_rotation_y function."""
    np.random.seed(0)
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
    assert_almost_equal(res_1, expected_res_1)

    # Test with a given RNG
    rng = np.random.default_rng(0)
    res_1 = utils.random_rotation_y(1, rng=rng)
    res_2 = utils.random_rotation_y(2, rng=rng)

    assert res_1.shape == (1, 3, 3)
    assert res_2.shape == (2, 3, 3)

    expected_res_1 = np.array(
        [
            [
                [0.652016263584366, 0.0, 0.7582049802141123],
                [0.0, 1.0, 0.0],
                [-0.7582049802141123, 0.0, 0.652016263584366],
            ]
        ]
    )
    assert_almost_equal(res_1, expected_res_1)


class TestLoadMorphologyList:
    """Test the load_morphology_list function."""

    @pytest.mark.parametrize("with_scale", [True, False])
    @pytest.mark.parametrize("gids", [None, [], [0, 1, 2]])
    def test_default(self, tmpdir, with_scale, gids):
        """Test with and without scale and with several types of GIDs."""
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
        """The mtypes used in the tests."""
        return pd.Series([cell_mtype] * 5)

    @pytest.mark.parametrize("threshold", [None, 0.25, 0.75])
    def test_default(self, mtypes, threshold, caplog):
        """Test with several thresholds."""
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
        """Test without any missing value."""
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
        """Test with and without missing values."""
        cells = CellCollection.from_dataframe(pd.DataFrame(index=[1, 2, 3]))
        morphs = {
            0: "a",
            1: "b",
        }
        if not with_missing:
            morphs[2] = "c"

        utils.assign_morphologies(cells, morphs)

        expected = pd.DataFrame({"morphology": ["a", "b", "c"]}, index=[1, 2, 3])

        if with_missing:
            cells.as_dataframe().equals(expected.drop(3))
        else:
            cells.as_dataframe().equals(expected)


@pytest.mark.parametrize("with_SHMDIR", [True, False])
@pytest.mark.parametrize("with_TMPDIR", [True, False])
@pytest.mark.parametrize("with_dask_config", [True, False])
def test_dask_config(
    tmpdir,
    with_SHMDIR,
    with_TMPDIR,
    with_dask_config,
    monkeypatch,
):
    """Test morphology synthesis."""
    tmp_folder = Path(tmpdir)

    parallel_args = {"nb_processes": 2}

    custom_scratch_config = str(tmp_folder / "custom_scratch_config")
    custom_scratch_env_SHMDIR = str(tmp_folder / "custom_scratch_SHMDIR")
    custom_scratch_env_TMPDIR = str(tmp_folder / "custom_scratch_TMPDIR")
    dask_config = None
    if with_dask_config is not None:
        dask_config = {"temporary-directory": custom_scratch_config}
        parallel_args["dask_config"] = dask_config

    current_config = deepcopy(dask.config.config)
    with dask.config.set(current_config):
        if with_SHMDIR:
            monkeypatch.setenv("SHMDIR", custom_scratch_env_SHMDIR)
        else:
            monkeypatch.delenv("SHMDIR", raising=False)
        if with_TMPDIR:
            monkeypatch.setenv("TMPDIR", custom_scratch_env_TMPDIR)
        else:
            monkeypatch.delenv("TMPDIR", raising=False)

        parallel_client = initialize_parallel_client(**parallel_args)

        if dask_config is not None:
            assert dask.config.get("temporary-directory", None) == custom_scratch_config
        elif with_TMPDIR:
            assert dask.config.get("temporary-directory", None) == custom_scratch_env_TMPDIR
        elif with_SHMDIR:
            assert dask.config.get("temporary-directory", None) == custom_scratch_env_SHMDIR
        else:
            assert dask.config.get("temporary-directory", None) is None

        close_parallel_client(parallel_client)


def test_no_mpi():
    """Test client initialization with missing MPI libraries."""
    with pytest.raises(MissingMpiError):
        initialize_parallel_client(with_mpi=True)


@pytest.mark.parametrize("expected_nb_workers", [None, 0, 1, 2])
def test_nb_workers(expected_nb_workers):
    """Test client initialization with missing MPI libraries."""
    parallel_client, nb_workers = initialize_parallel_client(nb_processes=expected_nb_workers)
    try:
        if expected_nb_workers is None:
            assert nb_workers is None
        elif expected_nb_workers == 0:
            assert nb_workers == os.cpu_count()
        else:
            assert nb_workers == expected_nb_workers
    finally:
        close_parallel_client(parallel_client)


def test_setup_logger():
    """Test setup_logger with default params."""
    setup_logger()
