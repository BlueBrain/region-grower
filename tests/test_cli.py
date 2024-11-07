"""Tests for the region_grower.cli module."""

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
from pathlib import Path

import dask
import dictdiffer
import pandas as pd
import pytest
import yaml
from voxcell import CellCollection

from region_grower.cli import main

DATA = Path(__file__).parent / "data"


class TestCli:
    """Test the CLI entries."""

    def test_generate_parameters(self, tmpdir, cli_runner):
        """Generate the parameters."""
        result = cli_runner.invoke(
            main,
            [
                "generate-parameters",
                str(DATA / "input-cells"),
                str(DATA / "input-cells/neurondb.dat"),
                "-pf",
                str(tmpdir / "parameters.json"),
            ],
        )

        assert result.exit_code == 0
        assert result.exception is None
        assert Path(tmpdir / "parameters.json").exists()

    def test_generate_parameters_external_diametrizer(self, tmpdir, cli_runner):
        """Generate the parameters with an external diametrizer."""
        result = cli_runner.invoke(
            main,
            [
                "generate-parameters",
                str(DATA / "input-cells"),
                str(DATA / "input-cells/neurondb.dat"),
                "-dc",
                str(DATA / "diametrizer_config.json"),
                "-pf",
                str(tmpdir / "parameters_external_diametrizer.json"),
            ],
        )

        assert result.exit_code == 0
        assert result.exception is None
        assert Path(tmpdir / "parameters_external_diametrizer.json").exists()

    def test_generate_parameters_tmd(self, tmpdir, cli_runner):
        """Generate the parameters with TMD parameters."""
        result = cli_runner.invoke(
            main,
            [
                "generate-parameters",
                str(DATA / "input-cells"),
                str(DATA / "input-cells/neurondb.dat"),
                "-tp",
                str(DATA / "tmd_parameters.json"),
                "-pf",
                str(tmpdir / "parameters_tmd_parameters.json"),
            ],
        )

        assert result.exit_code == 0
        assert result.exception is None
        assert Path(tmpdir / "parameters_tmd_parameters.json").exists()

    def test_generate_parameters_external_tmd(self, tmpdir, cli_runner):
        """Generate the parameters with both an external diametrizer and TMD parameters."""
        result = cli_runner.invoke(
            main,
            [
                "generate-parameters",
                str(DATA / "input-cells"),
                str(DATA / "input-cells/neurondb.dat"),
                "-tp",
                str(DATA / "tmd_parameters.json"),
                "-dc",
                str(DATA / "diametrizer_config.json"),
                "-pf",
                str(tmpdir / "parameters_external_tmd_parameters.json"),
            ],
        )

        assert result.exit_code == 0
        assert result.exception is None
        assert Path(tmpdir / "parameters_external_tmd_parameters.json").exists()

    def test_generate_distributions(self, tmpdir, cli_runner):
        """Generate the distributions."""
        result = cli_runner.invoke(
            main,
            [
                "generate-distributions",
                str(DATA / "input-cells"),
                str(DATA / "input-cells/neurondb.dat"),
                "-df",
                str(tmpdir / "distributions.json"),
            ],
        )

        assert result.exit_code == 0
        assert result.exception is None
        assert Path(tmpdir / "distributions.json").exists()

    def test_generate_distributions_external_diametrizer(self, tmpdir, cli_runner):
        """Generate the distributions with an external diametrizer."""
        result = cli_runner.invoke(
            main,
            [
                "generate-distributions",
                str(DATA / "input-cells"),
                str(DATA / "input-cells/neurondb.dat"),
                "-dc",
                str(DATA / "diametrizer_config.json"),
                "-df",
                str(tmpdir / "distributions_external_diametrizer.json"),
            ],
        )

        assert result.exit_code == 0
        assert result.exception is None
        assert Path(tmpdir / "distributions_external_diametrizer.json").exists()

    def test_synthesize_morphologies(
        self, tmpdir, cli_runner, small_O1_path, input_cells, axon_morph_tsv
    ):
        """Synthesize the morphologies."""
        # fmt: off
        result = cli_runner.invoke(
            main,
            [
                "synthesize-morphologies",
                "--input-cells", str(input_cells),
                "--tmd-parameters", str(DATA / "parameters.json"),
                "--tmd-distributions", str(DATA / "distributions.json"),
                "--morph-axon", str(axon_morph_tsv),
                "--base-morph-dir", str(DATA / "input-cells"),
                "--atlas", str(small_O1_path),
                "--seed", 0,
                "--out-cells", str(tmpdir / "test_cells.mvd3"),
                "--out-apical", str(tmpdir / "apical.yaml"),
                "--out-apical-nrn-sections", str(tmpdir / "apical_NRN_sections.yaml"),
                "--out-morph-dir", str(tmpdir),
                "--out-debug-data", str(tmpdir / "debug_data.pkl"),
                "--overwrite",
                "--out-morph-ext", "h5",
                "--out-morph-ext", "swc",
                "--out-morph-ext", "asc",
                "--max-drop-ratio", 0.5,
                "--scaling-jitter-std", 0.5,
                "--rotational-jitter-std", 10,
                "--nb-processes", 2,
                "--region-structure", str(DATA / "region_structure.yaml"),
            ],
            catch_exceptions=False,
        )
        # fmt: on

        assert result.exit_code == 0
        assert result.exception is None
        assert Path(tmpdir / "test_cells.mvd3").exists()
        assert Path(tmpdir / "apical.yaml").exists()
        assert Path(tmpdir / "debug_data.pkl").exists()

        expected_debug_data = pd.read_pickle(DATA / "debug_data.pkl")
        debug_data = pd.read_pickle(tmpdir / "debug_data.pkl")

        equal_infos = (
            expected_debug_data["debug_infos"]
            .to_frame()
            .join(debug_data["debug_infos"], lsuffix="_a", rsuffix="_b")
            .apply(
                lambda row: not list(dictdiffer.diff(row["debug_infos_a"], row["debug_infos_b"])),
                axis=1,
            )
        )
        assert equal_infos.all()
        assert debug_data["apical_sections"].tolist() == [
            [72],
            None,
            [23],
            None,
            [42],
            None,
            [38],
            None,
            [67],
            [74],
        ]
        assert debug_data["apical_NRN_sections"].tolist() == [
            [50],
            None,
            [9],
            None,
            [9],
            None,
            [12],
            None,
            [35],
            [20],
        ]

        cols = ["apical_sections", "apical_NRN_sections", "apical_points", "debug_infos"]
        debug_data.drop(columns=cols, inplace=True)
        expected_debug_data.drop(columns=cols, inplace=True)

        pd.testing.assert_frame_equal(debug_data, expected_debug_data, check_exact=False)

    def test_dask_config(self, tmpdir, cli_runner, small_O1_path, input_cells):
        """Test the dask config update when using the 'synthesize-morphologies' command."""
        # Reduce the number of cells
        cell_collection = CellCollection.load_mvd3(input_cells)
        cells_df = cell_collection.as_dataframe()
        cell_collection = CellCollection.from_dataframe(cells_df.drop(cells_df.index[4:]))
        cell_collection.save_mvd3(input_cells)

        # fmt: off
        params = [
            "synthesize-morphologies",
            "--input-cells", str(input_cells),
            "--tmd-parameters", str(DATA / "parameters.json"),
            "--tmd-distributions", str(DATA / "distributions.json"),
            "--base-morph-dir", str(DATA / "input-cells"),
            "--atlas", str(small_O1_path),
            "--seed", 0,
            "--out-cells", str(tmpdir / "test_cells.mvd3"),
            "--out-apical", str(tmpdir / "apical.yaml"),
            "--out-apical-nrn-sections", str(tmpdir / "apical_NRN_sections.yaml"),
            "--out-morph-dir", str(tmpdir),
            "--out-debug-data", str(tmpdir / "debug_data.pkl"),
            "--overwrite",
            "--out-morph-ext", "h5",
            "--out-morph-ext", "swc",
            "--out-morph-ext", "asc",
            "--max-drop-ratio", 0.5,
            "--scaling-jitter-std", 0.5,
            "--rotational-jitter-std", 10,
            "--nb-processes", 2,
            "--region-structure", str(DATA / "region_structure.yaml"),
            "--log-level", "debug",
            "--show-pip-freeze",
        ]
        # fmt: on

        # Test with JSON string
        cli_runner.invoke(
            main,
            params
            + [
                "--dask-config",
                json.dumps({"temporary-directory": str(tmpdir / "custom_scratch_1")}),
            ],
            catch_exceptions=False,
        )

        assert Path(tmpdir / "custom_scratch_1").exists()

        # Test with path to YAML file
        dask.config.refresh()
        dask_config_path = Path(tmpdir) / "dask_config.yaml"
        with dask_config_path.open("w", encoding="utf-8") as file:
            yaml.dump({"temporary-directory": str(tmpdir / "custom_scratch_2")}, file)
        cli_runner.invoke(
            main,
            params + ["--dask-config", dask_config_path],
            catch_exceptions=False,
        )

        assert Path(tmpdir / "custom_scratch_2").exists()

        # Test temporary-directory using $SHMDIR
        dask.config.defaults = [dask.config.defaults[0]]
        dask.config.refresh()
        cli_runner.invoke(
            main,
            params,
            env={"SHMDIR": str(tmpdir / "custom_scratch_3")},
            catch_exceptions=False,
        )

        assert Path(tmpdir / "custom_scratch_3").exists()

        # Test temporary-directory using $SHMDIR
        dask.config.defaults = [dask.config.defaults[0]]
        dask.config.refresh()
        cli_runner.invoke(
            main,
            params,
            env={"TMPDIR": str(tmpdir / "custom_scratch_4")},
            catch_exceptions=False,
        )

        assert Path(tmpdir / "custom_scratch_4").exists()

        # Test invalid dask-config parameter
        with pytest.raises(
            ValueError,
            match=(
                r"The value for the --dask-config parameter is not an existing file path and could "
                r"not be parsed as a JSON string"
            ),
        ):
            cli_runner.invoke(
                main,
                params + ["--dask-config", "INVALID PARAMETER"],
                catch_exceptions=False,
            )

        # Test invalid chunksize parameter
        result = cli_runner.invoke(
            main,
            params + ["--chunksize", "-1"],
            catch_exceptions=False,
        )
        assert result.exit_code == 2
        assert (
            "Error: Invalid value for '--chunksize': -1 is not in the range x>=1" in result.output
        )

    def test_entry_point(self, script_runner):
        """Test the entry point."""
        ret = script_runner.run("region-grower", "--version")
        assert ret.success
        assert ret.stdout.startswith("region-grower, version ")
        assert ret.stderr == ""
