"""Test the region_grower.cli module."""
# pylint: disable=missing-function-docstring
# pylint: disable=no-self-use
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from region_grower.cli import cli

DATA = Path(__file__).parent / "data"


class TestCli:
    """Test the CLI entries."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_generate_parameters(self, tmpdir, runner):
        result = runner.invoke(
            cli,
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

    def test_generate_parameters_external_diametrizer(self, tmpdir, runner):
        result = runner.invoke(
            cli,
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

    def test_generate_parameters_tmd(self, tmpdir, runner):
        result = runner.invoke(
            cli,
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

    def test_generate_parameters_external_tmd(self, tmpdir, runner):
        result = runner.invoke(
            cli,
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

    def test_generate_distributions(self, tmpdir, runner):
        result = runner.invoke(
            cli,
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

    def test_generate_distributions_external_diametrizer(self, tmpdir, runner):
        result = runner.invoke(
            cli,
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
        self, tmpdir, runner, small_O1_path, input_cells, axon_morph_tsv
    ):
        # fmt: off
        result = runner.invoke(
            cli,
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
                "--out-debug-data", str(tmpdir / "debug_data.csv"),
                "--overwrite",
                "--out-morph-ext", "h5",
                "--out-morph-ext", "swc",
                "--out-morph-ext", "asc",
                "--max-drop-ratio", 0.5,
                "--scaling-jitter-std", 0.5,
                "--rotational-jitter-std", 10,
                "--nb-processes", 2,
            ],
            catch_exceptions=False,
        )
        # fmt: on

        assert result.exit_code == 0
        assert result.exception is None
        assert Path(tmpdir / "test_cells.mvd3").exists()
        assert Path(tmpdir / "apical.yaml").exists()
        assert Path(tmpdir / "debug_data.csv").exists()

        expected_debug_data = pd.read_csv(DATA / "debug_data.csv")
        debug_data = pd.read_csv(tmpdir / "debug_data.csv")
        pd.testing.assert_frame_equal(debug_data, expected_debug_data, check_exact=False)
