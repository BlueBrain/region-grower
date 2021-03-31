"""Test the region_grower.cli module."""
# pylint: disable=missing-function-docstring
# pylint: disable=no-self-use
from pathlib import Path

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
