"""Test the region_grower.cli module."""
from pathlib import Path
from tempfile import TemporaryDirectory

from click.testing import CliRunner
from nose.tools import assert_equal

from region_grower.cli import cli

DATA = Path(__file__).parent / "data"


def test_cli():
    """Test the CLI entries."""
    with TemporaryDirectory("test-report") as folder:
        folder = Path(folder)
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "generate-parameters",
                str(DATA / "input-cells"),
                str(DATA / "input-cells/neurondb.dat"),
                "-pf",
                str(folder / "parameters.json"),
            ],
        )

        assert_equal(result.exit_code, 0, result.exception)

        result = runner.invoke(
            cli,
            [
                "generate-parameters",
                str(DATA / "input-cells"),
                str(DATA / "input-cells/neurondb.dat"),
                "-dc",
                str(DATA / "diametrizer_config.json"),
                "-pf",
                str(folder / "parameters_external_diametrizer.json"),
            ],
        )

        assert_equal(result.exit_code, 0, result.exception)

        result = runner.invoke(
            cli,
            [
                "generate-parameters",
                str(DATA / "input-cells"),
                str(DATA / "input-cells/neurondb.dat"),
                "-tp",
                str(DATA / "tmd_parameters.json"),
                "-pf",
                str(folder / "parameters_tmd_parameters.json"),
            ],
        )

        assert_equal(result.exit_code, 0, result.exception)

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
                str(folder / "parameters_external_tmd_parameters.json"),
            ],
        )

        assert_equal(result.exit_code, 0, result.exception)

        result = runner.invoke(
            cli,
            [
                "generate-distributions",
                str(DATA / "input-cells"),
                str(DATA / "input-cells/neurondb.dat"),
                "-df",
                str(folder / "distributions.json"),
            ],
        )

        assert_equal(result.exit_code, 0, result.exception)

        result = runner.invoke(
            cli,
            [
                "generate-distributions",
                str(DATA / "input-cells"),
                str(DATA / "input-cells/neurondb.dat"),
                "-dc",
                str(DATA / "diametrizer_config.json"),
                "-df",
                str(folder / "distributions_external_diametrizer.json"),
            ],
        )

        assert_equal(result.exit_code, 0, result.exception)
