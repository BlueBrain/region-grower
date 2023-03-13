"""Tests for the region_grower.cli module."""
# pylint: disable=missing-function-docstring
from pathlib import Path

import dictdiffer
import pandas as pd

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
            [84],
            None,
            [19],
            None,
            [46],
            None,
            [48],
            None,
        ]
        assert debug_data["apical_NRN_sections"].tolist() == [
            [62],
            None,
            [5],
            None,
            [13],
            None,
            [22],
            None,
        ]

        cols = ["apical_sections", "apical_NRN_sections", "apical_points", "debug_infos"]
        debug_data.drop(columns=cols, inplace=True)
        expected_debug_data.drop(columns=cols, inplace=True)

        pd.testing.assert_frame_equal(debug_data, expected_debug_data, check_exact=False)
