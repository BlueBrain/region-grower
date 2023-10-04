"""Command Line Interface for the region_grower package."""
# pylint: disable=redefined-outer-name
import click

from region_grower import generate
from region_grower.synthesize_morphologies import SynthesizeMorphologies
from region_grower.utils import setup_logger


@click.group()
@click.version_option()
def main():
    """A tool for space synthesis management."""


@main.command(short_help="Generate the TMD parameter file")
@click.argument(
    "input_folder",
    type=str,
    required=True,
)
@click.argument(
    "dat_file",
    type=str,
    required=True,
)
@click.option(
    "-dc",
    "--diametrizer-config",
    type=click.File("r"),
    default=None,
    help="config file for diametrizer",
)
@click.option(
    "-tp",
    "--tmd-parameters",
    type=click.File("r"),
    default=None,
    help="tmd_parameter json if different from default generated by TMD",
)
@click.option(
    "-pf",
    "--parameter-filename",
    type=click.Path(file_okay=True, dir_okay=False),
    default="tmd_parameters.json",
    help="name of outputted .json file",
)
@click.option(
    "-e",
    "--ext",
    type=click.Choice([".h5", ".swc", ".asc"]),
    default=".h5",
    help="extension for neuron files",
)
def generate_parameters(
    input_folder,
    dat_file,
    parameter_filename,
    diametrizer_config,
    tmd_parameters,
    ext,
):
    """Generate JSON files containing the TMD parameters for each mtype in input_folder.

    INPUT_FOLDER is folder containing the cells.

    DAT_FILE is the .dat file with mtype for each cell.
    """
    generate.generate_parameters(
        input_folder,
        dat_file,
        parameter_filename,
        diametrizer_config,
        tmd_parameters,
        ext,
    )


@main.command(short_help="Create the TMD distribution file")
@click.argument(
    "input_folder",
    type=str,
    required=True,
)
@click.argument(
    "dat_file",
    type=str,
    required=True,
)
@click.option(
    "-df",
    "--distribution-filename",
    type=click.Path(file_okay=True, dir_okay=False),
    default="tmd_distributions.json",
    help="name of outputted .json file",
)
@click.option(
    "-dc",
    "--diametrizer-config",
    type=click.File("r"),
    default=None,
    help="config file for diametrizer",
)
@click.option(
    "-e",
    "--ext",
    type=click.Choice([".h5", ".swc", ".asc"]),
    default=".h5",
    help="extension for neuron files",
)
def generate_distributions(
    input_folder,
    dat_file,
    distribution_filename,
    diametrizer_config,
    ext,
):
    """Generate JSON files containing the TMD distributions for each mtype in input_folder.

    INPUT_FOLDER is folder containing the cells.

    DAT_FILE is the .dat file with mtype for each cell.
    """
    generate.generate_distributions(
        input_folder, dat_file, distribution_filename, diametrizer_config, ext
    )


@main.command(
    short_help=(
        "Synthesize morphologies into an given atlas according to the given TMD parameters and "
        "distributions."
    )
)
@click.option(
    "--input-cells",
    help=(
        "Path to a MVD3/sonata file storing cells collection whose positions are used as new soma "
        "locations"
    ),
    required=True,
)
@click.option("--tmd-parameters", help="Path to JSON with TMD parameters", required=True)
@click.option("--tmd-distributions", help="Path to JSON with TMD distributions", required=True)
@click.option("--morph-axon", help="TSV file with axon morphology list (for grafting)")
@click.option("--base-morph-dir", help="Path to base morphology release folder")
@click.option("--atlas", help="Atlas URL", required=True)
@click.option("--atlas-cache", help="Atlas cache folder")
@click.option("--seed", help="Random number generator seed (default: 0)", type=int, default=0)
@click.option("--out-cells", help="Path to output cells file.", required=True)
@click.option(
    "--out-apical",
    help=(
        "Path to output YAML apical file containing "
        "the coordinates where apical dendrites are tufting"
    ),
)
@click.option(
    "--out-apical-nrn-sections",
    help=(
        "Path to output YAML apical file containing"
        " the neuron section ids where apical dendrites"
        " are tufting"
    ),
)
@click.option("--out-morph-dir", help="Path to output morphology folder", default="out")
@click.option(
    "--out-morph-ext",
    help="Morphology export format(s)",
    type=click.Choice(["swc", "asc", "h5"]),
    multiple=True,
    default=["swc"],
)
@click.option(
    "--max-files-per-dir",
    help="Maximum files per level for morphology output folder",
    type=int,
)
@click.option(
    "--overwrite",
    help="Overwrite output morphology folder (default: False)",
    is_flag=True,
    default=False,
)
@click.option(
    "--max-drop-ratio",
    help="Max drop ratio for any mtype (default: 0)",
    type=float,
    default=0.0,
)
@click.option(
    "--scaling-jitter-std",
    help="Apply scaling jitter to all axon sections with the given std.",
    type=float,
)
@click.option(
    "--rotational-jitter-std",
    help="Apply rotational jitter to all axon sections with the given std.",
    type=float,
)
@click.option(
    "--out-debug-data",
    help="Export the debug data of each cell to this file.",
    type=str,
)
@click.option(
    "--nb-processes",
    help="Number of processes when MPI is not used.",
    type=int,
)
@click.option(
    "--with-mpi",
    help="Use MPI for parallel computation.",
    is_flag=True,
    default=False,
)
@click.option(
    "--log-level",
    help="The logger level.",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    default="info",
)
@click.option(
    "--min-depth",
    help="Minimum depth to use",
    type=float,
    default=25,
)
@click.option(
    "--skip-write",
    help="Skip writing of morphologies",
    is_flag=True,
    default=False,
)
@click.option(
    "--min-hard-scale",
    help="Minimum hard limit scale below which neurite is deleted",
    type=float,
    default=0.2,
)
@click.option(
    "--region-structure",
    help="Path to region structure file",
    default="region_structure.yaml",
)
@click.option(
    "--container-path",
    help="Path to container file of all morphologies (if None, not container created)",
    default=None,
)
@click.option(
    "--hide-progress-bar",
    help="Do not display the progress bar during the computation",
    is_flag=True,
    default=False,
)
def synthesize_morphologies(**kwargs):  # pylint: disable=too-many-arguments, too-many-locals
    """Synthesize morphologies."""
    setup_logger(kwargs.pop("log_level", "info"))

    SynthesizeMorphologies(**kwargs).synthesize()


if __name__ == "__main__":  # pragma: no cover
    main()
