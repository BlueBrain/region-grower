"""Functions fo generate parameters and distributions."""
import json
import logging
from functools import partial
import multiprocessing
import yaml
import pkg_resources

from tqdm import tqdm

from tns import extract_input
from diameter_synthesis import build_models
from region_grower.utils import NumpyEncoder, create_morphologies_dict

L = logging.getLogger(__name__)
PC_IN_TYPES_FILE = pkg_resources.resource_filename(
    "region_grower", "data/pc_in_types.yaml"
)


def generate_parameters(
    input_folder, dat_file, parameter_filename, diametrizer_config, tmd_parameters, ext,
):
    """Generate JSON files containing the TMD parameters for
    each mtype in input_folder, using dat_file for mtypes
    """

    L.info("Extracting TMD parameters for each mtype...")
    morphologies_dict = create_morphologies_dict(dat_file, input_folder, ext=ext)

    with open(PC_IN_TYPES_FILE, "rb") as pc_in_file:
        pc_in_files = yaml.full_load(pc_in_file)

    neurite_types = {
        mtype: ["basal"] if pc_in_files[mtype] == "IN" else ["basal", "apical"]
        for mtype in morphologies_dict
    }

    if diametrizer_config is not None:
        diametrizer_config = json.load(diametrizer_config)

    if tmd_parameters is not None:
        tmd_parameters = json.load(tmd_parameters)

    def get_parameters(mtype):
        """allow for precomputed tmd_parameter to be stacked with diameter parameters"""
        if tmd_parameters is None:
            return extract_input.parameters(
                neurite_types=neurite_types[mtype],
                diameter_parameters=diametrizer_config,
            )
        if tmd_parameters is not None and diametrizer_config is not None:
            try:
                parameters = tmd_parameters[mtype]
            except KeyError:
                L.error("%s is not in the given tmd_parameter.json", mtype)
                parameters = {}
            parameters["diameter_params"] = diametrizer_config
            parameters["diameter_params"]["method"] = "external"
            return parameters
        return tmd_parameters[mtype]

    parameters = {
        mtype: get_parameters(mtype) for mtype in tqdm(morphologies_dict.keys())
    }

    with open(parameter_filename, "w") as f:
        json.dump(parameters, f, cls=NumpyEncoder, indent=4)


class Worker:
    """Worker to get distributions."""

    def __init__(self, neurite_types, diameter_model_function):
        """Set needed data."""
        self.neurite_types = neurite_types
        self.diameter_model_function = diameter_model_function

    def __call__(self, morphology_item):
        """Generate distributions for a given mytpe."""
        return (
            morphology_item[0],
            extract_input.distributions(
                morphology_item[1],
                neurite_types=self.neurite_types[morphology_item[0]],
                diameter_input_morph=morphology_item[1],
                diameter_model=self.diameter_model_function,
            ),
        )


def generate_distributions(
    input_folder, dat_file, distribution_filename, diametrizer_config, ext,
):
    """Generate JSON files containing the TMD distributions for
    each mtype in input_folder, using dat_file for mtypes.
    """

    L.info("Extracting TMD distributions for each mtype.\n" "This can take a while...")

    morphologies_dict = create_morphologies_dict(dat_file, input_folder, ext=ext)

    with open(PC_IN_TYPES_FILE, "rb") as pc_in_file:
        pc_in_files = yaml.full_load(pc_in_file)

    neurite_types = {
        mtype: ["basal"] if pc_in_files[mtype] == "IN" else ["basal", "apical"]
        for mtype in morphologies_dict
    }

    config = None
    diameter_model_function = None
    if diametrizer_config is not None:
        config = json.load(diametrizer_config)
        diameter_model_function = partial(build_models.build, config=config)

    L.info("Extracting TMD distributions for each mtype...")

    results = multiprocessing.Pool().imap_unordered(
        Worker(neurite_types, diameter_model_function), morphologies_dict.items(),
    )

    distributions = {
        "mtypes": dict(tqdm(results, total=len(morphologies_dict))),
        "metadata": {"cortical_thickness": [165, 149, 353, 190, 525, 700]},
    }

    with open(distribution_filename, "w") as f:
        json.dump(distributions, f, cls=NumpyEncoder, indent=4)
