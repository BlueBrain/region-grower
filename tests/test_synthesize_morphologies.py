"""Test the region_grower.synthesize_morphologies module."""
# pylint: disable=missing-function-docstring
import logging
import shutil
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from uuid import uuid4

import jsonschema
import pandas as pd
import pytest
import yaml
from morph_tool.utils import iter_morphology_files
from numpy.testing import assert_allclose

from region_grower import RegionGrowerError
from region_grower.synthesize_morphologies import SynthesizeMorphologies

DATA = Path(__file__).parent / "data"


def check_yaml(ref_path, tested_path):
    assert ref_path.exists()
    assert tested_path.exists()
    with open(ref_path) as ref_file, open(tested_path) as tested_file:
        ref_obj = yaml.load(ref_file, Loader=yaml.FullLoader)
        tested_obj = yaml.load(tested_file, Loader=yaml.FullLoader)
    assert ref_obj.keys() == tested_obj.keys()
    for k in ref_obj.keys():
        assert_allclose(ref_obj[k], tested_obj[k])


def create_args(
    with_mpi,
    tmp_folder,
    input_cells,
    atlas_path,
    axon_morph_tsv,
    out_apical_NRN_sections,
):
    args = {}

    # Circuit
    args["cells_path"] = input_cells

    # Atlas
    args["atlas"] = atlas_path

    # Parameters
    args["tmd_distributions"] = DATA / "distributions.json"
    args["tmd_parameters"] = DATA / "parameters.json"
    args["seed"] = 0

    # Internals
    args["overwrite"] = True
    args["out_morph_ext"] = ["h5", "swc", "asc"]
    args["out_morph_dir"] = tmp_folder
    args["out_apical"] = tmp_folder / "apical.yaml"
    args["out_cells_path"] = str(tmp_folder / "test_cells.mvd3")
    if out_apical_NRN_sections:
        args["out_apical_nrn_sections"] = tmp_folder / out_apical_NRN_sections
    else:
        args["out_apical_nrn_sections"] = None
    if with_mpi:
        args["with_mpi"] = with_mpi
    else:
        args["nb_processes"] = 2

    # Axons
    args["base_morph_dir"] = str(DATA / "input-cells")
    args["morph_axon"] = axon_morph_tsv
    args["max_drop_ratio"] = 0.5
    args["rotational_jitter_std"] = 10
    args["scaling_jitter_std"] = 0.5

    return args


@pytest.mark.parametrize("with_axon", [True, False])
@pytest.mark.parametrize("with_NRN", [True, False])
def test_synthesize(
    tmpdir, small_O1_path, input_cells, axon_morph_tsv, with_axon, with_NRN
):  # pylint: disable=unused-argument
    tmp_folder = Path(tmpdir)

    args = create_args(
        False,
        tmp_folder,
        input_cells,
        small_O1_path,
        axon_morph_tsv if with_axon else None,
        "apical_NRN_sections.yaml" if with_NRN else None,
    )

    synthesizer = SynthesizeMorphologies(**args)
    synthesizer.synthesize()

    # Check results
    if with_axon:
        expected_size = 12
    else:
        expected_size = 36
    assert len(list(iter_morphology_files(tmp_folder))) == expected_size

    if with_axon:
        apical_suffix = ""
    else:
        apical_suffix = "_no_axon"
    check_yaml(DATA / ("apical" + apical_suffix + ".yaml"), args["out_apical"])
    if with_NRN:
        check_yaml(
            DATA / ("apical_NRN_sections" + apical_suffix + ".yaml"),
            args["out_apical_nrn_sections"],
        )


def run_with_mpi():
    # pylint: disable=import-outside-toplevel, too-many-locals, import-error
    from data_factories import generate_axon_morph_tsv
    from data_factories import generate_cell_collection
    from data_factories import generate_cells_df
    from data_factories import generate_input_cells
    from data_factories import generate_small_O1
    from data_factories import input_cells_path
    from mpi4py import MPI

    from region_grower.utils import setup_logger

    COMM = MPI.COMM_WORLD  # pylint: disable=c-extension-no-member
    rank = COMM.Get_rank()
    MASTER_RANK = 0
    is_master = rank == MASTER_RANK

    tmp_folder = Path("/tmp/test-run-synthesis_" + str(uuid4()))
    tmp_folder = COMM.bcast(tmp_folder, root=MASTER_RANK)
    input_cells = input_cells_path(tmp_folder)
    small_O1_path = str(tmp_folder / "atlas")

    args = create_args(
        True,
        tmp_folder,
        input_cells,
        small_O1_path,
        tmp_folder / "axon_morphs.tsv",
        "apical_NRN_sections.yaml",
    )

    setup_logger("info")

    if is_master:
        tmp_folder.mkdir(exist_ok=True)
        print(f"============= #{rank}: Create data")
        cells_raw_data = generate_cells_df()
        cell_collection = generate_cell_collection(cells_raw_data)
        generate_input_cells(cell_collection, tmp_folder)
        generate_small_O1(small_O1_path)
        generate_axon_morph_tsv(tmp_folder)

    synthesizer = SynthesizeMorphologies(**args)
    try:
        print(f"============= #{rank}: Start synthesize")
        synthesizer.synthesize()

        # Check results
        print(f"============= #{rank}: Checking results")
        expected_size = 12
        assert len(list(iter_morphology_files(tmp_folder))) == expected_size

        check_yaml(DATA / ("apical.yaml"), args["out_apical"])
        check_yaml(
            DATA / ("apical_NRN_sections.yaml"),
            args["out_apical_nrn_sections"],
        )
    finally:
        # Clean the directory
        print(f"============= #{rank}: Cleaning")
        shutil.rmtree(tmp_folder)


def test_verify(tmd_distributions, tmd_parameters):
    mtype = "L2_TPC:A"
    initial_params = deepcopy(tmd_parameters)

    SynthesizeMorphologies.verify([mtype], tmd_distributions, tmd_parameters)
    with pytest.raises(RegionGrowerError):
        SynthesizeMorphologies.verify(["UNKNOWN_MTYPE"], tmd_distributions, tmd_parameters)

    failing_params = deepcopy(initial_params)

    del failing_params[mtype]
    with pytest.raises(RegionGrowerError):
        SynthesizeMorphologies.verify([mtype], tmd_distributions, failing_params)

    failing_params = deepcopy(initial_params)
    del failing_params[mtype]["origin"]
    with pytest.raises(jsonschema.exceptions.ValidationError):
        SynthesizeMorphologies.verify([mtype], tmd_distributions, failing_params)

    # Fail when missing attributes
    attributes = ["layer", "fraction", "slope", "intercept"]
    good_params = deepcopy(initial_params)
    good_params[mtype]["context_constraints"] = {
        "apical": {
            "hard_limit_min": {
                "layer": 1,
                "fraction": 0.1,
            },
            "extent_to_target": {
                "slope": 0.5,
                "intercept": 1,
                "layer": 1,
                "fraction": 0.5,
            },
            "hard_limit_max": {
                "layer": 1,
                "fraction": 0.9,
            },
        }
    }
    tmd_parameters = deepcopy(good_params)
    SynthesizeMorphologies.verify([mtype], tmd_distributions, tmd_parameters)
    for i in range(1, 5):
        for missing_attributes in combinations(attributes, i):
            failing_params = deepcopy(good_params[mtype])
            for att in missing_attributes:
                del failing_params["context_constraints"]["apical"]["extent_to_target"][att]
            tmd_parameters[mtype] = failing_params
            with pytest.raises(jsonschema.exceptions.ValidationError):
                SynthesizeMorphologies.verify([mtype], tmd_distributions, tmd_parameters)


def test_check_axon_morphology(caplog):
    # pylint: disable=protected-access
    # Test with no missing name
    caplog.set_level(logging.WARNING)
    caplog.clear()
    df = pd.DataFrame({"axon_name": ["a", "b", "c"], "other_col": [1, 2, 3]})

    assert SynthesizeMorphologies._check_axon_morphology(df) is None
    assert caplog.record_tuples == []

    # Test with no missing names
    caplog.clear()
    df = pd.DataFrame({"axon_name": ["a", None, "c"], "other_col": [1, 2, 3]})

    assert SynthesizeMorphologies._check_axon_morphology(df).tolist() == [0, 2]
    assert caplog.record_tuples == [
        (
            "region_grower.synthesize_morphologies",
            30,
            "The following gids were not found in the axon morphology list: [1]",
        ),
    ]


if __name__ == "__main__":  # pragma: no cover
    run_with_mpi()
