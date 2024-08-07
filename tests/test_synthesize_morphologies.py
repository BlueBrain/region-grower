"""Test the region_grower.synthesize_morphologies module."""
# pylint: disable=missing-function-docstring
import json
import logging
import os
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import attr
import jsonschema
import neurots
import pandas as pd
import pytest
import yaml
from morph_tool.utils import iter_morphology_files
from morphio import Morphology
from numpy.testing import assert_allclose
from voxcell import CellCollection
from voxcell import RegionMap

from region_grower import RegionGrowerError
from region_grower.synthesize_morphologies import RegionMapper
from region_grower.synthesize_morphologies import SynthesizeMorphologies
from region_grower.utils import close_parallel_client
from region_grower.utils import initialize_parallel_client

DATA = Path(__file__).parent / "data"


def check_yaml(ref_path, tested_path):
    """Compare a YAML file to a reference file."""
    print(f"Check YAML:\n\tref: {ref_path}\n\ttested: {tested_path}")
    assert ref_path.exists()
    assert tested_path.exists()
    with open(ref_path, encoding="utf-8") as ref_file, open(
        tested_path, encoding="utf-8"
    ) as tested_file:
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
    min_depth,
):
    """Create the arguments used for tests."""
    args = {}
    parallel_args = {}

    # Circuit
    args["input_cells"] = input_cells

    # Atlas
    args["atlas"] = atlas_path

    # Parameters
    args["tmd_distributions"] = DATA / "distributions.json"
    args["tmd_parameters"] = DATA / "parameters.json"
    args["seed"] = 0
    args["min_depth"] = min_depth

    # Internals
    args["overwrite"] = True
    args["out_morph_ext"] = ["h5", "swc", "asc"]
    args["out_morph_dir"] = tmp_folder
    args["out_apical"] = tmp_folder / "apical.yaml"
    args["out_cells"] = str(tmp_folder / "test_cells.mvd3")
    if out_apical_NRN_sections:
        args["out_apical_nrn_sections"] = tmp_folder / out_apical_NRN_sections
        parallel_args["no_daemon"] = True
    else:
        args["out_apical_nrn_sections"] = None
        parallel_args["no_daemon"] = False
    if with_mpi:
        parallel_args["with_mpi"] = with_mpi
    else:
        parallel_args["nb_processes"] = 2

    # Axons
    args["base_morph_dir"] = str(DATA / "input-cells")
    args["morph_axon"] = axon_morph_tsv
    args["max_drop_ratio"] = 0.5
    args["rotational_jitter_std"] = 10
    args["scaling_jitter_std"] = 0.5
    args["region_structure"] = DATA / "region_structure.yaml"

    return args, parallel_args


@pytest.mark.parametrize("min_depth", [25, 800])
@pytest.mark.parametrize("with_axon", [True, False])
@pytest.mark.parametrize("with_NRN", [True, False])
def test_synthesize(
    tmpdir,
    small_O1_path,
    input_cells,
    axon_morph_tsv,
    with_axon,
    with_NRN,
    min_depth,
):  # pylint: disable=unused-argument
    """Test morphology synthesis."""
    tmp_folder = Path(tmpdir)

    args, parallel_args = create_args(
        False,
        tmp_folder,
        input_cells,
        small_O1_path,
        axon_morph_tsv if with_axon else None,
        "apical_NRN_sections.yaml" if with_NRN else None,
        min_depth,
    )

    parallel_client, args["nb_workers"] = initialize_parallel_client(**parallel_args)
    synthesizer = SynthesizeMorphologies(**args)
    synthesizer.synthesize()

    # Check results
    if with_axon:
        expected_size = 18
    else:
        expected_size = 42

    assert len(list(iter_morphology_files(tmp_folder))) == expected_size

    if with_axon:
        apical_suffix = ""
    else:
        apical_suffix = "_no_axon"

    # pylint: disable=unsubscriptable-object
    max_y = Morphology(sorted(iter_morphology_files(tmp_folder))[0]).points[:, 1].max()
    if min_depth == 25:
        check_yaml(DATA / ("apical" + apical_suffix + ".yaml"), args["out_apical"])
        if with_NRN:
            check_yaml(
                DATA / ("apical_NRN_sections" + apical_suffix + ".yaml"),
                args["out_apical_nrn_sections"],
            )
        if with_NRN and with_axon:
            assert_allclose(max_y, 168.1047)
    else:
        if with_NRN and with_axon:
            assert_allclose(max_y, 149.30412)

    close_parallel_client(parallel_client)


def test_synthesize_empty_population(
    tmp_path,
    small_O1_path,
    input_cells,
):
    """Test morphology synthesis."""
    args, parallel_args = create_args(
        False,
        tmp_path,
        input_cells,
        small_O1_path,
        None,
        None,
        25,
    )
    # Update population to make it empty
    cells = CellCollection.load(args["input_cells"])
    cells_df = cells.as_dataframe()
    empty_cells = CellCollection.from_dataframe(pd.DataFrame().reindex_like(cells_df.iloc[:0, :]))
    empty_cells.save(args["input_cells"])

    parallel_client, args["nb_workers"] = initialize_parallel_client(**parallel_args)
    synthesizer = SynthesizeMorphologies(**args)
    synthesizer.synthesize()

    # Check results
    assert len(list(iter_morphology_files(tmp_path))) == 0
    assert Path(args["out_cells"]).exists()

    close_parallel_client(parallel_client)


@pytest.mark.parametrize("nb_processes", [0, 2, None])
@pytest.mark.parametrize("chunksize", [1, 5, 999])
def test_synthesize_skip_write(
    tmpdir,
    small_O1_path,
    input_cells,
    axon_morph_tsv,
    nb_processes,
    chunksize,
):  # pylint: disable=unused-argument
    """Test morphology synthesis but skip write step."""
    tmp_folder = Path(tmpdir)

    args, parallel_args = create_args(
        False,
        tmp_folder,
        input_cells,
        small_O1_path,
        axon_morph_tsv,
        "apical_NRN_sections.yaml",
        min_depth=25,
    )
    args["skip_write"] = True
    args["hide_progress_bar"] = True
    args["out_apical"] = None
    args["chunksize"] = chunksize
    args["skip_checks"] = True
    parallel_args["nb_processes"] = nb_processes

    print("Number of available CPUs", os.cpu_count())

    parallel_client, args["nb_workers"] = initialize_parallel_client(**parallel_args)
    synthesizer = SynthesizeMorphologies(**args)
    res = synthesizer.synthesize()

    assert (res["x"] == 200).all()
    assert (res["y"] == 200).all()
    assert (res["z"] == 200).all()

    assert list(res["name"]) == [
        "e3e70682c2094cac629f6fbed82c07cd",
        None,
        "216363698b529b4a97b750923ceb3ffd",
        None,
        "14a03569d26b949692e5dfe8cb1855fe",
        None,
        "4462ebfc5f915ef09cfbac6e7687a66e",
        None,
        "87751d4ca8501e2c44dcda6a797d76de",
        "e8d79f49af6d114c4a6f188a424e617b",
    ]
    assert [[list(i[0])] if i else i for i in list(res["apical_points"])] == [
        [[-5.780806541442871, 164.6251678466797, 0.9229676723480225]],
        None,
        [[-4.380718231201172, 116.65441131591797, 6.396718978881836]],
        None,
        [[5.5464959144592285, 113.9891357421875, -2.365058422088623]],
        None,
        [[-44.474491119384766, 68.06861877441406, -7.087465286254883]],
        None,
        [[4.237551689147949, 369.3919677734375, -0.3368062376976013]],
        [[-28.59885025024414, 215.1460723876953, -12.506669998168945]],
    ]

    # Check that the morphologies were not written
    res_files = tmpdir.listdir()
    assert len(res_files) == 4
    assert sorted(i.basename for i in res_files) == [
        "apical_NRN_sections.yaml",
        "axon_morphs.tsv",
        "input_cells.mvd3",
        "test_cells.mvd3",
    ]

    close_parallel_client(parallel_client)


def test_verify(cell_collection, tmd_distributions, tmd_parameters, small_O1_path):
    """Test the `verify` step."""
    mtype = "L2_TPC:A"

    @attr.s(auto_attribs=True)
    class Data:
        """Container to mimic SynthesizeMorphologies class."""

        tmd_distributions: dict
        tmd_parameters: dict
        cells: CellCollection
        region_mapper: dict
        skip_checks: bool = False

    region_mapper = RegionMapper(
        ["test"], RegionMap.load_json(Path(small_O1_path) / "hierarchy.json")
    )
    data = Data(
        tmd_distributions=tmd_distributions,
        tmd_parameters=tmd_parameters,
        cells=cell_collection,
        region_mapper=region_mapper,
    )
    SynthesizeMorphologies.verify(data)

    cell_collection.properties.loc[0, "mtype"] = "UNKNOWN_MTYPE"
    data = Data(
        tmd_distributions=tmd_distributions,
        tmd_parameters=tmd_parameters,
        cells=cell_collection,
        region_mapper=region_mapper,
    )
    with pytest.raises(
        RegionGrowerError,
        match="Missing distributions for mtype 'UNKNOWN_MTYPE' in region 'default'",
    ):
        SynthesizeMorphologies.verify(data)

    cell_collection.properties.loc[0, "mtype"] = "L2_TPC:A"

    failing_params = deepcopy(tmd_parameters)
    del failing_params["default"][mtype]
    data = Data(
        tmd_distributions=tmd_distributions,
        tmd_parameters=failing_params,
        cells=cell_collection,
        region_mapper=region_mapper,
    )
    with pytest.raises(
        RegionGrowerError, match="Missing parameters for mtype 'L2_TPC:A' in region 'default'"
    ):
        SynthesizeMorphologies.verify(data)

    failing_params = deepcopy(tmd_parameters)
    del failing_params["default"][mtype]["origin"]
    data = Data(
        tmd_distributions=tmd_distributions,
        tmd_parameters=failing_params,
        cells=cell_collection,
        region_mapper=region_mapper,
    )
    with pytest.raises(neurots.validator.ValidationError, match=r"'origin' is a required property"):
        SynthesizeMorphologies.verify(data)

    # Fail when missing attributes
    attributes = ["layer", "fraction", "slope", "intercept"]
    good_params = deepcopy(tmd_parameters)
    good_params["default"][mtype]["context_constraints"] = {
        "apical_dendrite": {
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
    data = Data(
        tmd_distributions=tmd_distributions,
        tmd_parameters=good_params,
        cells=cell_collection,
        region_mapper=region_mapper,
    )
    SynthesizeMorphologies.verify(data)
    for i in range(1, 5):
        for missing_attributes in combinations(attributes, i):
            failing_params = deepcopy(good_params["default"][mtype])
            for att in missing_attributes:
                del failing_params["context_constraints"]["apical_dendrite"]["extent_to_target"][
                    att
                ]
            tmd_parameters["default"][mtype] = failing_params
            data = Data(
                tmd_distributions=tmd_distributions,
                tmd_parameters=tmd_parameters,
                cells=cell_collection,
                region_mapper=region_mapper,
            )
            with pytest.raises(
                jsonschema.exceptions.ValidationError, match="is a required property"
            ):
                SynthesizeMorphologies.verify(data)


def test_check_axon_morphology(caplog):
    """Test the _check_axon_morphology() function."""
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


def test_RegionMapper(small_O1_path):
    region_mapper = RegionMapper(
        ["O0", "UNKNOWN"], RegionMap.load_json(Path(small_O1_path) / "hierarchy.json")
    )
    # pylint: disable=protected-access
    assert region_mapper.mapper == {
        "mc0_Column": "O0",
        "mc0;6": "O0",
        "mc0;5": "O0",
        "mc0;4": "O0",
        "mc0;3": "O0",
        "mc0;2": "O0",
        "mc0;1": "O0",
        "O0": "O0",
    }
    assert region_mapper.inverse_mapper == {
        "O0": set(["mc0_Column", "mc0;1", "mc0;2", "mc0;3", "mc0;4", "mc0;5", "mc0;6", "O0"]),
        "default": set(),
    }

    assert region_mapper["O0"] == "O0"
    assert region_mapper["mc0;1"] == "O0"
    assert region_mapper["OTHER"] == "default"


def test_inconsistent_params(
    tmpdir,
    small_O1_path,
    input_cells,
    axon_morph_tsv,
):  # pylint: disable=unused-argument
    """Test morphology synthesis but skip write step."""
    min_depth = 25
    tmp_folder = Path(tmpdir)

    args, _ = create_args(
        False,
        tmp_folder,
        input_cells,
        small_O1_path,
        axon_morph_tsv,
        "apical_NRN_sections.yaml",
        min_depth,
    )
    args["out_morph_ext"] = ["h5"]

    with pytest.raises(
        ValueError,
        match=(
            r"""The 'out_morph_ext' parameter must contain one of \["asc", "swc"\] when """
            r"""'with_NRN_sections' is set to True \(current value is \['h5'\]\)\."""
        ),
    ):
        SynthesizeMorphologies(**args)


def test_inconsistent_context(
    tmpdir,
    small_O1_path,
    input_cells,
    axon_morph_tsv,
    caplog,
):  # pylint: disable=unused-argument
    """Test morphology synthesis with inconsistent context constraints."""
    min_depth = 25
    tmp_folder = Path(tmpdir)

    args, parallel_args = create_args(
        False,
        tmp_folder,
        input_cells,
        small_O1_path,
        axon_morph_tsv,
        "apical_NRN_sections.yaml",
        min_depth,
    )
    parallel_args["nb_processes"] = 0

    with args["tmd_parameters"].open("r", encoding="utf-8") as f:
        tmd_parameters = json.load(f)

    tmd_parameters["default"]["L2_TPC:A"]["context_constraints"] = {
        "apical_dendrite": {
            "extent_to_target": {
                "slope": 0.5,
                "intercept": 1,
                "layer": 1,
                "fraction": 0.5,
            }
        }
    }

    args["tmd_parameters"] = tmp_folder / "tmd_parameters.json"
    with args["tmd_parameters"].open("w", encoding="utf-8") as f:
        json.dump(tmd_parameters, f)

    parallel_client, args["nb_workers"] = initialize_parallel_client(**parallel_args)
    synthesizer = SynthesizeMorphologies(**args)

    caplog.set_level(logging.WARNING)
    caplog.clear()
    synthesizer.synthesize()
    assert caplog.record_tuples[0] == (
        "region_grower.synthesize_morphologies",
        30,
        "The morphologies with the following region/mtype couples have inconsistent context and "
        "constraints: [('default', 'L2_TPC:A')]",
    )

    close_parallel_client(parallel_client)
