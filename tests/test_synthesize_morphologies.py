"""Test the region_grower.synthesize_morphologies module."""
# pylint: disable=missing-function-docstring
import json
import sys
import logging
import os
import shutil
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from uuid import uuid4

import attr
import dask
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
    region_structure,
):
    """Create the arguments used for tests."""
    args = {}

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
    args["region_structure"] = region_structure

    return args


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

    args = create_args(
        False,
        tmp_folder,
        input_cells,
        small_O1_path,
        axon_morph_tsv if with_axon else None,
        "apical_NRN_sections.yaml" if with_NRN else None,
        min_depth,
        DATA / "region_structure.yaml",
    )

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
            assert_allclose(max_y, 167.36578)
    else:
        if with_NRN and with_axon:
            assert_allclose(max_y, 150.18933)


@pytest.mark.parametrize("with_SHMDIR", [True, False])
@pytest.mark.parametrize("with_TMPDIR", [True, False])
@pytest.mark.parametrize("with_dask_config", [True, False])
def test_synthesize_dask_config(
    tmpdir,
    small_O1_path,
    input_cells,
    with_SHMDIR,
    with_TMPDIR,
    with_dask_config,
    monkeypatch,
):  # pylint: disable=unused-argument
    """Test morphology synthesis."""
    tmp_folder = Path(tmpdir)

    args = create_args(
        False,
        tmp_folder,
        input_cells,
        small_O1_path,
        None,
        None,
        100,
    )

    custom_scratch_config = str(tmp_folder / "custom_scratch_config")
    custom_scratch_env_SHMDIR = str(tmp_folder / "custom_scratch_SHMDIR")
    custom_scratch_env_TMPDIR = str(tmp_folder / "custom_scratch_TMPDIR")
    dask_config = None
    if with_dask_config is not None:
        dask_config = {"temporary-directory": custom_scratch_config}
        args["dask_config"] = dask_config

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

        synthesizer = SynthesizeMorphologies(**args)
        synthesizer._init_parallel(mpi_only=True)  # pylint: disable=protected-access

        if dask_config is not None:
            assert dask.config.get("temporary-directory", None) == custom_scratch_config
        elif with_TMPDIR:
            assert dask.config.get("temporary-directory", None) == custom_scratch_env_TMPDIR
        elif with_SHMDIR:
            assert dask.config.get("temporary-directory", None) == custom_scratch_env_SHMDIR
        else:
            assert dask.config.get("temporary-directory", None) is None


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
    with_axon = True
    with_NRN = True
    min_depth = 25
    tmp_folder = Path(tmpdir)

    args = create_args(
        False,
        tmp_folder,
        input_cells,
        small_O1_path,
        axon_morph_tsv if with_axon else None,
        "apical_NRN_sections.yaml" if with_NRN else None,
        min_depth,
        DATA / "region_structure.yaml",
    )
    args["skip_write"] = True
    args["nb_processes"] = nb_processes
    args["hide_progress_bar"] = True
    args["out_apical"] = None
    args["chunksize"] = chunksize

    print("Number of available CPUs", os.cpu_count())

    synthesizer = SynthesizeMorphologies(**args)
    res = synthesizer.synthesize()

    assert (res["x"] == 200).all()
    assert (res["y"] == 200).all()
    assert (res["z"] == 200).all()

    assert res["name"].tolist() == [
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
    assert [[i[0].tolist()] if i else i for i in res["apical_points"].tolist()] == [
        [[-4.8529953956604, 158.9239959716797, 1.4561792612075806]],
        None,
        [[-3.42299222946167, 116.98204040527344, -1.922537922859192]],
        None,
        [[5.770873546600342, 112.7794189453125, -10.71194839477539]],
        None,
        [[12.359382629394531, 90.04046630859375, -1.825372576713562]],
        None,
        [[6.219625949859619, 371.3540344238281, 6.843407154083252]],
        [[50.62730407714844, 181.1993865966797, -23.8173828125]],
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


@pytest.mark.parametrize("with_sections", [True, False])
@pytest.mark.parametrize("with_trunks", [True, False])
def test_synthesize_boundary(
    tmpdir,
    small_O1_path,
    input_cells,
    axon_morph_tsv,
    mesh,
    with_sections,
    with_trunks,
):  # pylint: disable=unused-argument
    """Test morphology synthesis but skip write step."""
    with_axon = True
    with_NRN = True
    min_depth = 25
    tmp_folder = Path(tmpdir)

    # pylint: disable=import-outside-toplevel
    from .data_factories import generate_region_structure_boundary

    region_structure = "region_structure.yaml"
    generate_region_structure_boundary(
        DATA / "region_structure.yaml", region_structure, mesh, with_sections, with_trunks
    )
    args = create_args(
        False,
        tmp_folder,
        input_cells,
        small_O1_path,
        axon_morph_tsv if with_axon else None,
        "apical_NRN_sections.yaml" if with_NRN else None,
        min_depth,
        region_structure,
    )
    args["skip_write"] = True

    synthesizer = SynthesizeMorphologies(**args)
    res = synthesizer.synthesize()

    assert (res["x"] == 200).all()
    assert (res["y"] == 200).all()
    assert (res["z"] == 200).all()
    assert res["name"].tolist() == [
        "e3e70682c2094cac629f6fbed82c07cd",
        None,
        "216363698b529b4a97b750923ceb3ffd",
        None,
        "14a03569d26b949692e5dfe8cb1855fe",
        None,
        "4462ebfc5f915ef09cfbac6e7687a66e",
        None,
    ]
    if with_sections and with_trunks:
        assert_allclose(res["apical_points"][0], [[16.126877, 160.5279, 22.8638]])
        assert res["apical_points"][1] is None

        assert_allclose(res["apical_points"][3], [[5.216095, 114.55658, -12.494751]])
        assert res["apical_points"][4] is None

        assert_allclose(res["apical_points"][6], [[4.36911, 55.535736, -6.2556915]])
        assert res["apical_points"][7] is None

        assert_allclose(res["apical_points"][9], [[-11.084274, 111.49124, 0.98043823]])
        assert res["apical_points"][10] is None

    if with_sections and not with_trunks:
        assert_allclose(res["apical_points"][0], [[-0.29234314, 58.81488, 0.5384369]])
        assert res["apical_points"][1] is None

        assert_allclose(res["apical_points"][3], [[8.230438, 116.570435, -7.345169]])
        assert res["apical_points"][4] is None

        assert_allclose(res["apical_points"][6], [[11.181992, 96.11371, -12.706863]])
        assert res["apical_points"][7] is None

        assert_allclose(res["apical_points"][9], [[3.5267944, 164.42618, 1.2018433]])
        assert res["apical_points"][10] is None

    if not with_sections and with_trunks:
        assert_allclose(res["apical_points"][0], [[8.792313, 131.91104, -4.2198334]])
        assert res["apical_points"][1] is None

        assert_allclose(res["apical_points"][3], [[2.0048676, 116.672, -5.334656]])
        assert res["apical_points"][4] is None

        assert_allclose(res["apical_points"][6], [[-2.347641, 58.229614, -0.56985474]])
        assert res["apical_points"][7] is None

        assert_allclose(res["apical_points"][9], [[6.861679, 112.31445, -13.528854]])
        assert res["apical_points"][10] is None

    # Check that the morphologies were not written
    res_files = tmpdir.listdir()
    assert len(res_files) == 5
    assert sorted(i.basename for i in res_files) == [
        "apical.yaml",
        "apical_NRN_sections.yaml",
        "axon_morphs.tsv",
        "input_cells.mvd3",
        "test_cells.mvd3",
    ]


def run_with_mpi():
    """Test morphology synthesis with MPI."""
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
        min_depth=25,
        region_structure=DATA / "region_structure.yaml",
    )

    setup_logger("debug", prefix=f"Rank = {rank} - ")
    logging.getLogger("distributed").setLevel(logging.ERROR)

    if is_master:
        tmp_folder.mkdir(exist_ok=True)
        print(f"============= #{rank}: Create data")
        cells_raw_data = generate_cells_df()
        cell_collection = generate_cell_collection(cells_raw_data)
        generate_input_cells(cell_collection, tmp_folder)
        generate_small_O1(small_O1_path)
        generate_axon_morph_tsv(tmp_folder)

        for dest in range(1, COMM.Get_size()):
            req = COMM.isend("done", dest=dest)
    else:
        print(f"============= #{rank}: Waiting for initialization")
        req = COMM.irecv(source=0)
        req.wait()

    synthesizer = SynthesizeMorphologies(**args)
    try:
        print(f"============= #{rank}: Start synthesize")
        synthesizer.synthesize()

        # Check results
        print(f"============= #{rank}: Checking results")
        expected_size = 18
        assert len(list(iter_morphology_files(tmp_folder))) == expected_size
        check_yaml(DATA / "apical.yaml", args["out_apical"])
        check_yaml(DATA / "apical_NRN_sections.yaml", args["out_apical_nrn_sections"])
    finally:
        # Clean the directory
        print(f"============= #{rank}: Cleaning")
        shutil.rmtree(tmp_folder)


def run_with_mpi_boundary():
    """Test morphology synthesis with MPI."""
    # pylint: disable=import-outside-toplevel, too-many-locals, import-error
    from data_factories import generate_axon_morph_tsv
    from data_factories import generate_cell_collection
    from data_factories import generate_cells_df
    from data_factories import generate_input_cells
    from data_factories import generate_mesh
    from data_factories import generate_region_structure_boundary
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
    region_structure_path = str(tmp_folder / "region_structure_boundary.yaml")
    args = create_args(
        True,
        tmp_folder,
        input_cells,
        small_O1_path,
        tmp_folder / "axon_morphs.tsv",
        "apical_NRN_sections.yaml",
        min_depth=25,
        region_structure=region_structure_path,
    )

    setup_logger("info")
    logging.getLogger("distributed").setLevel(logging.ERROR)

    if is_master:
        tmp_folder.mkdir(exist_ok=True)
        print(f"============= #{rank}: Create data")
        cells_raw_data = generate_cells_df()
        cell_collection = generate_cell_collection(cells_raw_data)
        generate_input_cells(cell_collection, tmp_folder)
        generate_small_O1(small_O1_path)
        atlas = {"atlas": small_O1_path, "structure": DATA / "region_structure.yaml"}
        generate_mesh(atlas, tmp_folder / "mesh.obj")
        region_structure_path = generate_region_structure_boundary(
            DATA / "region_structure.yaml", region_structure_path, str(tmp_folder / "mesh.obj")
        )

        generate_axon_morph_tsv(tmp_folder)
        for dest in range(1, COMM.Get_size()):
            req = COMM.isend("done", dest=dest)
    else:
        req = COMM.irecv(source=0)
        req.wait()

    synthesizer = SynthesizeMorphologies(**args)
    try:
        print(f"============= #{rank}: Start synthesize")
        synthesizer.synthesize()

        # Check results
        print(f"============= #{rank}: Checking results")
        expected_size = 12
        assert len(list(iter_morphology_files(tmp_folder))) == expected_size
        check_yaml(DATA / "apical_boundary.yaml", args["out_apical"])
    finally:
        # Clean the directory
        print(f"============= #{rank}: Cleaning")
        shutil.rmtree(tmp_folder)


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

    args = create_args(
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

    args = create_args(
        False,
        tmp_folder,
        input_cells,
        small_O1_path,
        axon_morph_tsv,
        "apical_NRN_sections.yaml",
        min_depth,
    )
    args["nb_processes"] = 0

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


if __name__ == "__main__":  # pragma: no cover
    if sys.argv[-1] == "boundary":
        run_with_mpi_boundary()
    else:
        run_with_mpi()
