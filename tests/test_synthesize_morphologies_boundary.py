"""Test the region_grower.synthesize_morphologies module."""
# pylint: disable=missing-function-docstring
import logging
import shutil
from pathlib import Path
from uuid import uuid4

from morph_tool.utils import iter_morphology_files
from numpy.testing import assert_allclose

from region_grower.synthesize_morphologies import SynthesizeMorphologies

from .test_synthesize_morphologies import check_yaml

DATA = Path(__file__).parent / "data"


def create_args(
    with_mpi,
    tmp_folder,
    input_cells,
    atlas_path,
    axon_morph_tsv,
    out_apical_NRN_sections,
    min_depth,
    region_structure_boundary,
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
    args["region_structure"] = region_structure_boundary

    return args


def test_synthesize(
    tmpdir,
    small_O1_path,
    input_cells,
    axon_morph_tsv,
    mesh,
    region_structure_boundary,
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
        region_structure_boundary,
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
    print(res["apical_points"])
    assert_allclose(res["apical_points"][0], [[-38.110016, 152.74228, -20.12851]])
    assert res["apical_points"][1] is None

    assert_allclose(res["apical_points"][3], [[-3.50943, 115.28183, 5.8795013]])
    assert res["apical_points"][4] is None

    assert_allclose(res["apical_points"][6], [[2.271698, 114.72031, -5.102951]])
    assert res["apical_points"][7] is None

    assert_allclose(res["apical_points"][9], [[12.257843, 85.439545, -19.165298]])
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
        region_structure_boundary=region_structure_path,
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


if __name__ == "__main__":  # pragma: no cover
    run_with_mpi()
