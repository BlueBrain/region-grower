"""
This test is being run through mpiexec, not nosetests
mpiexec -n 4 python test_synthesize_morphologies.py

The test function should NOT have *test* in the name, or nosetests
will run it.
"""
# pylint: disable=missing-function-docstring
import shutil
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from mock import MagicMock
from mock import patch
from morph_tool.utils import iter_morphology_files
from numpy.testing import assert_allclose

import region_grower.synthesize_morphologies as tested
from region_grower.placement_algorithm_mpi_app import MASTER_RANK
from region_grower.placement_algorithm_mpi_app import _run

try:
    from .atlas_mock import generate_cell_collection
    from .atlas_mock import generate_cells_df
    from .atlas_mock import generate_input_cells
    from .atlas_mock import generate_small_O1
    from .atlas_mock import input_cells_path
except ImportError:
    # Usefull to run the tests with mpiexec
    from atlas_mock import generate_cell_collection
    from atlas_mock import generate_cells_df
    from atlas_mock import generate_input_cells
    from atlas_mock import generate_small_O1
    from atlas_mock import input_cells_path

PATH = Path(__file__).parent
DATA = Path(PATH, "data").resolve()


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
    no_mpi,
    tmp_folder,
    input_cells,
    atlas_path,
    axon_morph_tsv,
    out_apical_NRN_sections,
):
    args = MagicMock()

    # Circuit
    args.mvd3 = None
    args.cells_path = input_cells

    # Atlas
    args.atlas = atlas_path

    # Parameters
    args.tmd_distributions = DATA / "distributions.json"
    args.tmd_parameters = DATA / "parameters.json"
    args.seed = 0

    # Internals
    args.num_files = 12
    args.max_files_per_dir = 256
    args.overwrite = True
    args.no_mpi = no_mpi
    args.out_morph_ext = ["h5", "swc", "asc"]
    args.out_morph_dir = tmp_folder
    args.out_apical = tmp_folder / "apical.yaml"
    args.out_cells_path = str(tmp_folder / "test_cells.mvd3")
    args.out_mvd3 = None
    if out_apical_NRN_sections:
        args.out_apical_NRN_sections = tmp_folder / out_apical_NRN_sections
    else:
        args.out_apical_NRN_sections = None

    # Axons
    args.base_morph_dir = str(DATA / "input-cells")
    args.morph_axon = axon_morph_tsv
    args.max_drop_ratio = 0.5
    args.rotational_jitter_std = 10
    args.scaling_jitter_std = 0.5

    return args


def run_mpi():
    # pylint: disable=import-error,import-outside-toplevel,c-extension-no-member
    from mpi4py import MPI

    tmp_folder = Path("/tmp/test-run-synthesis_" + str(uuid4()))

    COMM = MPI.COMM_WORLD  # pylint: disable=c-extension-no-member
    tmp_folder = COMM.bcast(tmp_folder, root=MASTER_RANK)
    rank = COMM.Get_rank()
    is_master = rank == MASTER_RANK

    input_cells = input_cells_path(tmp_folder)
    small_O1_path = str(tmp_folder / "atlas")

    args = create_args(
        False,
        tmp_folder,
        input_cells,
        small_O1_path,
        None,
        "apical_NRN_sections.yaml",
    )

    if is_master:
        tmp_folder.mkdir(exist_ok=True)
        print(f"#{rank}: Create data")
        cells_raw_data = generate_cells_df()
        cell_collection = generate_cell_collection(cells_raw_data)
        generate_input_cells(cell_collection, tmp_folder)
        generate_small_O1(small_O1_path)

    print(f"#{rank}: Joining all processes")
    COMM.barrier()

    if not is_master:
        print(f"#{rank}: Running child")
        _run(tested.Master, args)
        print(f"#{rank}: Ending child")
        return

    try:
        print(f"#{rank}: Running master")
        _run(tested.Master, args)
        print(f"#{rank}: Checking results")
        assert len(list(iter_morphology_files(tmp_folder))) == 36
        check_yaml(DATA / "apical.yaml", args.out_apical)
        check_yaml(DATA / "apical_NRN_sections.yaml", args.out_apical_NRN_sections)
        print(f"#{rank}: Ending master")
    finally:
        print(f"#{rank}: Cleaning")
        shutil.rmtree(tmp_folder)


@pytest.mark.parametrize("with_axon", [True, False])
@pytest.mark.parametrize("with_NRN", [True, False])
def test_run_no_mpi(
    tmpdir, small_O1_path, input_cells, axon_morph_tsv, with_axon, with_NRN
):  # pylint: disable=unused-argument
    tmp_folder = Path(tmpdir)

    args = create_args(
        True,
        tmp_folder,
        input_cells,
        small_O1_path,
        axon_morph_tsv if with_axon else None,
        "apical_NRN_sections.yaml" if with_NRN else None,
    )

    master = tested.Master()
    worker = master.setup(args)
    worker.setup(args)
    result = {k: worker(k) for k in master.task_ids}
    master.finalize(result)
    assert len(list(iter_morphology_files(tmp_folder))) == 36
    if with_axon:
        check_yaml(DATA / "apical_with_axons.yaml", args.out_apical)
        if with_NRN:
            check_yaml(DATA / "apical_NRN_sections_with_axons.yaml", args.out_apical_NRN_sections)
    else:
        check_yaml(DATA / "apical.yaml", args.out_apical)
        if with_NRN:
            check_yaml(DATA / "apical_NRN_sections.yaml", args.out_apical_NRN_sections)


def test_parser():
    master = tested.Master()
    with patch(
        "sys.argv",
        [
            "synthesize_morphologies",
            "--tmd-parameters",
            "test_params",
            "--tmd-distributions",
            "test_distributions",
            "--atlas",
            "test_atlas",
            "--out-apical",
            "test_out",
        ],
    ):
        res = master.parse_args()
        assert res.tmd_parameters == "test_params"
        assert res.tmd_distributions == "test_distributions"
        assert res.atlas == "test_atlas"
        assert res.out_apical == "test_out"


if __name__ == "__main__":
    run_mpi()
