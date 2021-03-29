"""
This test is being run through mpiexec, not nosetests
mpiexec -n 4 python test_synthesize_morphologies.py

The test function should NOT have *test* in the name, or nosetests
will run it.
"""
# pylint: disable=missing-function-docstring
import shutil
import tempfile
from pathlib import Path

import yaml
from mock import MagicMock
from mock import patch
from morph_tool.utils import iter_morphology_files
from nose.tools import assert_equal
from nose.tools import ok_
from numpy.testing import assert_allclose

import region_grower.synthesize_morphologies as tested
from region_grower.placement_algorithm_mpi_app import MASTER_RANK
from region_grower.placement_algorithm_mpi_app import _run

try:
    from .atlas_mock import CellCollectionMock
    from .atlas_mock import small_O1_placement_algo
except ImportError:
    # Usefull to run the tests with mpiexec
    from atlas_mock import CellCollectionMock
    from atlas_mock import small_O1_placement_algo

PATH = Path(__file__).parent
DATA = Path(PATH, "data").resolve()


def check_yaml(ref_path, tested_path):
    ok_(ref_path.exists())
    ok_(tested_path.exists())
    with open(ref_path) as ref_file, open(tested_path) as tested_file:
        ref_obj = yaml.load(ref_file, Loader=yaml.FullLoader)
        tested_obj = yaml.load(tested_file, Loader=yaml.FullLoader)
    assert_equal(ref_obj.keys(), tested_obj.keys())
    for k in ref_obj.keys():
        assert_allclose(ref_obj[k], tested_obj[k])


def create_args(no_mpi, tmp_folder):
    args = MagicMock()
    args.tmd_distributions = DATA / "distributions.json"
    args.tmd_parameters = DATA / "parameters.json"
    args.morph_axon = None
    args.seed = 0
    args.num_files = 12
    args.max_files_per_dir = 256
    args.overwrite = True
    args.out_morph_ext = ["h5", "swc", "asc"]
    args.out_morph_dir = tmp_folder
    args.out_apical = tmp_folder / "apical.yaml"
    args.atlas = str(tmp_folder)
    args.no_mpi = no_mpi
    args.out_apical_NRN_sections = tmp_folder / "apical_NRN_sections.yaml"

    return args


@patch(
    "region_grower.placement_algorithm_utils.CellCollection.load_mvd3",
    MagicMock(return_value=CellCollectionMock()),
)
def run_mpi():
    # pylint: disable=import-error,import-outside-toplevel,c-extension-no-member
    from mpi4py import MPI

    tmp_folder = Path("/tmp/test-run-synthesis")

    args = create_args(False, tmp_folder)

    is_master = MPI.COMM_WORLD.Get_rank() == MASTER_RANK

    if not is_master:
        _run(tested.Master, args)
        return

    tmp_folder.mkdir(exist_ok=True)

    try:
        small_O1_placement_algo(tmp_folder)
        _run(tested.Master, args)
        assert_equal(len(list(iter_morphology_files(tmp_folder))), 36)
        check_yaml(DATA / "apical.yaml", args.out_apical)
        check_yaml(DATA / "apical_NRN_sections.yaml", args.out_apical_NRN_sections)
    finally:
        shutil.rmtree(tmp_folder)


@patch(
    "region_grower.placement_algorithm_utils.CellCollection.load_mvd3",
    MagicMock(return_value=CellCollectionMock()),
)
def test_run_no_mpi():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_folder = Path(tmpdirname)

        args = create_args(True, tmp_folder)

        small_O1_placement_algo(tmp_folder)
        master = tested.Master()
        worker = master.setup(args)
        worker.setup(args)
        result = {k: worker(k) for k in master.task_ids}
        master.finalize(result)
        assert_equal(len(list(iter_morphology_files(tmp_folder))), 36)
        check_yaml(DATA / "apical.yaml", args.out_apical)
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
