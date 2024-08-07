"""Test the region_grower.synthesize_morphologies module using MPI."""
# pylint: disable=import-error
import logging
import shutil
from pathlib import Path
from uuid import uuid4

from data_factories import generate_axon_morph_tsv
from data_factories import generate_cell_collection
from data_factories import generate_cells_df
from data_factories import generate_input_cells
from data_factories import generate_small_O1
from data_factories import input_cells_path
from morph_tool.utils import iter_morphology_files
from mpi4py import MPI
from test_synthesize_morphologies import check_yaml
from test_synthesize_morphologies import create_args

from region_grower.synthesize_morphologies import SynthesizeMorphologies
from region_grower.utils import close_parallel_client
from region_grower.utils import initialize_parallel_client
from region_grower.utils import setup_logger

DATA = Path(__file__).parent / "data"


def run_with_mpi():
    """Test morphology synthesis with MPI."""
    # pylint: disable=too-many-locals, import-error
    COMM = MPI.COMM_WORLD  # pylint: disable=c-extension-no-member
    rank = COMM.Get_rank()
    MASTER_RANK = 0
    is_master = rank == MASTER_RANK

    tmp_folder = Path("/tmp/test-run-synthesis_" + str(uuid4()))
    tmp_folder = COMM.bcast(tmp_folder, root=MASTER_RANK)
    input_cells = input_cells_path(tmp_folder)
    small_O1_path = str(tmp_folder / "atlas")

    args, parallel_args = create_args(
        True,
        tmp_folder,
        input_cells,
        small_O1_path,
        tmp_folder / "axon_morphs.tsv",
        "apical_NRN_sections.yaml",
        min_depth=25,
    )
    args["chunksize"] = 3

    setup_logger("debug", set_worker_prefix=True)
    logging.getLogger("distributed").setLevel(logging.ERROR)
    logger = logging.getLogger("============ TEST-SYNTHESIS-MPI ============")

    if is_master:
        tmp_folder.mkdir(exist_ok=True)
        logger.info("Create data")
        cells_raw_data = generate_cells_df()
        cell_collection = generate_cell_collection(cells_raw_data)
        generate_input_cells(cell_collection, tmp_folder)
        generate_small_O1(small_O1_path)
        generate_axon_morph_tsv(tmp_folder)

        for dest in range(1, COMM.Get_size()):
            req = COMM.isend("done", dest=dest)
    else:
        logger.info("Waiting for initialization")
        req = COMM.irecv(source=0)
        req.wait()

    parallel_client, args["nb_workers"] = initialize_parallel_client(**parallel_args)
    synthesizer = SynthesizeMorphologies(**args)

    try:
        logger.info("Start synthesize")
        synthesizer.synthesize()

        # Check results
        logger.info("Checking results")
        expected_size = 18
        assert len(list(iter_morphology_files(tmp_folder))) == expected_size

        check_yaml(DATA / ("apical.yaml"), args["out_apical"])
        check_yaml(
            DATA / ("apical_NRN_sections.yaml"),
            args["out_apical_nrn_sections"],
        )
    finally:
        # Clean the directory
        logger.info("Cleaning")
        shutil.rmtree(tmp_folder)
    close_parallel_client(parallel_client)


if __name__ == "__main__":  # pragma: no cover
    run_with_mpi()
