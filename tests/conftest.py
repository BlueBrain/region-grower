"""Setup test fixtures."""
# pylint: disable=redefined-outer-name
# pylint: disable=missing-function-docstring
from pathlib import Path

import numpy as np
import pytest
from voxcell.nexus.voxelbrain import Atlas

from region_grower.context import CellState
from region_grower.context import ComputationParameters
from region_grower.context import SpaceContext
from region_grower.context import SpaceWorker
from region_grower.context import SynthesisParameters
from region_grower.morph_io import MorphLoader
from region_grower.morph_io import MorphWriter
from region_grower.synthesize_morphologies import SynthesizeMorphologies

from .data_factories import generate_axon_morph_tsv
from .data_factories import generate_cell_collection
from .data_factories import generate_cells_df
from .data_factories import generate_input_cells
from .data_factories import generate_small_O1
from .data_factories import get_cell_mtype
from .data_factories import get_cell_orientation
from .data_factories import get_cell_position
from .data_factories import get_tmd_distributions
from .data_factories import get_tmd_parameters

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def small_O1_path(tmpdir_factory):
    """Generate the atlas."""
    atlas_directory = str(tmpdir_factory.mktemp("atlas_small_O1"))
    generate_small_O1(atlas_directory)
    return atlas_directory


@pytest.fixture(scope="session")
def small_O1(small_O1_path):
    """Open the atlas."""
    return Atlas.open(small_O1_path)


@pytest.fixture(scope="function")
def cells_df():
    """Raw data for the cell collection."""
    return generate_cells_df()


@pytest.fixture(scope="function")
def cell_collection(cells_df):
    """The cell collection."""
    return generate_cell_collection(cells_df)


@pytest.fixture(scope="function")
def input_cells(cell_collection, tmpdir):
    """The path to the MVD3 file containing the cell collection."""
    return generate_input_cells(cell_collection, tmpdir)


@pytest.fixture
def axon_morph_tsv(tmpdir):
    """The TSV file containing the morphologies from which the axon must be used for grafting."""
    return generate_axon_morph_tsv(tmpdir)


@pytest.fixture(scope="function")
def cell_position():
    return get_cell_position()


@pytest.fixture(scope="function")
def cell_mtype():
    return get_cell_mtype()


@pytest.fixture(scope="function")
def cell_orientation():
    return get_cell_orientation()


@pytest.fixture(scope="function")
def tmd_parameters():
    return get_tmd_parameters(DATA / "parameters.json")


@pytest.fixture(scope="function")
def tmd_distributions():
    return get_tmd_distributions(DATA / "distributions.json")


@pytest.fixture(scope="function")
def cell_state(cell_position, cell_mtype, cell_orientation, small_O1_path):
    current_depth, _, _ = SynthesizeMorphologies.atlas_lookups(
        small_O1_path,
        [cell_position],
    )
    return CellState(
        position=cell_position,
        orientation=cell_orientation,
        mtype=cell_mtype,
        depth=current_depth[0],
    )


@pytest.fixture(scope="function")
def space_context(cell_position, small_O1_path, tmd_distributions):
    _, layer_depth, _ = SynthesizeMorphologies.atlas_lookups(
        small_O1_path,
        [cell_position],
    )
    return SpaceContext(
        layer_depths=layer_depth[:, 0].tolist(),
        cortical_depths=np.cumsum(tmd_distributions["metadata"]["cortical_thickness"]).tolist(),
    )


@pytest.fixture(scope="function")
def synthesis_parameters(cell_mtype, tmd_distributions, tmd_parameters):
    return SynthesisParameters(
        tmd_distributions=tmd_distributions["mtypes"][cell_mtype],
        tmd_parameters=tmd_parameters[cell_mtype],
        min_hard_scale=0.2,
    )


@pytest.fixture(scope="function")
def computation_parameters():
    return ComputationParameters()


@pytest.fixture(scope="function")
def small_context_worker(cell_state, space_context, synthesis_parameters, computation_parameters):
    return SpaceWorker(cell_state, space_context, synthesis_parameters, computation_parameters)


@pytest.fixture(scope="session")
def synthesized_cell(small_O1_path):
    np.random.seed(0)

    tmd_parameters = get_tmd_parameters(DATA / "parameters.json")
    tmd_distributions = get_tmd_distributions(DATA / "distributions.json")

    cell_position = get_cell_position()
    cell_mtype = get_cell_mtype()
    cell_orientation = get_cell_orientation()

    current_depth, layer_depth, _ = SynthesizeMorphologies.atlas_lookups(
        small_O1_path,
        [cell_position],
    )

    cell_state = CellState(
        position=cell_position,
        orientation=cell_orientation,
        mtype=cell_mtype,
        depth=current_depth[0],
    )
    space_context = SpaceContext(
        layer_depths=layer_depth[:, 0].tolist(),
        cortical_depths=np.cumsum(tmd_distributions["metadata"]["cortical_thickness"]).tolist(),
    )
    synthesis_parameters = SynthesisParameters(
        tmd_distributions=tmd_distributions["mtypes"][cell_mtype],
        tmd_parameters=tmd_parameters[cell_mtype],
        min_hard_scale=0.2,
    )
    computation_parameters = ComputationParameters()
    small_context_worker = SpaceWorker(
        cell_state,
        space_context,
        synthesis_parameters,
        computation_parameters,
    )

    return small_context_worker.synthesize()


@pytest.fixture(scope="function")
def morph_loader():
    return MorphLoader(DATA / "input-cells", file_ext="h5")


@pytest.fixture(scope="function")
def morph_writer(tmpdir):
    return MorphWriter(tmpdir, file_exts=["h5"])
