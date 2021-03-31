"""Setup test fixtures."""
# pylint: disable=redefined-outer-name
import pytest
from voxcell.nexus.voxelbrain import Atlas

from .atlas_mock import generate_axon_morph_tsv
from .atlas_mock import generate_cell_collection
from .atlas_mock import generate_cells_df
from .atlas_mock import generate_input_cells
from .atlas_mock import generate_small_O1


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


@pytest.fixture
def cells_df():
    """Raw data for the cell collection."""
    return generate_cells_df()


@pytest.fixture
def cell_collection(cells_df):
    """The cell collection."""
    return generate_cell_collection(cells_df)


@pytest.fixture
def input_cells(cell_collection, tmpdir):
    """The path to the MVD3 file containing the cell collection."""
    return generate_input_cells(cell_collection, tmpdir)


@pytest.fixture
def axon_morph_tsv(tmpdir):
    """The TSV file containing the morphologies from which the axon must be used for grafting."""
    return generate_axon_morph_tsv(tmpdir)
