"""Configuration for the pytest test suite."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of region-grower.
# See https://github.com/BlueBrain/region-grower for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

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
    """The position of a cell."""
    return get_cell_position()


@pytest.fixture(scope="function")
def cell_mtype():
    """The mtype of a cell."""
    return get_cell_mtype()


@pytest.fixture(scope="function")
def cell_orientation():
    """The orientation of a cell."""
    return get_cell_orientation()


@pytest.fixture(scope="function")
def tmd_parameters():
    """The TMD parameters."""
    return get_tmd_parameters(DATA / "parameters.json")


@pytest.fixture(scope="function")
def tmd_distributions():
    """The TMD distributions."""
    return get_tmd_distributions(DATA / "distributions.json")


@pytest.fixture(scope="function")
def cell_state(cell_position, cell_mtype, cell_orientation):
    """A cell state object."""
    return CellState(
        position=cell_position,
        orientation=cell_orientation,
        mtype=cell_mtype,
        depth=250,
    )


@pytest.fixture(scope="function")
def space_context():
    """A space context object."""
    layer_depth = [0.0, 200.0, 300.0, 400.0, 500.0, 600.0, 800.0]
    thicknesses = [165, 149, 353, 190, 525, 700]
    return SpaceContext(layer_depths=layer_depth, cortical_depths=np.cumsum(thicknesses))


@pytest.fixture(scope="function")
def synthesis_parameters(cell_mtype, tmd_distributions, tmd_parameters):
    """Synthesis parameters object."""
    return SynthesisParameters(
        tmd_distributions=tmd_distributions["default"][cell_mtype],
        tmd_parameters=tmd_parameters["default"][cell_mtype],
        min_hard_scale=0.2,
    )


@pytest.fixture(scope="function")
def computation_parameters():
    """Computation parameters object."""
    return ComputationParameters()


@pytest.fixture(scope="function")
def small_context_worker(cell_state, space_context, synthesis_parameters, computation_parameters):
    """A small space worker object."""
    return SpaceWorker(cell_state, space_context, synthesis_parameters, computation_parameters)


@pytest.fixture(scope="session")
def synthesized_cell():
    """A synthesized cell."""
    np.random.seed(0)

    tmd_parameters = get_tmd_parameters(DATA / "parameters.json")
    tmd_distributions = get_tmd_distributions(DATA / "distributions.json")

    cell_position = get_cell_position()
    cell_mtype = get_cell_mtype()
    cell_orientation = get_cell_orientation()

    cell_state = CellState(
        position=cell_position,
        orientation=cell_orientation,
        mtype=cell_mtype,
        depth=250,
    )

    layer_depth = [0.0, 200.0, 300.0, 400.0, 500.0, 600.0, 800.0]
    thicknesses = [165, 149, 353, 190, 525, 700]
    space_context = SpaceContext(
        layer_depths=layer_depth,
        cortical_depths=np.cumsum(thicknesses),
    )
    synthesis_parameters = SynthesisParameters(
        tmd_distributions=tmd_distributions["default"][cell_mtype],
        tmd_parameters=tmd_parameters["default"][cell_mtype],
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
    """The morph loader."""
    return MorphLoader(DATA / "input-cells", file_ext="h5")


@pytest.fixture(scope="function")
def morph_writer(tmpdir):
    """The morph writer."""
    return MorphWriter(tmpdir, file_exts=["h5"])
