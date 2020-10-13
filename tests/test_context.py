from pathlib import Path
from tempfile import TemporaryDirectory
from copy import deepcopy

import numpy as np
from nose.tools import assert_equal
from nose.tools import assert_almost_equal
from nose.tools import assert_raises
from numpy.testing import assert_array_almost_equal
import jsonschema

from voxcell.nexus.voxelbrain import Atlas

from atlas_mock import small_O1
from region_grower import RegionGrowerError
from region_grower.context import SpaceContext


DATA = Path(__file__).parent / "data"


def test_context():
    np.random.seed(0)

    with TemporaryDirectory() as tempdir:
        small_O1(tempdir)
        context = SpaceContext(
            Atlas.open(tempdir), DATA / "distributions.json", DATA / "parameters.json"
        )

    result = context.synthesize([100, -100, 100], "L2_TPC:A")

    assert_array_almost_equal(
        result.apical_points, np.array([[9.40834, 114.985021, -25.603346]])
    )

    # This tests that input orientations are not mutated by the synthesize() call
    assert_array_almost_equal(
        context.tmd_parameters["L2_TPC:A"]["apical"]["orientation"], [[0.0, 1.0, 0.0]]
    )

    assert_array_almost_equal(
        result.neuron.soma.points,
        np.array(
            [
                [-9.1213598, 0.34562492, 0.0],
                [4.6709661, -7.8600206, 0.0],
                [9.1049757, -0.83503044, 0.0],
                [0.12655652, 9.0572395, 0.0],
                [0.0, 9.1431866, 0.0],
                [-4.1508913, 7.8628845, 0.0],
                [-9.0374002, 1.3868117, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    assert_equal(len(result.neuron.root_sections), 7)
    assert_array_almost_equal(
        next(result.neuron.iter()).points,
        np.array(
            [
                [0.126557, 9.05724, 1.244288],
                [0.13983, 10.007194, 1.374793],
                [0.249028, 10.898591, 1.370834],
            ],
            dtype=np.float32,
        ),
    )
    assert_array_almost_equal(
        next(result.neuron.iter()).diameters,
        np.array([0.6, 0.6, 0.6], dtype=np.float32),
    )


def test_context_external_diametrizer():
    np.random.seed(0)

    with TemporaryDirectory() as tempdir:
        small_O1(tempdir)
        context = SpaceContext(
            Atlas.open(tempdir),
            DATA / "distributions_external_diametrizer.json",
            DATA / "parameters_external_diametrizer.json",
        )

    result = context.synthesize([100, -100, 100], "L2_TPC:A")

    assert_array_almost_equal(
        result.apical_points, np.array([[9.40834, 114.985021, -25.603346]])
    )

    # This tests that input orientations are not mutated by the synthesize() call
    assert_array_almost_equal(
        context.tmd_parameters["L2_TPC:A"]["apical"]["orientation"], [[0.0, 1.0, 0.0]]
    )

    assert_array_almost_equal(
        result.neuron.soma.points,
        np.array(
            [
                [-9.1213598, 0.34562492, 0.0],
                [4.6709661, -7.8600206, 0.0],
                [9.1049757, -0.83503044, 0.0],
                [0.12655652, 9.0572395, 0.0],
                [0.0, 9.1431866, 0.0],
                [-4.1508913, 7.8628845, 0.0],
                [-9.0374002, 1.3868117, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    assert_equal(len(result.neuron.root_sections), 7)
    assert_array_almost_equal(
        next(result.neuron.iter()).points,
        np.array(
            [
                [0.126557, 9.05724, 1.244288],
                [0.13983, 10.007194, 1.374793],
                [0.249028, 10.898591, 1.370834],
            ],
            dtype=np.float32,
        ),
    )

    assert_array_almost_equal(
        next(result.neuron.iter()).diameters,
        np.array([0.983045, 0.981993, 0.981009], dtype=np.float32),
    )


def test_verify():
    with TemporaryDirectory() as tempdir:
        small_O1(tempdir)
        context = SpaceContext(
            Atlas.open(tempdir), DATA / "distributions.json", DATA / "parameters.json"
        )

    context.verify(["L2_TPC:A"])
    assert_raises(RegionGrowerError, context.verify, ["UNKNOWN_MTYPE"])

    good_params = deepcopy(context.tmd_parameters)

    del context.tmd_parameters["L2_TPC:A"]
    assert_raises(RegionGrowerError, context.verify, ["L2_TPC:A"])

    context.tmd_parameters = good_params
    del context.tmd_parameters["L2_TPC:A"]['origin']
    assert_raises(jsonschema.exceptions.ValidationError, context.verify, ["L2_TPC:A"])




def test_scale():
    np.random.seed(0)

    with TemporaryDirectory() as tempdir:
        small_O1(tempdir)
        context = SpaceContext(
            Atlas.open(tempdir), DATA / "distributions.json", DATA / "parameters.json"
        )

    result = context.synthesize([100, -100, 100], "L2_TPC:A")

    assert_array_almost_equal(
        result.apical_points, np.array([[9.40834, 114.985021, -25.603346]])
    )

    # Test scale computation
    neurite = result.neuron.root_sections[0]

    params = context.tmd_parameters["L2_TPC:A"]

    # Set bias value
    params["apical"]["bias_length"] = 1

    assert params.get("apical", {}).get("modify", {}) is None
    assert params.get("basal", {}).get("modify", {}) is None

    fixed_params = context._correct_position_orientation_scaling(
        params, [100, -100, 100]
    )

    expected = {"reference_thickness": 314, "target_thickness": 300.0}
    assert (
        fixed_params.get("apical", {}).get("modify", {}).get("kwargs", {}) == expected
    )
    assert_almost_equal(
        fixed_params.get("apical", {}).get("bias_length"), 0.9554140127388535
    )
    assert fixed_params.get("basal", {}).get("modify", {}).get("kwargs", {}) == expected
