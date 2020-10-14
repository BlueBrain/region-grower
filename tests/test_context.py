import itertools
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import jsonschema
import numpy as np
from nose.tools import assert_equal, assert_raises
from numpy.testing import assert_array_almost_equal
from region_grower import RegionGrowerError
# from region_grower.context import SpaceNeuronGrower
from region_grower.context import SpaceContext
from voxcell.nexus.voxelbrain import Atlas

from atlas_mock import small_O1

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

    mtype = "L2_TPC:A"
    initial_params = deepcopy(context.tmd_parameters)

    context.verify([mtype])
    assert_raises(RegionGrowerError, context.verify, ["UNKNOWN_MTYPE"])

    good_params = deepcopy(initial_params)

    del context.tmd_parameters[mtype]
    assert_raises(RegionGrowerError, context.verify, [mtype])

    context.tmd_parameters = good_params
    del context.tmd_parameters[mtype]['origin']
    assert_raises(jsonschema.exceptions.ValidationError, context.verify, [mtype])

    # Fail when missing attributes
    attributes = ["layer", "fraction", "slope", "intercept"]
    good_params = deepcopy(initial_params)
    good_params[mtype]["context_constraints"] = {
        "apical": {
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
            }
        }
    }
    context.tmd_parameters = deepcopy(good_params)
    context.verify([mtype])
    for i in range(1, 5):
        for missing_attributes in itertools.combinations(attributes, i):
            failing_params = deepcopy(good_params[mtype])
            for att in missing_attributes:
                del failing_params["context_constraints"]["apical"]["extent_to_target"][att]
            context.tmd_parameters[mtype] = failing_params
            assert_raises(
                jsonschema.exceptions.ValidationError,
                context.verify,
                [mtype],
            )


def test_scale():
    np.random.seed(0)
    mtype = "L2_TPC:A"

    with TemporaryDirectory() as tempdir:
        small_O1(tempdir)
        context = SpaceContext(
            Atlas.open(tempdir), DATA / "distributions.json", DATA / "parameters.json"
        )

    context.tmd_parameters[mtype]["context_constraints"] = {
        "apical": {
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
                "fraction": 0.1,  # Set max < target to ensure a rescaling is processed
            }
        }
    }
    context.verify([mtype])
    result = context.synthesize([100, -100, 100], mtype)

    assert_array_almost_equal(
        result.apical_points, np.array([[-1.54834383, 43.40647383, -1.60015317]])
    )

    # Test scale computation
    params = context.tmd_parameters[mtype]

    assert params.get("apical", {}).get("modify", {}) is None
    assert params.get("basal", {}).get("modify", {}) is None

    fixed_params = context._correct_position_orientation_scaling(params)

    expected_apical = {"target_path_distance": 76}
    expected_basal = {"reference_thickness": 314, "target_thickness": 300.0}
    assert (
        fixed_params.get("apical", {}).get("modify", {}).get("kwargs", {}) == expected_apical
    )
    assert fixed_params.get("basal", {}).get("modify", {}).get("kwargs", {}) == expected_basal
