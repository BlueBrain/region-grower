import itertools
import logging
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import jsonschema
import numpy as np
from morphio import SectionType
from nose.tools import assert_equal, assert_raises
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal
from region_grower import RegionGrowerError
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

    # Synthesize in L2
    result = context.synthesize([0, 500, 0], "L2_TPC:A")

    assert_array_equal(
        result.apical_sections, np.array([48])
    )
    assert_array_almost_equal(
        [result.neuron.sections[i].points[-1] for i in result.apical_sections],
        np.array([[9.40834045, 114.9850235, -25.60334587]])
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

    result = context.synthesize([0, 500, 0], "L2_TPC:A")

    assert_array_equal(
        result.apical_sections, np.array([48])
    )
    assert_array_almost_equal(
        [result.neuron.sections[i].points[-1] for i in result.apical_sections],
        np.array([[9.40834045, 114.9850235, -25.60334587]])
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
        np.array([0.79342693, 0.79173011, 0.79014111], dtype=np.float32),
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
    mtype = "L2_TPC:A"

    with TemporaryDirectory() as tempdir:
        small_O1(tempdir)
        context = SpaceContext(
            Atlas.open(tempdir), DATA / "distributions.json", DATA / "parameters.json"
        )

    # Test with no hard limit scaling
    context.tmd_parameters[mtype]["context_constraints"] = {
        "apical": {
            "extent_to_target": {
                "slope": 0.5,
                "intercept": 1,
                "layer": 1,
                "fraction": 0.5,
            }
        }
    }
    context.verify([mtype])
    np.random.seed(0)
    result = context.synthesize([0, 500, 0], mtype)

    expected_types = [
        SectionType.basal_dendrite,
        SectionType.basal_dendrite,
        SectionType.basal_dendrite,
        SectionType.apical_dendrite,
        SectionType.basal_dendrite,
        SectionType.basal_dendrite,
        SectionType.basal_dendrite
    ]
    assert [i.type for i in result.neuron.root_sections] == expected_types

    assert_array_almost_equal([  # Check only first and last points of neurites
        np.around(np.array([neu.points[0], neu.points[-1]]), 6)
        for neu in result.neuron.root_sections
    ],
    [
        [[0.126557, 9.05724, 1.244288],
         [0.249028, 10.898591, 1.370834]],
        [[-4.150891, 7.862884, -2.131432],
         [-7.36457, 13.100216, -3.620522]],
        [[-9.12136, 0.345625, 0.528383],
         [-14.683837, -0.210737, 1.082018]],
        [[0.0, 9.143187, 0.0],
         [-0.704811, 18.29415, -0.160817]],
        [[-0.94584, -3.288574, 8.47871],
         [-1.874288, -8.647104, 19.25721]],
        [[2.139962, 4.295261, 7.782618],
         [7.751178, 16.725683, 37.513992]],
        [[6.060588, -4.2906, 5.334593],
         [27.177965, -21.629234, 24.54295]]
    ])

    assert_array_equal(
        result.apical_sections, np.array([15])
    )
    assert_array_almost_equal(
        [result.neuron.sections[i].points[-1] for i in result.apical_sections],
        np.array([[-1.54834378, 43.40647507, -1.60015321]])
    )

    # Test with hard limit scale
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
    np.random.seed(0)
    result = context.synthesize([0, 500, 0], mtype)

    assert [i.type for i in result.neuron.root_sections] == expected_types

    assert_array_almost_equal([  # Check only first and last points of neurites
        np.around(np.array([neu.points[0], neu.points[-1]]), 6)
        for neu in result.neuron.root_sections
    ],
    [
        [[0.126557, 9.05724, 1.244288],
         [0.249028, 10.898591, 1.370834]],
        [[-4.150891, 7.862884, -2.131432],
         [-7.36457, 13.100216, -3.620522]],
        [[-9.12136, 0.345625, 0.528383],
         [-14.683837, -0.210737, 1.082018]],
        [[0.0, 9.143187, 0.0],
         [-0.661098, 17.726597, -0.150843]],
        [[-0.94584, -3.288574, 8.47871],
         [-1.874288, -8.647104, 19.25721]],
        [[2.139962, 4.295261, 7.782618],
         [7.751178, 16.725683, 37.513992]],
        [[6.060588, -4.2906, 5.334593],
         [27.177965, -21.629234, 24.54295]]
    ])

    assert_array_equal(
        result.apical_sections, np.array([15])
    )
    assert_array_almost_equal(
        [result.neuron.sections[i].points[-1] for i in result.apical_sections],
        np.array([[-1.4523139, 41.28142929, -1.50091016]])
    )

    # Test scale computation
    params = context.tmd_parameters[mtype]

    assert params["apical"]["modify"] is None
    assert params["basal"]["modify"] is None

    fixed_params = context._correct_position_orientation_scaling(params)

    expected_apical = {"target_path_distance": 76}
    expected_basal = {"reference_thickness": 314, "target_thickness": 300.0}
    assert (
        fixed_params["apical"]["modify"]["kwargs"] == expected_apical
    )
    assert fixed_params["basal"]["modify"]["kwargs"] == expected_basal

    result = context.synthesize([0, 500, 0], mtype)

    assert_array_equal(
        result.apical_sections, np.array([23])
    )
    assert_array_almost_equal(
        [result.neuron.sections[i].points[-1] for i in result.apical_sections],
        np.array([[6.34619808, 39.97901917, -2.28157949]])
    )

    assert_array_almost_equal([  # Check only first and last points of neurites
        np.around(np.array([neu.points[0], neu.points[-1]]), 6)
        for neu in result.neuron.root_sections
    ],
    [
        [[-3.692442, 6.994462, 1.865072],
         [-4.863886, 9.07998, 2.420243]],
        [[-6.711138, -3.087257, -3.38594],
         [-8.628284, -3.758522, -4.406445]],
        [[0.106869, 7.648302, -2.743569],
         [0.168794, 8.93822, -3.23856]],
        [[7.051836, 3.94105, -0.880256],
         [11.152796, 6.810006, -1.316257]],
        [[0., 8.1262, 0.],
         [1.903886, 20.11955, -0.986548]],
        [[3.794402, -4.909248, 5.247564],
         [9.407939, -11.910494, 11.469416]],
        [[-4.378488, -2.390506, 6.414783],
         [-21.02975, -13.127062, 33.190144]],
    ])


def test_debug_scales():
    def capture_logging(names=None):
        # Capture logger entries
        records = []

        class CaptureHandler(logging.Handler):
            def emit(self, record):
                if names is None or record.name == names or record.name in names:
                    records.append(record)

            def __enter__(self):
                logging.getLogger().addHandler(self)
                return records

            def __exit__(self, exc_type, exc_val, exc_tb):
                logging.getLogger().removeHandler(self)

        return CaptureHandler()

    # Test debug logger
    np.random.seed(0)
    mtype = "L2_TPC:A"
    position = [0, 500, 0]

    with TemporaryDirectory() as tempdir:
        small_O1(tempdir)
        context = SpaceContext(
            Atlas.open(tempdir),
            DATA / "distributions.json",
            DATA / "parameters.json",
        )

        # Test with hard limit scale
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
                },
            }
        }

        with capture_logging(names="region_grower.utils") as log:
            context.synthesize(position, mtype)

    expected_messages = (
        [
            f'Neurite type and position: {{"mtype": "L2_TPC:A", "position": {position}}}'
        ]
        + [
            'Default barcode scale: {"max_p": NaN, "reference_thickness": 314, "target_thickness":'
            ' 300.0, "scaling_ratio": 0.9554140127388535}'
        ] * 6
        + [
            'Target barcode scale: {"max_ph": 255.45843100194077, "target_path_distance": 76.0, '
            '"scaling_ratio": 0.2975043716581138}'
        ]
        + [
            f'Neurite hard limit rescaling: {{"neurite_id": {i}, "neurite_type": "basal", '
            '"scale": 1.0, "target_min_length": null, "target_max_length": null}'
            for i in [0, 1, 2]
        ]
        + [
            'Neurite hard limit rescaling: {"neurite_id": 5, "neurite_type": "apical", "scale": '
            '0.9379790269315096, "target_min_length": 70.0, "target_max_length": 70.0}'
        ]
        + [
            f'Neurite hard limit rescaling: {{"neurite_id": {i}, "neurite_type": "basal", '
            '"scale": 1.0, "target_min_length": null, "target_max_length": null}'
            for i in [6, 14, 16]
        ]
    )

    assert [i.msg % i.args for i in log] == expected_messages
