"""Test the region_grower.context module."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of region-grower.
# See https://github.com/BlueBrain/region-grower for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

# pylint: disable=missing-function-docstring
# pylint: disable=use-implicit-booleaness-not-comparison

from pathlib import Path

import dictdiffer
import numpy as np
import pytest
from morphio import SectionType
from neurots import NeuroTSError
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal

import region_grower
from region_grower import RegionGrowerError
from region_grower import SkipSynthesisError
from region_grower.context import PIA_DIRECTION
from region_grower.context import SpaceWorker
from region_grower.context import SynthesisParameters

from .data_factories import get_tmd_distributions
from .data_factories import get_tmd_parameters

DATA = Path(__file__).parent / "data"


class TestCellState:
    """Test private functions of the CellState class."""

    def test_lookup_orientation(self, cell_state):
        """Test the `lookup_orientation()` method."""
        assert cell_state.lookup_orientation() == PIA_DIRECTION
        vectors = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1 / 3, 1 / 3, 1 / 3],
        ]
        for vector in vectors:
            assert (cell_state.lookup_orientation(vector) == vector).all()


class TestSpaceContext:
    """Test private functions of the SpaceContext class."""

    def test_layer_fraction_to_position(self, space_context):
        """Test the `layer_fraction_to_position()` method."""
        assert space_context.layer_fraction_to_position(2, 0.25) == 275
        assert space_context.layer_fraction_to_position(2, 0.5) == 250
        assert space_context.layer_fraction_to_position(2, 0.75) == 225
        assert space_context.layer_fraction_to_position(6, 0.5) == 700

        # Tests with fractions not in the [0, 1] interval
        assert (
            space_context.layer_fraction_to_position(1, -0.5)
            == space_context.layer_fraction_to_position(2, 0)
            == 300
        )
        assert space_context.layer_fraction_to_position(1, 1.5) == -100
        assert space_context.layer_fraction_to_position(6, -0.5) == 900

    def test_lookup_target_reference_depths(self, cell_state, space_context):
        """Test the `lookup_target_reference_depths()` method."""
        assert space_context.lookup_target_reference_depths(cell_state.depth) == (300, 314)

        with pytest.raises(RegionGrowerError):
            space_context.lookup_target_reference_depths(9999)

    def test_distance_to_constraint(self, space_context):
        """Test the `distance_to_constraint()` method."""
        assert space_context.distance_to_constraint(0, {}) is None
        assert space_context.distance_to_constraint(0, {"layer": 1, "fraction": 1}) == 0
        assert space_context.distance_to_constraint(0, {"layer": 1, "fraction": 0}) == -200
        assert space_context.distance_to_constraint(50, {"layer": 1, "fraction": 0}) == -150
        assert space_context.distance_to_constraint(500, {"layer": 1, "fraction": 0}) == 300
        assert space_context.distance_to_constraint(0, {"layer": 6, "fraction": 0}) == -800
        assert space_context.distance_to_constraint(-100, {"layer": 6, "fraction": 0}) == -900
        assert space_context.distance_to_constraint(1000, {"layer": 6, "fraction": 0}) == 200

        for i in [np.nan, None, []]:
            space_context.layer_depths = i
            assert space_context.distance_to_constraint(1000, {"layer": 6, "fraction": 0}) is None


class TestSpaceWorker:
    """Test private functions of the SpaceWorker class."""

    @pytest.mark.parametrize("recenter", [True, False])
    def test_context(
        self,
        cell_state,
        space_context,
        synthesis_parameters,
        computation_parameters,
        recenter,
    ):
        """Test the whole context."""
        synthesis_parameters.recenter = recenter
        context_worker = SpaceWorker(
            cell_state,
            space_context,
            synthesis_parameters,
            computation_parameters,
        )

        # Synthesize in L2
        result = context_worker.synthesize()
        assert_array_equal(result.apical_sections, np.array([48]))
        assert_array_almost_equal(
            result.apical_points,
            np.array([[20.272340774536133, 266.1555480957031, 6.746281147003174]]),
        )

        # This tests that input orientations are not mutated by the synthesize() call
        assert_array_almost_equal(
            synthesis_parameters.tmd_parameters["apical_dendrite"]["orientation"], [[0.0, 1.0, 0.0]]
        )

        assert_array_almost_equal(
            result.neuron.soma.points,
            np.array(
                [
                    [-5.785500526428223, 4.9841227531433105, 0.0],
                    [-7.5740227699279785, -0.9734848141670227, 0.0],
                    [-1.966903805732727, -7.378671169281006, 0.0],
                    [3.565324068069458, -6.752922534942627, 0.0],
                    [7.266839027404785, -2.346604108810425, 0.0],
                    [7.384983062744141, 1.059786081314087, 0.0],
                    [6.818241119384766, 3.4387624263763428, 0.0],
                    [4.675901919111924e-16, 7.636327266693115, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        assert len(result.neuron.root_sections) == 4
        assert_array_almost_equal(
            next(result.neuron.iter()).points,
            np.array(
                [
                    [-1.8836766, -3.876395, 6.3038735],
                    [-2.0679207, -4.2555485, 6.9204607],
                    [-2.2532284, -4.607763, 7.632429],
                    [-2.4301565, -5.4218144, 8.747698],
                    [-2.7331889, -6.253548, 9.650066],
                    [-2.960198, -6.9949017, 10.46672],
                ],
                dtype=np.float32,
            ),
        )
        assert_array_almost_equal(
            next(result.neuron.iter()).diameters,
            np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6], dtype=np.float32),
        )

    def test_context_external_diametrizer(
        self,
        cell_state,
        space_context,
        computation_parameters,
    ):
        """Test the whole context with an external diametrizer."""
        synthesis_parameters = SynthesisParameters(
            tmd_distributions=get_tmd_distributions(
                DATA / "distributions_external_diametrizer.json"
            )["default"][cell_state.mtype],
            tmd_parameters=get_tmd_parameters(DATA / "parameters_external_diametrizer.json")[
                "default"
            ][cell_state.mtype],
        )

        context_worker = SpaceWorker(
            cell_state,
            space_context,
            synthesis_parameters,
            computation_parameters,
        )

        result = context_worker.synthesize()

        assert_array_equal(result.apical_sections, np.array([48]))
        assert_array_almost_equal(
            result.apical_points,
            np.array([[20.272340774536133, 266.1555480957031, 6.746281147003174]]),
        )

        # This tests that input orientations are not mutated by the synthesize() call
        assert_array_almost_equal(
            synthesis_parameters.tmd_parameters["apical_dendrite"]["orientation"], [[0.0, 1.0, 0.0]]
        )

        assert_array_almost_equal(
            result.neuron.soma.points,
            np.array(
                [
                    [-5.785500526428223, 4.9841227531433105, 0.0],
                    [-7.5740227699279785, -0.9734848141670227, 0.0],
                    [-1.966903805732727, -7.378671169281006, 0.0],
                    [3.565324068069458, -6.752922534942627, 0.0],
                    [7.266839027404785, -2.346604108810425, 0.0],
                    [7.384983062744141, 1.059786081314087, 0.0],
                    [6.818241119384766, 3.4387624263763428, 0.0],
                    [4.675901919111924e-16, 7.636327266693115, 0.0],
                ],
                dtype=np.float32,
            ),
        )

        assert len(result.neuron.root_sections) == 4
        assert_array_almost_equal(
            next(result.neuron.iter()).points,
            np.array(
                [
                    [-1.8836766, -3.876395, 6.3038735],
                    [-2.0679207, -4.2555485, 6.9204607],
                    [-2.2532284, -4.607763, 7.632429],
                    [-2.4301565, -5.4218144, 8.747698],
                    [-2.7331889, -6.253548, 9.650066],
                    [-2.960198, -6.9949017, 10.46672],
                ],
                dtype=np.float32,
            ),
        )
        assert_array_almost_equal(
            next(result.neuron.iter()).diameters,
            np.array(
                [0.76706356, 0.7660498, 0.764943, 0.7630538, 0.76133823, 0.75981],
                dtype=np.float32,
            ),
        )

    def test_scale(self, small_context_worker, tmd_parameters):
        """Test the whole context with scaling."""
        mtype = small_context_worker.cell.mtype

        # Test with no hard limit scaling
        tmd_parameters["default"][mtype]["context_constraints"] = {
            "apical_dendrite": {
                "extent_to_target": {
                    "slope": 0.5,
                    "intercept": 1,
                    "layer": 1,
                    "fraction": 0.5,
                }
            }
        }
        result1 = small_context_worker.synthesize()

        expected_types = [
            SectionType.apical_dendrite,
            SectionType.basal_dendrite,
            SectionType.basal_dendrite,
            SectionType.basal_dendrite,
        ]
        assert [i.type for i in result1.neuron.root_sections] == expected_types

        assert_array_almost_equal(
            [  # Check only first and last points of neurites
                np.around(np.array([neu.points[0], neu.points[-1]]), 6)
                for neu in result1.neuron.root_sections
            ],
            [
                [
                    [0.0, 7.636328220367432, 0.0],
                    [-0.05446799844503403, 9.565113067626953, -0.004176999907940626],
                ],
                [
                    [-1.8836770057678223, -3.8763949871063232, 6.3038740158081055],
                    [-3.030216932296753, -6.87820291519165, 10.731352806091309],
                ],
                [
                    [7.384983062744141, 1.0597859621047974, 1.6286120414733887],
                    [17.74557876586914, 0.2534179985523224, 3.5686450004577637],
                ],
                [
                    [-3.335297107696533, 4.92931604385376, 4.784468173980713],
                    [-19.434040069580078, 27.80714988708496, 24.4342041015625],
                ],
            ],
        )

        assert_array_equal(result1.apical_sections, np.array([25]))
        assert_array_almost_equal(
            result1.apical_points,
            np.array([[-25.359481811523438, 65.18755340576172, -6.167412757873535]]),
        )

        # Test with no hard limit scaling for basal
        tmd_parameters["default"][mtype]["context_constraints"] = {
            "basal_dendrite": {
                "extent_to_target": {
                    "slope": 0.5,
                    "intercept": 1,
                    "layer": 1,
                    "fraction": 0.5,
                }
            }
        }
        result2 = small_context_worker.synthesize()

        basal_expected_types = [
            SectionType.basal_dendrite,
            SectionType.apical_dendrite,
            SectionType.basal_dendrite,
            SectionType.basal_dendrite,
        ]
        assert [i.type for i in result2.neuron.root_sections] == basal_expected_types
        assert_array_almost_equal(
            [  # Check only first and last points of neurites
                np.around(np.array([neu.points[0], neu.points[-1]]), 6)
                for neu in result2.neuron.root_sections
            ],
            [
                [
                    [-1.8836770057678223, -3.8763949871063232, 6.3038740158081055],
                    [-2.4301559925079346, -5.42181396484375, 8.747697830200195],
                ],
                [
                    [0.0, 7.636328220367432, 0.0],
                    [-0.17696000635623932, 10.314826011657715, 0.06906899809837341],
                ],
                [
                    [7.384983062744141, 1.0597859621047974, 1.6286120414733887],
                    [13.206015586853027, 1.6560349464416504, 3.2374720573425293],
                ],
                [
                    [-3.335297107696533, 4.92931604385376, 4.784468173980713],
                    [-10.385510444641113, 19.43812370300293, 19.94303321838379],
                ],
            ],
        )

        # Test with hard limit scale
        tmd_parameters["default"][mtype]["context_constraints"] = {
            "apical_dendrite": {
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
        result3 = small_context_worker.synthesize()

        assert [i.type for i in result3.neuron.root_sections] == expected_types
        assert_array_almost_equal(
            [  # Check only first and last points of neurites
                np.around(np.array([neu.points[0], neu.points[-1]]), 6)
                for neu in result3.neuron.root_sections
            ],
            [
                [
                    [0.0, 7.636328220367432, 0.0],
                    [-0.0517829991877079, 9.470013618469238, -0.003971000202000141],
                ],
                [
                    [-1.8836770057678223, -3.8763949871063232, 6.3038740158081055],
                    [-3.030216932296753, -6.87820291519165, 10.731352806091309],
                ],
                [
                    [7.384983062744141, 1.0597859621047974, 1.6286120414733887],
                    [17.74557876586914, 0.2534179985523224, 3.5686450004577637],
                ],
                [
                    [-3.335297107696533, 4.92931604385376, 4.784468173980713],
                    [-19.434040069580078, 27.80714988708496, 24.4342041015625],
                ],
            ],
        )

        assert_array_equal(result3.apical_sections, np.array([25]))
        assert_array_almost_equal(
            result3.apical_points,
            np.array([[-24.10912322998047, 62.349971771240234, -5.8633270263671875]]),
        )

        # Test scale computation
        params = tmd_parameters["default"][mtype]

        assert params["apical_dendrite"]["modify"] is None
        assert params["basal_dendrite"]["modify"] is None

        # pylint: disable=protected-access
        fixed_params = small_context_worker._correct_position_orientation_scaling(params)

        expected_apical = {"target_path_distance": 76, "with_debug_info": False}
        expected_basal = {
            "reference_thickness": 314,
            "target_thickness": 300.0,
            "with_debug_info": False,
        }
        assert fixed_params["apical_dendrite"]["modify"]["kwargs"] == expected_apical
        assert fixed_params["basal_dendrite"]["modify"]["kwargs"] == expected_basal

        # Test with hard limit scale and min_hard_scale on apical
        tmd_parameters["default"][mtype]["context_constraints"] = {
            "apical_dendrite": {
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
                    "layer": 2,
                    "fraction": 0.6,  # Set max < target to ensure a rescaling is processed
                },
            }
        }
        with pytest.raises(RegionGrowerError):
            small_context_worker.synthesize()

        # Test with hard limit scale and min_hard_scale on basal
        tmd_parameters["default"][mtype]["grow_types"] = ["basal_dendrite"]
        tmd_parameters["default"][mtype]["context_constraints"] = {
            "basal_dendrite": {
                "hard_limit_min": {
                    "layer": 1,
                    "fraction": 0.1,
                },
                "hard_limit_max": {
                    "layer": 2,
                    "fraction": 0.6,  # Set max < target to ensure a rescaling is processed
                },
            }
        }
        small_context_worker.params.seed = 1
        result4 = small_context_worker.synthesize()
        assert [i.type for i in result4.neuron.root_sections] == [SectionType.basal_dendrite] * 5

        params["basal_dendrite"]["orientation"] = {}

        # pylint: disable=protected-access
        fixed_params = small_context_worker._correct_position_orientation_scaling(params)
        assert fixed_params["pia_direction"] == PIA_DIRECTION

    def test_debug_scales(self, small_context_worker, tmd_parameters):
        """Test debug logger."""
        mtype = small_context_worker.cell.mtype
        small_context_worker.internals.debug_data = True

        # Test with hard limit scale
        tmd_parameters["default"][mtype]["context_constraints"] = {
            "apical_dendrite": {
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

        small_context_worker.synthesize()

        expected_debug_infos = {
            "input_scaling": {
                "default_func": {
                    "inputs": {
                        "target_thickness": 300.0,
                        "reference_thickness": 314,
                        "min_target_thickness": 1.0,
                    },
                    "scaling": [
                        {
                            "max_ph": 126.96580088057513,
                            "scaling_ratio": 0.9554140127388535,
                        },
                        {
                            "max_ph": 139.93140451880743,
                            "scaling_ratio": 0.9554140127388535,
                        },
                        {
                            "max_ph": 143.28779114733433,
                            "scaling_ratio": 0.9554140127388535,
                        },
                    ],
                },
                "target_func": {
                    "inputs": {
                        "fit_slope": 0.5,
                        "fit_intercept": 1,
                        "apical_target_extent": 150.0,
                        "target_path_distance": 76.0,
                        "min_target_path_distance": 1.0,
                    },
                    "scaling": [
                        {
                            "max_ph": 420.70512274498446,
                            "scaling_ratio": 0.18064909574697127,
                        }
                    ],
                },
            },
            "neurite_hard_limit_rescaling": {
                0: {
                    "neurite_type": "apical_dendrite",
                    "scale": 0.9506947194040978,
                    "target_min_length": 70.0,
                    "target_max_length": 70.0,
                    "deleted": False,
                }
            },
        }
        assert (
            list(
                dictdiffer.diff(
                    small_context_worker.debug_infos, expected_debug_infos, tolerance=1e-6
                )
            )
            == []
        )

    def test_transform_morphology(self, small_context_worker, morph_loader, cell_orientation):
        """Test the `transform_morphology()` method."""
        np.random.seed(0)
        morph = morph_loader.get("C170797A-P1")
        assert_array_almost_equal(
            morph.root_sections[0].points[-1],
            [8.177688, -129.37207, 10.289684],
        )

        small_context_worker.transform_morphology(morph, cell_orientation)

        assert len(morph.root_sections) == 8
        assert len(morph.sections) == 52
        assert_array_almost_equal(
            morph.root_sections[0].points[-1],
            [10.902711, -129.37207, 7.340509],
        )

    def test_retry(
        self,
        cell_state,
        space_context,
        synthesis_parameters,
        computation_parameters,
        mocker,
    ):
        """Test the `retry` feature."""
        context_worker = SpaceWorker(
            cell_state,
            space_context,
            synthesis_parameters,
            computation_parameters,
        )

        mock = mocker.patch.object(
            region_grower.context.modify,
            "output_scaling",
            side_effect=[NeuroTSError, mocker.DEFAULT],
            return_value=1.0,
        )

        # Synthesize once
        with pytest.raises(SkipSynthesisError):
            result = context_worker.synthesize()

        # Synthesize 3 times
        mock.side_effect = [NeuroTSError] * 2 + [mocker.DEFAULT] * 7
        context_worker.internals.retries = 3
        result = context_worker.synthesize()

        assert_array_equal(result.apical_sections, np.array([72]))
        assert_array_almost_equal(
            result.apical_points, [[-74.1576919555664, 241.61322021484375, -26.836679458618164]]
        )
