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
from region_grower.context import Y_DIRECTION
from region_grower.context import SpaceWorker
from region_grower.context import SynthesisParameters

from .data_factories import get_tmd_distributions
from .data_factories import get_tmd_parameters

DATA = Path(__file__).parent / "data"


class TestCellState:
    """Test private functions of the CellState class."""

    def test_lookup_orientation(self, cell_state):
        """Test the `lookup_orientation()` method."""
        assert cell_state.lookup_orientation() == Y_DIRECTION
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

    def test_indices_to_positions(self, space_context):
        pos = space_context.indices_to_positions([0, 0, 0])
        assert_array_equal(pos, [-1100.0, -100.0, -1000.0])
        pos = space_context.indices_to_positions([100, 100, 100])
        assert_array_equal(pos, [8900.0, 9900.0, 9000.0])

    def test_positions_to_indices(self, space_context):
        indices = space_context.positions_to_indices([-1100.0, -100.0, -1000.0])
        assert_array_equal(indices, [0, 0, 0])


class TestSpaceWorker:
    """Test private functions of the SpaceWorker class."""

    def test_context(self, cell_state, space_context, synthesis_parameters, computation_parameters):
        """Test the whole context."""
        context_worker = SpaceWorker(
            cell_state,
            space_context,
            synthesis_parameters,
            computation_parameters,
        )

        # Synthesize in L2
        result = context_worker.synthesize()
        assert_array_equal(result.apical_sections, np.array([53]))
        assert_array_almost_equal(
            result.apical_points, np.array([[3.35894, 203.65117, -8.5437]]), decimal=5
        )

        # This tests that input orientations are not mutated by the synthesize() call
        assert (
            list(
                dictdiffer.diff(
                    synthesis_parameters.tmd_parameters["apical_dendrite"]["orientation"],
                    {
                        "mode": "normal_pia_constraint",
                        "values": {"direction": {"mean": 0.0, "std": 0.0}},
                    },
                )
            )
            == []
        )
        assert_array_almost_equal(
            result.neuron.soma.points,
            np.array(
                [
                    [-4.4719658e00, -5.5439525e00, 0.0000000e00],
                    [7.6192994e00, 5.0967985e-01, 0.0000000e00],
                    [2.7471251e00, 7.1250825e00, 0.0000000e00],
                    [4.6759019e-16, 7.6363273e00, 0.0000000e00],
                    [-5.3491087e00, 5.4498196e00, 0.0000000e00],
                    [-7.3899293e00, -1.5310667e00, 0.0000000e00],
                    [-6.1610136e00, -3.6210527e00, 0.0000000e00],
                ],
                dtype=np.float32,
            ),
        )
        assert len(result.neuron.root_sections) == 7
        assert_array_almost_equal(
            next(result.neuron.iter()).points,
            np.array(
                [
                    [-7.3899293, -1.5310667, -1.165452],
                    [-8.706363, -1.8038092, -1.3730644],
                    [-10.090032, -2.006144, -1.4496313],
                ],
                dtype=np.float32,
            ),
        )
        assert_array_almost_equal(
            next(result.neuron.iter()).diameters,
            np.array([0.6, 0.6, 0.6], dtype=np.float32),
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
                    [-5.785500526428223, 4.984130859375, 0.0],
                    [-7.5740227699279785, -0.973480224609375, 0.0],
                    [-1.966903805732727, -7.378662109375, 0.0],
                    [3.565324068069458, -6.7529296875, 0.0],
                    [7.266839027404785, -2.34661865234375, 0.0],
                    [7.384983062744141, 1.059783935546875, 0.0],
                    [6.818241119384766, 3.438751220703125, 0.0],
                    [4.675901919111924e-16, 7.636322021484375, 0.0],
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
            SectionType.basal_dendrite,
            SectionType.apical_dendrite,
            SectionType.basal_dendrite,
            SectionType.basal_dendrite,
            SectionType.basal_dendrite,
            SectionType.basal_dendrite,
            SectionType.basal_dendrite,
        ]
        print(tmd_parameters)
        print([i.type for i in result1.neuron.root_sections])
        assert [i.type for i in result1.neuron.root_sections] == expected_types
        assert_array_almost_equal(
            [  # Check only first and last points of neurites
                np.around(np.array([neu.points[0], neu.points[-1]]), 6)
                for neu in result1.neuron.root_sections
            ],
            [
                [[-7.38993, -1.531067, -1.165452], [-10.090032, -2.006144, -1.449631]],
                [
                    [0.0000000e00, 7.6363282e00, 0.0000000e00],
                    [-1.3458900e-01, 1.0937219e01, -4.8300001e-04],
                ],
                [[-4.471966, -5.543952, -2.753109], [-10.985586, -11.732729, -7.093396]],
                [[-6.161014, -3.621053, -2.691354], [-13.246824, -8.409419, -7.309253]],
                [[2.8443, 5.418238, -4.567948], [6.907934, 20.718624, -14.104201]],
                [[-4.320479, 2.283079, 5.868092], [-14.584823, 10.624202, 20.95233]],
                [[-1.976977, 0.131083, -7.374814], [-7.476844, -0.560946, -41.175533]],
            ],
        )

        assert_array_equal(result1.apical_sections, np.array([29]))
        assert_array_almost_equal(
            result1.apical_points, np.array([[5.711923, 56.423206, 4.124067]])
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
            SectionType.basal_dendrite,
            SectionType.basal_dendrite,
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
                [[-7.38993, -1.531067, -1.165452], [-10.090032, -2.006144, -1.449631]],
                [[-4.471966, -5.543952, -2.753109], [-8.223286, -9.750789, -4.556392]],
                [[-4.320479, 2.283079, 5.868092], [-7.893101, 5.56567, 11.433157]],
                [[2.8443, 5.418238, -4.567948], [5.717189, 11.469478, -8.845925]],
                [[0.0, 7.636328, 0.0], [-0.592816, 17.272392, 0.240599]],
                [[-6.161014, -3.621053, -2.691354], [-14.194309, -7.956396, -5.834008]],
                [[-1.976977, 0.131083, -7.374814], [-5.85143, 2.059288, -21.650974]],
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
                [[-7.38993, -1.531067, -1.165452], [-10.090032, -2.006144, -1.449631]],
                [
                    [0.0000000e00, 7.6363282e00, 0.0000000e00],
                    [-1.2766400e-01, 1.0767367e01, -4.5900000e-04],
                ],
                [[-4.471966, -5.543952, -2.753109], [-10.985586, -11.732729, -7.093396]],
                [[-6.161014, -3.621053, -2.691354], [-13.246824, -8.409419, -7.309253]],
                [[2.8443, 5.418238, -4.567948], [6.907934, 20.718624, -14.104201]],
                [[-4.320479, 2.283079, 5.868092], [-14.584823, 10.624202, 20.95233]],
                [[-1.976977, 0.131083, -7.374814], [-7.476844, -0.560946, -41.175533]],
            ],
        )

        assert_array_equal(result3.apical_sections, np.array([29]))
        assert_array_almost_equal(
            result3.apical_points,
            np.array([[5.418009, 53.912827, 3.911858]]),
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
        assert fixed_params["pia_direction"] == Y_DIRECTION

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
                "target_func": {
                    "inputs": {
                        "fit_slope": 0.5,
                        "fit_intercept": 1,
                        "apical_target_extent": 150.0,
                        "target_path_distance": 76.0,
                        "min_target_path_distance": 1.0,
                    },
                    "scaling": [
                        {"max_ph": 380.2369655856533, "scaling_ratio": 0.19987535899604694}
                    ],
                },
                "default_func": {
                    "inputs": {
                        "target_thickness": 300.0,
                        "reference_thickness": 314,
                        "min_target_thickness": 1.0,
                    },
                    "scaling": [
                        {"max_ph": 161.90961466339775, "scaling_ratio": 0.9554140127388535},
                        {"max_ph": 88.93359927070985, "scaling_ratio": 0.9554140127388535},
                        {"max_ph": 176.74138075182876, "scaling_ratio": 0.9554140127388535},
                        {"max_ph": 143.28779114733433, "scaling_ratio": 0.9554140127388535},
                        {"max_ph": 160.10899907966802, "scaling_ratio": 0.9554140127388535},
                        {"max_ph": 176.74138075182876, "scaling_ratio": 0.9554140127388535},
                    ],
                },
            },
            "neurite_hard_limit_rescaling": {
                19: {
                    "neurite_type": "apical_dendrite",
                    "scale": 0.9485438342260688,
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

        assert_array_equal(result.apical_sections, np.array([81]))
        assert_array_almost_equal(
            result.apical_points, [[-109.65276, 181.05916, -94.54867]], decimal=5
        )
