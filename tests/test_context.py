"""Test the region_grower.context module."""
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

    def test_indices_to_positions(self, space_context):
        pos = space_context.indices_to_positions([0, 0, 0])
        assert_array_equal(pos, [-1100.0, -100.0, -1000.0])
        pos = space_context.indices_to_positions([100, 100, 100])
        assert_array_equal(pos, [8900.0, 9900.0, 9000.0])

    def test_positions_to_indices(self, space_context):
        indices = space_context.positions_to_indices([-1100.0, -100.0, -1000.0])
        assert_array_equal(indices, [0, 0, 0])
        indices = space_context.positions_to_indices([8900.0, 9900.0, -9000.0])
        assert_array_equal(indices, [-1, -1, -1])


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
        assert_array_equal(result.apical_sections, np.array([27]))
        assert_array_almost_equal(
            result.apical_points,
            np.array([[8.657612800598145, 267.57452392578125, 8.040230751037598]]),
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
                    [7.0957971, -2.8218994, 0.0],
                    [0.0, 7.6363220, 0.0],
                    [-3.0748143, 6.9899292, 0.0],
                    [-4.8864875, 5.8681946, 0.0],
                    [-7.5141201, -1.3606873, 0.0],
                    [-6.1610136, -3.6210632, 0.0],
                    [-1.2113731, -7.5396423, 0.0],
                    [0.67872918, -7.6061096, 0.0],
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
                    [-2.3479688, -4.855046, 7.627214],
                    [-2.7025747, -5.2889, 8.423989],
                    [-3.0983155, -5.8653774, 9.400189],
                    [-3.426929, -6.1511984, 9.993391],
                    [-3.5540447, -6.3685727, 10.482327],
                ],
                dtype=np.float32,
            ),
        )
        assert_array_almost_equal(
            next(result.neuron.iter()).diameters,
            np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6], dtype=np.float32),
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

        assert_array_equal(result.apical_sections, np.array([27]))
        assert_array_almost_equal(
            result.apical_points,
            np.array([[8.657612800598145, 267.57452392578125, 8.040230751037598]]),
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
                    [-2.3479688, -4.855046, 7.627214],
                    [-2.7025747, -5.2889, 8.423989],
                    [-3.0983155, -5.8653774, 9.400189],
                    [-3.426929, -6.1511984, 9.993391],
                    [-3.5540447, -6.3685727, 10.482327],
                ],
                dtype=np.float32,
            ),
        )
        assert_array_almost_equal(
            next(result.neuron.iter()).diameters,
            np.array(
                [0.8960928, 0.89473546, 0.8929764, 0.89120615, 0.88902414, 0.88768685, 0.8866874],
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
        result = small_context_worker.synthesize()

        expected_types = [
            SectionType.apical_dendrite,
            SectionType.basal_dendrite,
            SectionType.basal_dendrite,
            SectionType.basal_dendrite,
        ]
        assert [i.type for i in result.neuron.root_sections] == expected_types

        assert_array_almost_equal(
            [  # Check only first and last points of neurites
                np.around(np.array([neu.points[0], neu.points[-1]]), 5)
                for neu in result.neuron.root_sections
            ],
            [
                [
                    [0.0, 7.636328, 0.0],
                    [0.040455, 9.585111, -0.106976],
                ],
                [
                    [-1.883677, -3.876395, 6.303874],
                    [-3.987723, -7.272282, 10.785336],
                ],
                [
                    [7.384983, 1.059786, 1.628612],
                    [19.919365, 1.562517, 3.345111],
                ],
                [
                    [-3.335297, 4.929316, 4.784468],
                    [-18.977179, 26.733006, 26.697065],
                ],
            ],
        )

        assert_array_equal(result.apical_sections, np.array([14]))
        assert_array_almost_equal(
            result.apical_points,
            np.array([[-2.841851234436035, 67.96326446533203, 1.485908031463623]]),
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
        result = small_context_worker.synthesize()

        basal_expected_types = [
            SectionType.basal_dendrite,
            SectionType.basal_dendrite,
            SectionType.apical_dendrite,
            SectionType.basal_dendrite,
        ]
        assert [i.type for i in result.neuron.root_sections] == basal_expected_types
        assert_array_almost_equal(
            [  # Check only first and last points of neurites
                np.around(np.array([neu.points[0], neu.points[-1]]), 6)
                for neu in result.neuron.root_sections
            ],
            [
                [
                    [-1.883677, -3.876395, 6.303874],
                    [-3.098316, -5.865378, 9.400189],
                ],
                [
                    [7.384983, 1.059786, 1.628612],
                    [14.03686, 1.485064, 3.365366],
                ],
                [
                    [0.0, 7.636328, 0.0],
                    [0.830408, 13.483992, -0.568914],
                ],
                [
                    [-3.335297, 4.929316, 4.784468],
                    [-13.890615, 18.832005, 19.1254],
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
        result = small_context_worker.synthesize()

        assert [i.type for i in result.neuron.root_sections] == expected_types
        assert_array_almost_equal(
            [  # Check only first and last points of neurites
                np.around(np.array([neu.points[0], neu.points[-1]]), 6)
                for neu in result.neuron.root_sections
            ],
            [
                [
                    [0.0, 7.636328, 0.0],
                    [0.039065, 9.518113, -0.103298],
                ],
                [
                    [-1.883677, -3.876395, 6.303874],
                    [-3.987723, -7.272282, 10.785336],
                ],
                [
                    [7.384983, 1.059786, 1.628612],
                    [19.919365, 1.562517, 3.345111],
                ],
                [
                    [-3.335297, 4.929316, 4.784468],
                    [-18.977179, 26.733006, 26.697065],
                ],
            ],
        )

        assert_array_equal(result.apical_sections, np.array([14]))
        assert_array_almost_equal(
            result.apical_points,
            np.array([[-2.7441508769989014, 65.8892822265625, 1.4348238706588745]]),
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
            result = small_context_worker.synthesize()

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
        result = small_context_worker.synthesize()
        assert [i.type for i in result.neuron.root_sections] == expected_types[1:]

        # test removing basals if scale is too small
        small_context_worker.params.min_hard_scale = 2.2
        result = small_context_worker.synthesize()
        assert [i.type for i in result.neuron.root_sections] == [expected_types[-1]]

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
                        {"max_ph": 78.53590605172101, "scaling_ratio": 0.9554140127388535},
                        {"max_ph": 126.96580088057513, "scaling_ratio": 0.9554140127388535},
                        {"max_ph": 126.96580088057513, "scaling_ratio": 0.9554140127388535},
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
                        {"max_ph": 255.45843100194077, "scaling_ratio": 0.2975043716581138}
                    ],
                },
            },
            "neurite_hard_limit_rescaling": {
                0: {
                    "neurite_type": "apical_dendrite",
                    "scale": 0.9656208571354055,
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

        assert_array_equal(result.apical_sections, np.array([85]))
        assert_array_almost_equal(
            result.apical_points, [[0.14583513140678406, 123.60313415527344, -2.226444959640503]]
        )
