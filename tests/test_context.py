"""Test the region_grower.context module."""
# pylint: disable=missing-function-docstring
# pylint: disable=no-self-use
# pylint: disable=protected-access
from pathlib import Path

import dictdiffer
import numpy as np
import pytest
from morphio import SectionType
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal

from region_grower import RegionGrowerError
from region_grower import SkipSynthesisError
from region_grower.context import PIA_DIRECTION
from region_grower.context import SpaceWorker
from region_grower.context import SynthesisParameters
from region_grower.synthesize_morphologies import SynthesizeMorphologies

from .data_factories import get_tmd_distributions
from .data_factories import get_tmd_parameters

DATA = Path(__file__).parent / "data"


class TestCellState:
    """Test private functions of the CellState class."""

    def test_lookup_orientation(self, cell_state):
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
        assert space_context.lookup_target_reference_depths(cell_state.depth) == (300, 314)

        with pytest.raises(RegionGrowerError):
            space_context.lookup_target_reference_depths(9999)

    def test_distance_to_constraint(self, space_context):
        assert space_context.distance_to_constraint(0, {}) is None
        assert space_context.distance_to_constraint(0, {"layer": 1, "fraction": 1}) == 0
        assert space_context.distance_to_constraint(0, {"layer": 1, "fraction": 0}) == -200
        assert space_context.distance_to_constraint(50, {"layer": 1, "fraction": 0}) == -150
        assert space_context.distance_to_constraint(500, {"layer": 1, "fraction": 0}) == 300
        assert space_context.distance_to_constraint(0, {"layer": 6, "fraction": 0}) == -800
        assert space_context.distance_to_constraint(-100, {"layer": 6, "fraction": 0}) == -900
        assert space_context.distance_to_constraint(1000, {"layer": 6, "fraction": 0}) == 200


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
        synthesis_parameters.recenter = recenter
        context_worker = SpaceWorker(
            cell_state,
            space_context,
            synthesis_parameters,
            computation_parameters,
        )

        # Synthesize in L2
        result = context_worker.synthesize()

        assert_array_equal(result.apical_sections, np.array([98]))
        assert_array_almost_equal(
            result.apical_points,
            np.array([[19.62138748, 353.8174438, 12.29593754]]),
        )

        # This tests that input orientations are not mutated by the synthesize() call
        assert_array_almost_equal(
            synthesis_parameters.tmd_parameters["apical"]["orientation"], [[0.0, 1.0, 0.0]]
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
                    [-1.8836766481399536, -3.8763949871063232, 6.303873538970947],
                    [-2.067920684814453, -4.255548477172852, 6.9204607009887695],
                    [-2.2532284259796143, -4.607762813568115, 7.632429122924805],
                    [-2.4301564693450928, -5.421814441680908, 8.747697830200195],
                    [-2.7331888675689697, -6.2535481452941895, 9.650066375732422],
                    [-2.960197925567627, -6.994901657104492, 10.466719627380371],
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
        synthesis_parameters = SynthesisParameters(
            tmd_distributions=get_tmd_distributions(
                DATA / "distributions_external_diametrizer.json"
            )["mtypes"][cell_state.mtype],
            tmd_parameters=get_tmd_parameters(DATA / "parameters_external_diametrizer.json")[
                cell_state.mtype
            ],
        )

        context_worker = SpaceWorker(
            cell_state,
            space_context,
            synthesis_parameters,
            computation_parameters,
        )

        result = context_worker.synthesize()

        assert_array_equal(result.apical_sections, np.array([98]))
        assert_array_almost_equal(
            result.apical_points,
            np.array([[19.62138748, 353.8174438, 12.29593754]]),
        )

        # This tests that input orientations are not mutated by the synthesize() call
        assert_array_almost_equal(
            synthesis_parameters.tmd_parameters["apical"]["orientation"], [[0.0, 1.0, 0.0]]
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
                    [-1.8836766481399536, -3.8763949871063232, 6.303873538970947],
                    [-2.067920684814453, -4.255548477172852, 6.9204607009887695],
                    [-2.2532284259796143, -4.607762813568115, 7.632429122924805],
                    [-2.4301564693450928, -5.421814441680908, 8.747697830200195],
                    [-2.7331888675689697, -6.2535481452941895, 9.650066375732422],
                    [-2.960197925567627, -6.994901657104492, 10.466719627380371],
                ],
                dtype=np.float32,
            ),
        )

        assert_array_almost_equal(
            next(result.neuron.iter()).diameters,
            np.array(
                [
                    0.8767099380493164,
                    0.8752812147140503,
                    0.8737207651138306,
                    0.8710576295852661,
                    0.868639349937439,
                    0.8664852380752563,
                ],
                dtype=np.float32,
            ),
        )

    def test_scale(self, small_context_worker, tmd_parameters, tmd_distributions):
        mtype = small_context_worker.cell.mtype

        # Test with no hard limit scaling
        tmd_parameters[mtype]["context_constraints"] = {
            "apical": {
                "extent_to_target": {
                    "slope": 0.5,
                    "intercept": 1,
                    "layer": 1,
                    "fraction": 0.5,
                }
            }
        }
        SynthesizeMorphologies.verify([mtype], tmd_distributions, tmd_parameters)
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
                np.around(np.array([neu.points[0], neu.points[-1]]), 6)
                for neu in result.neuron.root_sections
            ],
            [
                [
                    [0.0, 7.636328220367432, 0.0],
                    [-0.05446799844503403, 9.565113067626953, -0.004176999907940626],
                ],
                [
                    [-1.8836770057678223, -3.8763949871063232, 6.3038740158081055],
                    [-2.8858020305633545, -6.487758159637451, 10.838364601135254],
                ],
                [
                    [7.384983062744141, 1.0597859621047974, 1.6286120414733887],
                    [18.221834182739258, 0.030551999807357788, 2.9156060218811035],
                ],
                [
                    [-3.335297107696533, 4.92931604385376, 4.784468173980713],
                    [-21.92222785949707, 27.993545532226562, 22.405996322631836],
                ],
            ],
        )

        assert_array_equal(result.apical_sections, np.array([72]))
        assert_array_almost_equal(
            result.apical_points,
            np.array([[6.208977699279785, 59.538753509521484, -5.052260875701904]]),
        )

        # Test with hard limit scale
        tmd_parameters[mtype]["context_constraints"] = {
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
        SynthesizeMorphologies.verify([mtype], tmd_distributions, tmd_parameters)
        result = small_context_worker.synthesize()

        assert [i.type for i in result.neuron.root_sections] == expected_types

        assert_array_almost_equal(
            [  # Check only first and last points of neurites
                np.around(np.array([neu.points[0], neu.points[-1]]), 6)
                for neu in result.neuron.root_sections
            ],
            [
                [
                    [0.0, 7.636328220367432, 0.0],
                    [-0.05270500108599663, 9.502668380737305, -0.0040420000441372395],
                ],
                [
                    [-1.8836770057678223, -3.8763949871063232, 6.3038740158081055],
                    [-2.8858020305633545, -6.487758159637451, 10.838364601135254],
                ],
                [
                    [7.384983062744141, 1.0597859621047974, 1.6286120414733887],
                    [18.221834182739258, 0.030551999807357788, 2.9156060218811035],
                ],
                [
                    [-3.335297107696533, 4.92931604385376, 4.784468173980713],
                    [-21.92222785949707, 27.993545532226562, 22.405996322631836],
                ],
            ],
        )

        assert_array_equal(result.apical_sections, np.array([72]))
        assert_array_almost_equal(
            result.apical_points,
            np.array([[6.007961273193359, 57.8583984375, -4.888692855834961]]),
        )

        # Test scale computation
        params = tmd_parameters[mtype]

        assert params["apical"]["modify"] is None
        assert params["basal"]["modify"] is None

        # pylint: disable=protected-access
        fixed_params = small_context_worker._correct_position_orientation_scaling(params)

        expected_apical = {"target_path_distance": 76, "with_debug_info": False}
        expected_basal = {
            "reference_thickness": 314,
            "target_thickness": 300.0,
            "with_debug_info": False,
        }
        assert fixed_params["apical"]["modify"]["kwargs"] == expected_apical
        assert fixed_params["basal"]["modify"]["kwargs"] == expected_basal

        # Test with hard limit scale and min_hard_scale on apical
        tmd_parameters[mtype]["context_constraints"] = {
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
                    "layer": 2,
                    "fraction": 0.6,  # Set max < target to ensure a rescaling is processed
                },
            }
        }
        SynthesizeMorphologies.verify([mtype], tmd_distributions, tmd_parameters)
        with pytest.raises(RegionGrowerError):
            result = small_context_worker.synthesize()

        # Test with hard limit scale and min_hard_scale on basal
        tmd_parameters[mtype]["grow_types"] = ["basal"]
        tmd_parameters[mtype]["context_constraints"] = {
            "basal": {
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
        SynthesizeMorphologies.verify([mtype], tmd_distributions, tmd_parameters)
        result = small_context_worker.synthesize()
        assert [i.type for i in result.neuron.root_sections] == [expected_types[-1]]

    def test_debug_scales(self, small_context_worker, tmd_parameters):
        # Test debug logger
        mtype = small_context_worker.cell.mtype
        small_context_worker.internals.debug_data = True

        # Test with hard limit scale
        tmd_parameters[mtype]["context_constraints"] = {
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
                            "max_p": np.nan,
                            "reference_thickness": 314,
                            "target_thickness": 300.0,
                            "scaling_ratio": 0.9554140127388535,
                        },
                        {
                            "max_p": np.nan,
                            "reference_thickness": 314,
                            "target_thickness": 300.0,
                            "scaling_ratio": 0.9554140127388535,
                        },
                        {
                            "max_p": np.nan,
                            "reference_thickness": 314,
                            "target_thickness": 300.0,
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
                            "target_path_distance": 76.0,
                            "scaling_ratio": 0.18064909574697127,
                        }
                    ],
                },
            },
            "neurite_hard_limit_rescaling": {
                0: {
                    "neurite_type": "apical",
                    "scale": 0.9676248059345621,
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

    def test_load_morphology(self, small_context_worker, morph_loader):
        small_context_worker.internals.morph_loader = morph_loader

        with pytest.raises(SkipSynthesisError):
            small_context_worker.load_morphology("UNKNOWN")

        res = small_context_worker.load_morphology("C170797A-P1")
        assert len(res.root_sections) == 8
        assert len(res.sections) == 52

    def test_transform_morphology(self, small_context_worker, morph_loader, cell_orientation):
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
