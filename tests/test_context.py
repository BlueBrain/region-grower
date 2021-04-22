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

        assert_array_equal(result.apical_sections, np.array([48]))
        assert_array_almost_equal(
            result.apical_points,
            np.array([[9.40834045, 114.9850235, -25.60334587]]),
        )

        # This tests that input orientations are not mutated by the synthesize() call
        assert_array_almost_equal(
            synthesis_parameters.tmd_parameters["apical"]["orientation"], [[0.0, 1.0, 0.0]]
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

        assert len(result.neuron.root_sections) == 7
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

        assert_array_equal(result.apical_sections, np.array([48]))
        assert_array_almost_equal(
            result.apical_points,
            np.array([[9.40834045, 114.9850235, -25.60334587]]),
        )

        # This tests that input orientations are not mutated by the synthesize() call
        assert_array_almost_equal(
            synthesis_parameters.tmd_parameters["apical"]["orientation"], [[0.0, 1.0, 0.0]]
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

        assert len(result.neuron.root_sections) == 7
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
            SectionType.basal_dendrite,
            SectionType.basal_dendrite,
            SectionType.basal_dendrite,
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
                [[0.126557, 9.05724, 1.244288], [0.249028, 10.898591, 1.370834]],
                [[-4.150891, 7.862884, -2.131432], [-7.36457, 13.100216, -3.620522]],
                [[-9.12136, 0.345625, 0.528383], [-14.683837, -0.210737, 1.082018]],
                [[0.0, 9.143187, 0.0], [-0.704811, 18.29415, -0.160817]],
                [[-0.94584, -3.288574, 8.47871], [-1.874288, -8.647104, 19.25721]],
                [[2.139962, 4.295261, 7.782618], [7.751178, 16.725683, 37.513992]],
                [[6.060588, -4.2906, 5.334593], [27.177965, -21.629234, 24.54295]],
            ],
        )

        assert_array_equal(result.apical_sections, np.array([15]))
        assert_array_almost_equal(
            result.apical_points,
            np.array([[-1.54834378, 43.40647507, -1.60015321]]),
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
                [[0.126557, 9.05724, 1.244288], [0.249028, 10.898591, 1.370834]],
                [[-4.150891, 7.862884, -2.131432], [-7.36457, 13.100216, -3.620522]],
                [[-9.12136, 0.345625, 0.528383], [-14.683837, -0.210737, 1.082018]],
                [[0.0, 9.143187, 0.0], [-0.661098, 17.726597, -0.150843]],
                [[-0.94584, -3.288574, 8.47871], [-1.874288, -8.647104, 19.25721]],
                [[2.139962, 4.295261, 7.782618], [7.751178, 16.725683, 37.513992]],
                [[6.060588, -4.2906, 5.334593], [27.177965, -21.629234, 24.54295]],
            ],
        )

        assert_array_equal(result.apical_sections, np.array([15]))
        assert_array_almost_equal(
            result.apical_points,
            np.array([[-1.4523139, 41.28142929, -1.50091016]]),
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
                        "min_target_thickness": 1.0,
                        "reference_thickness": 314,
                        "target_thickness": 300.0,
                    },
                    "scaling": [
                        {
                            "max_p": np.nan,
                            "reference_thickness": 314,
                            "scaling_ratio": 0.9554140127388535,
                            "target_thickness": 300.0,
                        },
                        {
                            "max_p": np.nan,
                            "reference_thickness": 314,
                            "scaling_ratio": 0.9554140127388535,
                            "target_thickness": 300.0,
                        },
                        {
                            "max_p": np.nan,
                            "reference_thickness": 314,
                            "scaling_ratio": 0.9554140127388535,
                            "target_thickness": 300.0,
                        },
                        {
                            "max_p": np.nan,
                            "reference_thickness": 314,
                            "scaling_ratio": 0.9554140127388535,
                            "target_thickness": 300.0,
                        },
                        {
                            "max_p": np.nan,
                            "reference_thickness": 314,
                            "scaling_ratio": 0.9554140127388535,
                            "target_thickness": 300.0,
                        },
                        {
                            "max_p": np.nan,
                            "reference_thickness": 314,
                            "scaling_ratio": 0.9554140127388535,
                            "target_thickness": 300.0,
                        },
                    ],
                },
                "target_func": {
                    "inputs": {
                        "apical_target_extent": 150.0,
                        "fit_intercept": 1,
                        "fit_slope": 0.5,
                        "min_target_path_distance": 1.0,
                        "target_path_distance": 76.0,
                    },
                    "scaling": [
                        {
                            "max_ph": 255.45843100194077,
                            "target_path_distance": 76.0,
                            "scaling_ratio": 0.2975043716581138,
                        }
                    ],
                },
            },
            "neurite_hard_limit_rescaling": {
                5: {
                    "neurite_type": "apical",
                    "scale": 0.9379790269315096,
                    "target_max_length": 70.0,
                    "target_min_length": 70.0,
                },
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
