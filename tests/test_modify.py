"""Test the region_grower.modify module."""
# pylint: disable=missing-function-docstring
# pylint: disable=no-self-use
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal

from region_grower import RegionGrowerError
from region_grower import modify


def test_scale_default_barcode():
    ph = [[0.1, 0.2], [0.3, 0.4]]
    reference_thickness = 1
    target_thickness = 1
    res = modify.scale_default_barcode(ph, None, reference_thickness, target_thickness)
    assert_array_equal(res, ph)

    res = modify.scale_default_barcode(
        np.array(ph) * 10, None, reference_thickness, target_thickness
    )
    assert_array_equal(res, [[0.25, 0.5], [0.75, 1.0]])


def test_scale_target_barcode():
    ph = [[0.1, 0.2], [0.3, 0.4]]
    target_path_distance = 1
    res = modify.scale_target_barcode(ph, None, target_path_distance)
    assert_array_almost_equal(res, [[1 / 3, 2 / 3], [1, 4 / 3]])


def test_input_scaling():
    init_params = {
        "grow_types": ["apical", "basal"],
        "context_constraints": {
            "apical": {
                "extent_to_target": {
                    "slope": 0.5,
                    "intercept": 1,
                    "layer": 1,
                    "fraction": 0.5,
                }
            }
        },
        "apical": {},
        "basal": {},
    }
    reference_thickness = 1
    target_thickness = 2
    apical_target_distance = 3

    expected = deepcopy(init_params)
    expected["apical"] = {
        "modify": {
            "funct": modify.scale_target_barcode,
            "kwargs": {
                "target_path_distance": 2.5,
                "with_debug_info": False,
            },
        }
    }
    expected["basal"] = {
        "modify": {
            "funct": modify.scale_default_barcode,
            "kwargs": {
                "target_thickness": target_thickness,
                "reference_thickness": reference_thickness,
                "with_debug_info": False,
            },
        }
    }

    params = deepcopy(init_params)
    modify.MIN_TARGET_PATH_DISTANCE = 2
    modify.input_scaling(
        params,
        reference_thickness,
        target_thickness,
        apical_target_distance,
    )

    assert params.get("apical", {}) == expected["apical"]
    assert params.get("basal", {}) == expected["basal"]

    with pytest.raises(RegionGrowerError):
        modify.input_scaling(
            params,
            reference_thickness,
            0,
            apical_target_distance,
        )

    params = deepcopy(init_params)
    params["context_constraints"]["apical"]["extent_to_target"]["slope"] = -0.5
    with pytest.raises(RegionGrowerError):
        modify.input_scaling(
            params,
            reference_thickness,
            10,
            apical_target_distance,
        )


class TestOutputScaling:
    """Test the modify.output_scaling() function."""

    @pytest.fixture(scope="class")
    def root_sec(self, synthesized_cell):
        yield synthesized_cell.neuron.root_sections[0]

    def test_output_scaling_default(self, root_sec):
        assert modify.output_scaling(root_sec, [0, 1, 0], None, None) == 1

    def test_output_scaling_useless_min(self, root_sec):
        assert modify.output_scaling(root_sec, [0, 1, 0], 1, None) == 1

    def test_output_scaling_min(self, root_sec):
        assert modify.output_scaling(root_sec, [0, 1, 0], 9.76777, None) == pytest.approx(1.2)

    def test_output_scaling_useless_max(self, root_sec):
        assert modify.output_scaling(root_sec, [0, 1, 0], None, 10) == 1

    def test_output_scaling_max(self, root_sec):
        assert modify.output_scaling(root_sec, [0, 1, 0], None, 6.51185) == pytest.approx(0.8)

    def test_output_scaling_useless_min_useless_max(self, root_sec):
        assert modify.output_scaling(root_sec, [0, 1, 0], 1, 10) == 1

    def test_output_scaling_useless_min_max(self, root_sec):
        assert modify.output_scaling(root_sec, [0, 1, 0], 1, 6.51185) == pytest.approx(0.8)

    def test_output_scaling_min_max(self, root_sec):
        assert modify.output_scaling(root_sec, [0, 1, 0], 9.76777, 6.51185) == pytest.approx(1.2)
