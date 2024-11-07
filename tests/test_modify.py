"""Test the region_grower.modify module."""

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
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal

from region_grower import RegionGrowerError
from region_grower import modify


def test_scale_default_barcode():
    """Test the `scale_default_barcode()` function."""
    ph = [[0.1, 0.2], [0.3, 0.4]]
    reference_thickness = 1
    target_thickness = 1
    res = modify.scale_default_barcode(ph, None, reference_thickness, target_thickness)
    assert_array_equal(res, ph)

    res = modify.scale_default_barcode(
        np.array(ph) * 10, None, reference_thickness, target_thickness
    )
    assert_array_equal(res, [[1, 2], [3, 4]])

    reference_thickness = 2.0
    target_thickness = 0.5
    res = modify.scale_default_barcode(ph, None, reference_thickness, target_thickness)
    assert_array_equal(res, [[0.025, 0.05], [0.075, 0.1]])


def test_scale_target_barcode():
    """Test the `scale_target_barcode()` function."""
    ph = [[0.1, 0.2], [0.3, 0.4]]
    target_path_distance = 1
    res = modify.scale_target_barcode(ph, None, target_path_distance)
    assert_array_almost_equal(res, [[1 / 3, 2 / 3], [1, 4 / 3]])


def test_input_scaling():
    """Test the `input_scaling()` function."""
    init_params = {
        "grow_types": ["apical_dendrite", "basal_dendrite"],
        "context_constraints": {
            "apical_dendrite": {
                "extent_to_target": {
                    "slope": 0.5,
                    "intercept": 1,
                    "layer": 1,
                    "fraction": 0.5,
                }
            }
        },
        "apical_dendrite": {},
        "basal_dendrite": {},
    }
    reference_thickness = 1
    target_thickness = 2
    apical_target_distance = 3

    expected = deepcopy(init_params)
    expected["apical_dendrite"] = {
        "modify": {
            "funct": modify.scale_target_barcode,
            "kwargs": {
                "target_path_distance": 2.5,
                "with_debug_info": False,
            },
        }
    }
    expected["basal_dendrite"] = {
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

    assert params.get("apical_dendrite", {}) == expected["apical_dendrite"]
    assert params.get("basal_dendrite", {}) == expected["basal_dendrite"]

    with pytest.raises(RegionGrowerError):
        modify.input_scaling(
            params,
            reference_thickness,
            0,
            apical_target_distance,
        )

    params = deepcopy(init_params)
    params["context_constraints"]["apical_dendrite"]["extent_to_target"]["slope"] = -0.5
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
        """The root section used for tests."""
        yield synthesized_cell.neuron.root_sections[0]

    def test_output_scaling_default(self, root_sec):
        """Test with default arguments."""
        assert modify.output_scaling(root_sec, [0, 1, 0], None, None) == 1

    def test_output_scaling_useless_min(self, root_sec):
        """Test with a min value that is smaller than all values (useless)."""
        assert modify.output_scaling(root_sec, [0, 1, 0], 1, None) == 1

    def test_output_scaling_min(self, root_sec):
        """Test with a min value that is greater than a few values (useful)."""
        assert modify.output_scaling(root_sec, [0, 1, 0], 40.657597, None) == pytest.approx(
            1.7270042868523088
        )

    def test_output_scaling_useless_max(self, root_sec):
        """Test with a max value that is greater than all values (useless)."""
        assert modify.output_scaling(root_sec, [0, 1, 0], None, 1000) == 1

    def test_output_scaling_max(self, root_sec):
        """Test with a max value that is greater than a few values (useful)."""
        assert modify.output_scaling(root_sec, [0, 1, 0], None, 27.105065) == pytest.approx(1)

    def test_output_scaling_useless_min_useless_max(self, root_sec):
        """Test with an useless min value and useless max values."""
        assert modify.output_scaling(root_sec, [0, 1, 0], 1, 1000) == 1

    def test_output_scaling_useless_min_max(self, root_sec):
        """Test with an useless min value and useful max values."""
        assert modify.output_scaling(root_sec, [0, 1, 0], 1, 27.105065) == pytest.approx(1)

    def test_output_scaling_min_max(self, root_sec):
        """Test with an useful min value and useful max values."""
        assert modify.output_scaling(root_sec, [0, 1, 0], 40.657597, 27.105065) == pytest.approx(
            1.7270042868523088
        )
