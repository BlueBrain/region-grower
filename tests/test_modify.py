from copy import deepcopy

import numpy as np
from nose.tools import assert_raises
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

from region_grower import modify
from region_grower import RegionGrowerError


def test_scale_default_barcode():
    ph = [[0.1, 0.2], [0.3, 0.4]]
    reference_thickness = 1
    target_thickness = 1
    res = modify.scale_default_barcode(
        ph, reference_thickness, target_thickness)
    assert_array_equal(res, ph)

    res = modify.scale_default_barcode(
        np.array(ph) * 10, reference_thickness, target_thickness)
    assert_array_equal(res, [[0.25, 0.5], [0.75, 1.0]])


def test_scale_target_barcode():
    ph = [[0.1, 0.2], [0.3, 0.4]]
    target_path_distance = 1
    res = modify.scale_target_barcode(ph, target_path_distance)
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
        "basal": {}
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
            },
        }
    }
    expected["basal"] = {
        "modify": {
            "funct": modify.scale_default_barcode,
            "kwargs": {
                "target_thickness": target_thickness,
                "reference_thickness": reference_thickness,
            },
        }
    }

    params = deepcopy(init_params)
    modify.input_scaling(
        params,
        reference_thickness,
        target_thickness,
        apical_target_distance,
    )

    assert params.get("apical", {}) == expected["apical"]
    assert params.get("basal", {}) == expected["basal"]

    assert_raises(
        RegionGrowerError,
        modify.input_scaling,
        params,
        reference_thickness,
        0,
        apical_target_distance,
    )

    params = deepcopy(init_params)
    params["context_constraints"]["apical"]["extent_to_target"]["slope"] = -0.5
    assert_raises(
        ValueError,
        modify.input_scaling,
        params,
        reference_thickness,
        10,
        apical_target_distance,
    )
