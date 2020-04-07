""" Use spatial properties to modify synthesis input."""

import numpy as np
from tmd.Topology.transformations import tmd_scale

from region_grower import RegionGrowerError


def scale_barcode(ph, reference_thickness, target_thickness):
    """Modifies the barcode according to the reference and target thicknesses.
       Reference thickness defines the property of input data.
       Target thickness defines the property of space, which should be used by synthesis.
    """
    max_p = np.max(ph)
    scaling_reference = 1.0
    # If cell is larger than the reference thickness it should be scaled down
    # This ensures that the cell will not grow more than the target thickness
    if 1 - max_p / reference_thickness < 0:
        scaling_reference = reference_thickness / max_p

    return tmd_scale(ph, scaling_reference * target_thickness / reference_thickness)


def scale_bias(bias_length, reference_thickness, target_thickness):
    """Scales length according to reference and target thickness"""
    return bias_length * target_thickness / reference_thickness


def input_scaling(params, reference_thickness, target_thickness):
    """Modifies the input parameters to match the input data
       taken from the spatial properties of the Atlas:
       The reference_thicness is the expected thickness of input data
       The target_thickness is the expected thickness that the synthesized
       cells should live in. Input should be modified accordingly
       All neurite types are scaled uniformly. This should be corrected eventually.
    """
    if target_thickness < 1e-8:
        raise RegionGrowerError(
            "target_thickness too small to be able to scale the bar code"
        )

    par = dict(params)

    for neurite_type in params["grow_types"]:
        par[neurite_type].update(
            {
                "modify": {
                    "funct": scale_barcode,
                    "kwargs": {
                        "target_thickness": target_thickness,
                        "reference_thickness": reference_thickness,
                    },
                }
            }
        )
        if "bias_length" in par[neurite_type]:
            par[neurite_type]["bias_length"] = scale_bias(
                par[neurite_type]["bias_length"], reference_thickness, target_thickness
            )
    return par
