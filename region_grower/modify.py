""" Use spatial properties to modify synthesis input."""

from typing import Dict
from typing import List
from typing import Optional

import numpy as np
from morphio import Section
from neurom import COLS
from tmd.Topology.transformations import tmd_scale

from region_grower import RegionGrowerError

# if the given variables are < these values, a hard crash will happen
MIN_TARGET_PATH_DISTANCE = 1.0
MIN_TARGET_THICKNESS = 1.0


def scale_default_barcode(
    persistent_homologies: List[float],
    context,
    reference_thickness: float,
    target_thickness: float,
    with_debug_info: bool = False,
):
    """Modifies the barcode according to the reference and target thicknesses.
    Reference thickness defines the property of input data.
    Target thickness defines the property of space, which should be used by synthesis.
    """
    max_p = np.max(persistent_homologies)
    scaling_reference = np.nan_to_num(reference_thickness / max_p, nan=np.inf)
    # If cell is larger than the reference thickness it should be scaled down
    # This ensures that the cell will not grow more than the target thickness
    scaling_reference = min(scaling_reference, 1.0)

    scaling_ratio = scaling_reference * target_thickness / reference_thickness

    if with_debug_info:
        context["debug_data"]["default_func"]["scaling"].append(
            {
                "max_p": max_p,
                "reference_thickness": reference_thickness,
                "target_thickness": target_thickness,
                "scaling_ratio": scaling_ratio,
            }
        )

    return tmd_scale(persistent_homologies, scaling_ratio)


def scale_target_barcode(
    persistent_homologies: List[float],
    context,
    target_path_distance: float,
    with_debug_info: bool = False,
):
    """Modifies the barcode according to the target thicknesses.
    Target thickness defines the total extend at which the cell can grow.
    """
    max_ph = np.nanmax([i[0] for i in persistent_homologies])
    scaling_ratio = target_path_distance / max_ph

    if with_debug_info:
        context["debug_data"]["target_func"]["scaling"].append(
            {
                "max_ph": max_ph,
                "target_path_distance": target_path_distance,
                "scaling_ratio": scaling_ratio,
            }
        )

    return tmd_scale(persistent_homologies, scaling_ratio)


def input_scaling(
    params: Dict,
    reference_thickness: float,
    target_thickness: float,
    apical_target_extent: Optional[float],
    debug_info: Optional[Dict] = None,
):
    """Modifies the input parameters so that grown cells fit into the volume

    If expected_apical_size is not None, the apical scaling uses a different scaling
    than the rest of the dendrites.

    Args:
        params: the param dictionary that this function will modify
        reference_thicness: the expected thickness of input data
        target_thickness: the expected thickness that the synthesized cells should live in.
        apical_target_extent: the expected extent of the apical dendrite
    """
    for neurite_type in params["grow_types"]:
        if neurite_type == "apical" and apical_target_extent is not None:
            apical_constraint = params["context_constraints"]["apical"]["extent_to_target"]
            linear_fit = np.poly1d((apical_constraint["slope"], apical_constraint["intercept"]))
            target_path_distance = linear_fit(apical_target_extent)
            if debug_info is not None:
                debug_info.update(
                    {
                        "target_func": {
                            "inputs": {
                                "fit_slope": apical_constraint["slope"],
                                "fit_intercept": apical_constraint["intercept"],
                                "apical_target_extent": apical_target_extent,
                                "target_path_distance": target_path_distance,
                                "min_target_path_distance": MIN_TARGET_PATH_DISTANCE,
                            },
                            "scaling": [],
                        }
                    }
                )
            if target_path_distance < MIN_TARGET_PATH_DISTANCE:
                raise RegionGrowerError(
                    f"The target path distance computed from the fit is {target_path_distance}"
                    f" < {MIN_TARGET_PATH_DISTANCE}!"
                )

            params[neurite_type]["modify"] = {
                "funct": scale_target_barcode,
                "kwargs": {
                    "target_path_distance": target_path_distance,
                    "with_debug_info": debug_info is not None,
                },
            }

        else:
            if debug_info is not None:
                debug_info.update(
                    {
                        "default_func": {
                            "inputs": {
                                "target_thickness": target_thickness,
                                "reference_thickness": reference_thickness,
                                "min_target_thickness": MIN_TARGET_THICKNESS,
                            },
                            "scaling": [],
                        }
                    }
                )
            if target_thickness < MIN_TARGET_THICKNESS:
                raise RegionGrowerError(
                    f"The target thickness {target_thickness} is too small to be able to scale the"
                    f" bar code with {MIN_TARGET_THICKNESS}"
                )
            params[neurite_type]["modify"] = {
                "funct": scale_default_barcode,
                "kwargs": {
                    "target_thickness": target_thickness,
                    "reference_thickness": reference_thickness,
                    "with_debug_info": debug_info is not None,
                },
            }


def output_scaling(
    root_section: Section,
    orientation: List[float],
    target_min_length: Optional[float] = None,
    target_max_length: Optional[float] = None,
) -> float:
    """Returns the scaling factor to be applied on Y axis for this neurite

    The scaling is chosen such that the neurite is:
    - upscaled to reach the min target length if it is too short
    - downscaled to stop at the max target length if it is too long

    Args:
        root_section: the neurite for which the scale is computed
        orientation: 3-array with y direction to project points to get the extend of the cell
        target_min_length: min length that the neurite must reach
        target_max_length: max length that the neurite can reach

    Returns: scale factor to apply"""
    max_y = max(p for sec in root_section.iter() for p in sec.points[:, COLS.XYZ].dot(orientation))
    y_extent = max_y - root_section.points[0, COLS.XYZ].dot(orientation)

    # in case the neurite goes to -y direction
    if np.round(y_extent, 3) == 0.0:
        return 1.0

    if target_min_length is not None:
        min_scale = target_min_length / y_extent
        if min_scale > 1:
            return min_scale

    if target_max_length is not None:
        max_scale = target_max_length / y_extent
        if max_scale < 1:
            return max_scale

    return 1.0
