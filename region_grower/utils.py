"""Utils module."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of region-grower.
# See https://github.com/BlueBrain/region-grower for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import json
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from voxcell.math_utils import angles_to_matrices

L = logging.getLogger(__name__)


def setup_logger(level="info", prefix="", suffix=""):
    """Setup application logger."""
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    logging.basicConfig(
        format=prefix + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + suffix,
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=levels[level],
    )

    if levels[level] >= logging.INFO:  # pragma: no cover
        logging.getLogger("distributed").level = max(
            logging.getLogger("distributed").level, logging.WARNING
        )


class NumpyEncoder(json.JSONEncoder):
    """To encode numpy arrays."""

    def default(self, o):  # pylint: disable=method-hidden
        """Actual encoder."""
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        return json.JSONEncoder.default(self, o)  # pragma: no cover


def create_morphologies_dict(dat_file, morph_path, ext=".asc"):
    """Create dict to load the morphologies from a directory, with dat file."""
    morph_name = pd.read_csv(dat_file, sep=" ", dtype={0: object})
    name_dict = defaultdict(list)
    for morph in morph_name.values:
        name_dict[morph[2]].append(os.path.join(morph_path, str(morph[0]) + ext))
    return name_dict


def random_rotation_y(n, rng=np.random):
    """Random rotation around Y-axis.

    Args:
        n: number of rotation matrices to generate

    Returns:
        n x 3 x 3 NumPy array with rotation matrices.
    """
    # copied from `brainbuilder.cell_orientations` to avoid a heavy dependency
    # consider reusing `brainbuilder` methods if we need something more general
    # (like user-defined distributions for rotation angles)
    angles = rng.uniform(-np.pi, np.pi, size=n)
    return angles_to_matrices(angles, axis="y")


def load_morphology_list(filepath, check_gids=None):
    """Read morphology list from tab-separated file."""
    result = pd.read_csv(
        filepath, sep=r"\s+", index_col=0, dtype={"morphology": object, "scale": float}
    )
    result.loc[result["morphology"].isnull(), "morphology"] = None
    if "scale" not in result:
        result["scale"] = None
    if check_gids is not None:
        if sorted(result.index) != sorted(check_gids):
            raise RuntimeError("Morphology list GIDs mismatch")
    return result


def _failure_ratio_by_mtype(mtypes, na_mask):
    """Calculate ratio of N/A occurrences per mtype."""
    failed = mtypes.loc[na_mask].value_counts()
    overall = mtypes.value_counts()
    result = (
        pd.DataFrame(
            {
                "N/A": failed,
                "out of": overall,
            }
        )
        .dropna()
        .astype(int)
    )
    result["ratio, %"] = 100.0 * result["N/A"] / result["out of"]
    result.sort_values("ratio, %", ascending=False, inplace=True)
    return result


def check_na_morphologies(morph_list, mtypes, threshold=None):
    """Check N/A ratio per mtype."""
    na_mask = morph_list["morphology"].isnull()
    if na_mask.any():
        stats = _failure_ratio_by_mtype(mtypes, na_mask)
        L.warning("N/A morphologies for %d position(s)", np.count_nonzero(na_mask))
        L.info("N/A ratio by mtypes:\n%s", stats.to_string(float_format="%.1f"))
        if threshold is not None:
            exceeded = 0.01 * stats["ratio, %"] > threshold
            if exceeded.any():
                raise RuntimeError(
                    f"Max N/A ratio ({100.0 * threshold:.1f}%) exceeded for mtype(s): "
                    f"{', '.join(exceeded[exceeded].index)}"
                )


def assign_morphologies(cells, morphologies):
    """Assign morphologies to CellCollection.

    Args:
        cells: CellCollection to be augmented
        morphologies: dictionary {gid -> morphology_name}

    No return value; `cells` is input/output argument.
    """
    cells.properties["morphology"] = pd.Series(morphologies)
    na_mask = cells.properties["morphology"].isnull()
    if na_mask.any():
        L.info(
            "Dropping %d cells with no morphologies assigned and reindexing...",
            np.count_nonzero(na_mask),
        )
        cells.remove_unassigned_cells()

    cells.properties["morphology_producer"] = "synthesis"
