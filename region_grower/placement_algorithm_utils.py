"""
Miscellaneous utilities.
"""
import logging
import os
import random
import uuid
from pathlib import Path

import morphio
import numpy as np
import pandas as pd
from voxcell import CellCollection

LOGGER = logging.getLogger(__name__)


def setup_logger():  # pragma: no cover
    """ Setup application logger. """
    logging.basicConfig(
        format="%(asctime)s;%(levelname)s;%(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=logging.ERROR,
    )
    LOGGER.setLevel(logging.INFO)


def load_cells(filepath, mvd3_filepath=None):
    """Load CellCollection from file.

    Args:
        filepath (str): cells file
        mvd3_filepath (str): MVD3 cells file. Temporary for backward compatibility.

    Returns:
        CellCollection: cells
    """
    if mvd3_filepath is not None:  # pragma: no cover
        LOGGER.warning("--mvd3 option is deprecated. Use --cells_path instead.")
        return CellCollection.load_mvd3(mvd3_filepath)
    if filepath is None:  # pragma: no cover
        raise ValueError("`--cells-path` option is required")
    return CellCollection.load(filepath)


def save_cells(cells, filepath, mvd3_filepath=None):
    """Save CellCollection to file.

    Args:
        cells (CellCollection): cells to be saved
        filepath (str): cells file
        mvd3_filepath (str): MVD3 cells file. Temporary for backward compatibility.
    """
    if mvd3_filepath:  # pragma: no cover
        LOGGER.warning("--out-mvd3 option is deprecated. Use --out-cells-path instead.")
        cells.save_mvd3(mvd3_filepath)
    elif filepath is None:  # pragma: no cover
        raise ValueError("`--out-cells-path` option is required")
    else:
        cells.save(filepath)


def random_rotation_y(n):  # pragma: no cover
    """
    Random rotation around Y-axis.

    Args:
        n: number of rotation matrices to generate

    Returns:
        n x 3 x 3 NumPy array with rotation matrices.
    """
    # copied from `brainbuilder.cell_orientations` to avoid a heavy dependency
    # consider reusing `brainbuilder` methods if we need something more general
    # (like user-defined distributions for rotation angles)
    # pylint: disable=import-outside-toplevel
    from voxcell.math_utils import angles_to_matrices

    angles = np.random.uniform(-np.pi, np.pi, size=n)
    return angles_to_matrices(angles, axis="y")


def load_morphology_list(filepath, check_gids=None):  # pragma: no cover
    """ Read morphology list from tab-separated file. """
    result = pd.read_csv(
        filepath, sep=r"\s+", index_col=0, dtype={"morphology": object, "scale": float}
    )
    result["morphology"].replace({np.nan: None}, inplace=True)
    if check_gids is not None:
        if sorted(result.index) != sorted(check_gids):
            raise RuntimeError("Morphology list GIDs mismatch")
    return result


def _failure_ratio_by_mtype(mtypes, na_mask):  # pragma: no cover
    """ Calculate ratio of N/A occurences per mtype. """
    failed = mtypes[na_mask].value_counts()
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


def check_na_morphologies(morph_list, mtypes, threshold=None):  # pragma: no cover
    """ Check `N/A` ratio per mtype. """
    na_mask = morph_list.isnull().any(axis=1)
    if na_mask.any():
        stats = _failure_ratio_by_mtype(mtypes, na_mask)
        LOGGER.warning("N/A morphologies for %d position(s)", np.count_nonzero(na_mask))
        LOGGER.info("N/A ratio by mtypes:\n%s", stats.to_string(float_format="%.1f"))
        if threshold is not None:
            exceeded = 0.01 * stats["ratio, %"] > threshold
            if exceeded.any():
                raise RuntimeError(
                    "Max N/A ratio (%.1f%%) exceeded for mtype(s): %s"
                    % (100.0 * threshold, ", ".join(exceeded[exceeded].index))
                )


class MorphWriter(object):
    """ Helper class for writing morphologies. """

    def __init__(self, output_dir, file_ext):
        self.output_dir = os.path.realpath(output_dir)
        self.file_ext = file_ext
        self._dir_depth = None
        self.last_paths = None

    @staticmethod
    def _calc_dir_depth(num_files, max_files_per_dir=None):  # pragma: no cover
        """ Directory depth required to have no more than given number of files per folder. """
        if (max_files_per_dir is None) or (num_files < max_files_per_dir):
            return None
        if max_files_per_dir < 256:
            raise RuntimeError(
                """
                Less than 256 files per folder is too restrictive.
            """
            )
        result, capacity = 0, max_files_per_dir
        while capacity < num_files:
            result += 1
            capacity *= 256
        if result > 3:
            raise RuntimeError(
                """
                More than three intermediate folders is a bit too much.
            """
            )
        return result

    @staticmethod
    def _make_subdirs(dirpath, depth):  # pragma: no cover
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        if depth <= 0:
            return
        for sub in range(256):
            MorphWriter._make_subdirs(os.path.join(dirpath, "%02x" % sub), depth - 1)

    def prepare(self, num_files, max_files_per_dir=None, overwrite=False):
        """
        Prepare output directory.

          - ensure it either does not exist, or is empty
          - if it does not exist, create an empty one
        """
        self._dir_depth = MorphWriter._calc_dir_depth(
            num_files * len(self.file_ext), max_files_per_dir
        )
        if os.path.exists(self.output_dir):
            if not overwrite and os.listdir(self.output_dir):  # pragma: no cover
                raise RuntimeError("Non-empty morphology output folder '%s'" % self.output_dir)
        else:  # pragma: no cover
            os.makedirs(self.output_dir)
        if self._dir_depth is not None:  # pragma: no cover
            MorphWriter._make_subdirs(os.path.join(self.output_dir, "hashed"), self._dir_depth)

    def _generate_name(self, seed):
        morph_name = uuid.UUID(int=random.Random(seed).getrandbits(128)).hex
        if self._dir_depth is None:
            subdirs = ""
        else:  # pragma: no cover
            subdirs = "hashed"
            assert len(morph_name) >= 2 * self._dir_depth
            for k in range(self._dir_depth):
                sub = morph_name[2 * k : 2 * k + 2]
                subdirs = os.path.join(subdirs, sub)
        return morph_name, subdirs

    def filepaths(self, full_stem):
        """Returns the paths to the morphology"""
        return [Path(self.output_dir, full_stem.with_suffix("." + ext)) for ext in self.file_ext]

    def __call__(self, morph, seed):
        # TODO: pass nrn_order option directly to .write()
        morph = morphio.mut.Morphology(  # pylint: disable=no-member
            morph, options=morphio.Option.nrn_order
        )
        morph_name, subdirs = self._generate_name(seed)

        full_stem = Path(subdirs, morph_name)
        self.last_paths = self.filepaths(full_stem)

        for path in self.last_paths:
            morph.write(path)
        return str(full_stem)


def assign_morphologies(cells, morphologies):
    """
    Assign morphologies to CellCollection.

    Args:
        cells: CellCollection to be augmented
        morphologies: dictionary {gid -> morphology_name}

    No return value; `cells` is input/output argument.
    """
    cells.properties["morphology"] = pd.Series(morphologies)
    na_mask = cells.properties["morphology"].isnull()
    if na_mask.any():  # pragma: no cover
        LOGGER.info(
            "Dropping %d cells with no morphologies assigned; reindexing", np.count_nonzero(na_mask)
        )
        cells.remove_unassigned_cells()
