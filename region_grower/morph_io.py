"""Module to load and write morphologies."""
import os
import random
import uuid
from pathlib import Path

import morphio


class MorphLoader:
    """Morphology loader.

    Args:
        base_dir: path to directory with morphology files
        file_ext: file extension to look for
    """

    def __init__(self, base_dir, file_ext):
        self.base_dir = base_dir
        self.file_ext = self._ensure_startswith_point(file_ext)

    @staticmethod
    def _ensure_startswith_point(file_ext):
        if file_ext.startswith("."):
            return file_ext
        return "." + file_ext

    def get(self, name, options=None):
        """Load a morphology given its name."""
        filepath = (Path(self.base_dir) / name).with_suffix(self.file_ext)
        if not filepath.exists():
            return None
        kwargs = {"options": options} if options is not None else {}
        return morphio.mut.Morphology(str(filepath), **kwargs)  # pylint: disable=no-member


class MorphWriter:
    """Helper class for writing morphologies."""

    def __init__(self, output_dir, file_exts):
        self.output_dir = os.path.realpath(output_dir)
        self.file_exts = file_exts
        self._dir_depth = None

    @staticmethod
    def _calc_dir_depth(num_files, max_files_per_dir=None):
        """Directory depth required to have no more than given number of files per folder."""
        if (max_files_per_dir is None) or (num_files < max_files_per_dir):
            return None
        if max_files_per_dir < 256:
            raise RuntimeError("""Less than 256 files per folder is too restrictive.""")
        result, capacity = 0, max_files_per_dir
        while capacity < num_files:
            result += 1
            capacity *= 256
        if result > 3:
            raise RuntimeError("""More than three intermediate folders is a bit too much.""")
        return result

    @staticmethod
    def _make_subdirs(dirpath, depth):
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        if depth <= 0:
            return
        for sub in range(256):
            MorphWriter._make_subdirs(os.path.join(dirpath, "%02x" % sub), depth - 1)

    def prepare(self, num_files, max_files_per_dir=None, overwrite=False):
        """Prepare output directory.

        - ensure it either does not exist, or is empty
        - if it does not exist, create an empty one
        """
        self._dir_depth = MorphWriter._calc_dir_depth(
            num_files * len(self.file_exts), max_files_per_dir
        )
        if os.path.exists(self.output_dir):
            if not overwrite and os.listdir(self.output_dir):
                raise RuntimeError("Non-empty morphology output folder '%s'" % self.output_dir)
        else:
            os.makedirs(self.output_dir)
        if self._dir_depth is not None:
            MorphWriter._make_subdirs(os.path.join(self.output_dir, "hashed"), self._dir_depth)

    def _generate_name(self, seed):
        morph_name = uuid.UUID(int=random.Random(seed).getrandbits(128)).hex
        if self._dir_depth is None:
            subdirs = ""
        else:
            subdirs = "hashed"
            assert len(morph_name) >= 2 * self._dir_depth
            for k in range(self._dir_depth):
                sub = morph_name[2 * k : 2 * k + 2]
                subdirs = os.path.join(subdirs, sub)
        return morph_name, subdirs

    def filepaths(self, full_stem):
        """Returns the paths to the morphology"""
        return [Path(self.output_dir, full_stem.with_suffix("." + ext)) for ext in self.file_exts]

    def __call__(self, morph, seed):
        """Write the given morphology."""
        morph = morphio.mut.Morphology(  # pylint: disable=no-member
            morph, options=morphio.Option.nrn_order
        )
        morph_name, subdirs = self._generate_name(seed)

        full_stem = Path(subdirs, morph_name)
        ext_paths = self.filepaths(full_stem)

        for path in ext_paths:
            morph.write(path)
        return str(full_stem), ext_paths