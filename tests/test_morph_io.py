"""Test the region_grower.utils module."""

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
# pylint: disable=protected-access
from pathlib import Path

import pytest

from region_grower import SkipSynthesisError
from region_grower.morph_io import MorphLoader
from region_grower.morph_io import MorphWriter

DATA = Path(__file__).parent / "data"


class TestMorphLoader:
    """Test the MorphLoader class."""

    def test_ensure_ext_start_with_point(self):
        """Test extension handler."""
        assert MorphLoader("test_dir", file_ext="h5").file_ext == ".h5"
        assert MorphLoader("test_dir", file_ext=".h5").file_ext == ".h5"

    def test_get(self, morph_loader):
        """Test the get method."""
        with pytest.raises(SkipSynthesisError):
            morph_loader.get("UNKNOWN")

        res = morph_loader.get("C170797A-P1")
        assert len(res.root_sections) == 8
        assert len(res.sections) == 52


class TestMorphWriter:
    """Test the MorphWriter class."""

    def test_calc_dir_depth(self, morph_writer):
        """Test the `_calc_dir_depth()` method."""
        assert morph_writer._calc_dir_depth(0) is None
        assert morph_writer._calc_dir_depth(500) is None
        assert morph_writer._calc_dir_depth(5000, 500) == 1
        assert morph_writer._calc_dir_depth(256**2 + 1, 256) == 2
        assert morph_writer._calc_dir_depth(256**3 + 1, 256) == 3

        with pytest.raises(
            RuntimeError, match="Less than 256 files per folder is too restrictive."
        ):
            morph_writer._calc_dir_depth(1000, 200)

        with pytest.raises(
            RuntimeError, match="More than three intermediate folders is a bit too much."
        ):
            morph_writer._calc_dir_depth(256**4 + 1, 256)

    def test_make_subdirs(self, morph_writer, tmpdir):
        """Test the `_make_subdirs()` method."""
        dirpath = Path(tmpdir)
        morph_writer._make_subdirs(str(dirpath), 0)
        assert dirpath.exists()
        assert len(list(dirpath.iterdir())) == 0

        dirpath = Path(tmpdir / "depth_1")
        morph_writer._make_subdirs(str(dirpath), 1)
        assert dirpath.exists()
        assert len(list(dirpath.iterdir())) == 256

    def test_prepare(self, morph_writer, tmpdir):
        """Test the `prepare()` method."""
        morph_writer.prepare(500, 256, True)
        morph_writer.prepare(256**2 + 1, 256, True)

        with pytest.raises(
            RuntimeError, match=r"Non-empty morphology output folder '.*/test_prepare.*"
        ):
            morph_writer.prepare(256**2 + 1, 256, False)

        new_morph_writer = MorphWriter(tmpdir / "test_new_dir", file_exts=["h5"])
        new_morph_writer.prepare(500, 256)

    def test_file_paths(self, morph_writer):
        """Test the `filepaths()` method."""
        morph_writer = MorphWriter("/test", file_exts=["h5"])
        assert morph_writer.filepaths(Path("", "morph_name")) == [Path("/test/morph_name.h5")]

    def test_generate_name(self, morph_writer):
        """Test the `generate_name()` method."""
        assert morph_writer.generate_name(0) == ("e3e70682c2094cac629f6fbed82c07cd", "")

        morph_writer._dir_depth = 2
        assert morph_writer.generate_name(42) == (
            "bdd640fb06671ad11c80317fa3b1799d",
            "hashed/bd/d6",
        )

    def test_call(self, morph_writer, synthesized_cell, tmpdir):
        """Test the `__call__()` method."""
        full_stem, ext_paths = morph_writer(synthesized_cell.neuron, 0)

        expected_full_stem = "e3e70682c2094cac629f6fbed82c07cd"
        assert full_stem == expected_full_stem
        assert ext_paths == [(Path(tmpdir) / expected_full_stem).with_suffix(".h5")]
