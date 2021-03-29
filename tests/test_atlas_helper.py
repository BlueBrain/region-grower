"""Test the region_grower.atlas_helper.AtlasHelper module."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from nose.tools import assert_equal
from nose.tools import assert_raises
from voxcell.exceptions import VoxcellError
from voxcell.nexus.voxelbrain import Atlas

from region_grower.atlas_helper import AtlasHelper

from .atlas_mock import small_O1

DATA = Path(__file__).parent / "data"


def test_atlas_helper():
    """All test are made in a single function as we do not want to regenerate
    the atlas for each test
    """
    with TemporaryDirectory() as tempdir:
        small_O1(tempdir)
        helper = AtlasHelper(Atlas.open(tempdir))
        assert_equal(helper.layer_thickness(1).raw[5, 5, 5], 200)

        assert_equal(helper.pia_coord.raw[5, 5, 5], 800)

        cortical_depths = np.cumsum([4] * 6)

        # test in L1
        assert_equal(
            helper.lookup_target_reference_depths([0, 500, 0], cortical_depths),
            (300, 8),
        )

        assert_equal(
            helper.lookup_target_reference_depths([100, 600, 100], cortical_depths),
            (200, 4),
        )

        assert_raises(
            VoxcellError,
            helper.lookup_target_reference_depths,
            [10000, 0, 0],
            cortical_depths,
        )