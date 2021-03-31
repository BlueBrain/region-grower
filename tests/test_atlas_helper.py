"""Test the region_grower.atlas_helper.AtlasHelper module."""
from pathlib import Path

from region_grower.atlas_helper import AtlasHelper

DATA = Path(__file__).parent / "data"


def test_atlas_helper(small_O1):
    """Test the atlas helper."""
    helper = AtlasHelper(small_O1)
    assert helper.layer_thickness(1).raw[5, 5, 5] == 200

    assert helper.pia_coord.raw[5, 5, 5] == 800
