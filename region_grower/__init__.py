"""region-grower package.

Synthesize cells in a given spatial context.
"""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of region-grower.
# See https://github.com/BlueBrain/region-grower for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import importlib.metadata

__version__ = importlib.metadata.version("region-grower")


class RegionGrowerError(Exception):
    """Exception thrown by region grower."""


class SkipSynthesisError(Exception):
    """An exception thrown when the morphology synthesis must be skipped."""
