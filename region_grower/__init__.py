"""region-grower package.

Synthesize cells in a given spatial context.
"""

import importlib.metadata

__version__ = importlib.metadata.version("region-grower")


class RegionGrowerError(Exception):
    """Exception thrown by region grower."""


class SkipSynthesisError(Exception):
    """An exception thrown when the morphology synthesis must be skipped."""
