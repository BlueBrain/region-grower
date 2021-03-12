'''Region grower module'''
from region_grower.version import VERSION as __version__  # noqa


class RegionGrowerError(Exception):
    '''Exception thrown by region grower'''


class SkipSynthesisError(Exception):
    '''An exception thrown when the morphology synthesis must be skipped'''
