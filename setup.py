"""Space synthesis"""
# pylint: disable=R0801
import os
from setuptools import setup
from setuptools import find_packages


VERSION = "0.0.0"

REQS = ['voxcell>=2.5.4',]


config = {
    'version': VERSION,
    'install_requires': REQS,
    'packages': find_packages(),
    'name': 'space_synthesis',
    'extras_require': {
    'validate': ['mayavi==4.5.0'],
}

setup(**config)
