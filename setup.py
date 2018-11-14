"""Space synthesis"""
# pylint: disable=R0801
import os
from setuptools import setup
from setuptools import find_packages


VERSION = "0.0.0"

REQS = ['voxcell>=2.5.4',
        'bluepy>=0.12.7',
        'mayavi>=4.6.2']


config = {
    'version': VERSION,
    'install_requires': REQS,
    'packages': find_packages(),
    'name': 'space_synthesis',
}

setup(**config)
