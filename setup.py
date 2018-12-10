"""Space synthesis"""
# pylint: disable=R0801
import os
from setuptools import setup
from setuptools import find_packages


VERSION = "0.0.0"


setup(**{
    'version': VERSION,
    'install_requires': [
        'voxcell>=2.5.4',
        'click>=7.0',
        'tqdm>=4.28.1',
        'tns>=1.0.1',
    ],
    'packages': find_packages(),
    'extras_require': {
        'validate': ['mayavi==4.5.0'],
    },
    'entry_points': {
        'console_scripts': ['region-grower=region_grower.app:cli']},
    'name': 'region_grower'
})
