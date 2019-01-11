"""Space synthesis"""

import imp
from setuptools import setup, find_packages


VERSION = imp.load_source("region_grower.version", "region_grower/version.py").VERSION


setup(
    name='region-grower',
    version=VERSION,
    install_requires=[
        'voxcell>=2.5.4',
        'click>=7.0',
        'tqdm>=4.28.1',
        'tns>=1.0.1',
    ],
    packages=find_packages(),
    extras_require={
        'validate': [
            'mayavi==4.5.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'region-grower=region_grower.app:cli'
        ]
    },
)
