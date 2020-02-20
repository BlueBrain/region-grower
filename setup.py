"""Space synthesis"""

import imp
from setuptools import setup, find_packages


VERSION = imp.load_source("region_grower.version", "region_grower/version.py").VERSION


setup(
    name='region-grower',
    version=VERSION,
    install_requires=[
        'attrs>=19.3.0',
        'click>=7.0',
        'diameter-synthesis>=0.0.13',
        'tns>=2.0.4',
        'tqdm>=4.28.1',
        'voxcell>=2.5.4',
    ],
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'region-grower=region_grower.cli:cli'
        ]
    },
)
