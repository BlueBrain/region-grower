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
        'diameter-synthesis>=0.1.7',
        'morphio>=2.3.4',
        'neuroc>=0.2.3',
        'tns>=2.2.4',
        'tqdm>=4.28.1',
        'voxcell>=2.6.3',
    ],
    packages=find_packages(),
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'region-grower=region_grower.cli:cli'
        ]
    },
    include_package_data=True,
)
