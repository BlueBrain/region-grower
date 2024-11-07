"""Setup for the region-grower package."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of region-grower.
# See https://github.com/BlueBrain/region-grower for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path

from setuptools import find_namespace_packages
from setuptools import setup

reqs = [
    "attrs>=19.3.0",
    "click>=8.0",
    "dask[dataframe, distributed]>=2023.3.2",
    "diameter-synthesis>=0.5.4,<1",
    "morphio>=3.3.6,<4",
    "morph-tool[nrn]>=2.9,<3",
    "neuroc>=0.3,<1",
    "neurom>=3,<4",
    "neurots>=3.6,<4",
    "numpy>=1.22",
    "pandas>=1.5.3",
    "tqdm>=4.28.1",
    "voxcell>=3.1.1,<4",
    "pynrrd>=0.4.0,<1.0",  # temp fix for nrrd saving in voxcell
]

mpi_extras = [
    "dask_mpi>=2022.4",
    "mpi4py>=3.1.1",
]

doc_reqs = [
    "docutils<0.21",
    "m2r2",
    "sphinx",
    "sphinx-bluebrain-theme",
    "sphinx-jsonschema",
    "sphinx-click",
]

test_reqs = [
    "brainbuilder>=0.20",
    "dictdiffer>=0.9",
    "pytest>=6.2.5",
    "pytest-click>=1",
    "pytest-console-scripts>=1.3",
    "pytest-cov>=3",
    "pytest-html>=2",
    "pytest-mock>=3.5",
    "pytest-xdist>=3.0.2",
]

setup(
    name="region-grower",
    author="Blue Brain Project, EPFL",
    description="Synthesize cells in a given spatial context.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://region-grower.readthedocs.io",
    project_urls={
        "Tracker": "https://github.com/BlueBrain/region-grower/issues",
        "Source": "https://github.com/BlueBrain/region-grower",
    },
    license="Apache License 2.0",
    packages=find_namespace_packages(include=["region_grower*"]),
    python_requires=">=3.9",
    use_scm_version=True,
    setup_requires=[
        "setuptools_scm",
    ],
    install_requires=reqs,
    extras_require={
        "docs": doc_reqs,
        "mpi": mpi_extras,
        "test": test_reqs,
    },
    entry_points={
        "console_scripts": [
            "region-grower=region_grower.cli:main",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
