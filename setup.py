"""Setup for the region-grower package."""
import importlib.util
from pathlib import Path

from setuptools import find_namespace_packages
from setuptools import setup

spec = importlib.util.spec_from_file_location(
    "region_grower.version",
    "region_grower/version.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.VERSION

reqs = [
    "attrs>=19.3.0",
    "click>=7.0",
    "dask[dataframe, distributed]>=2021.9",
    "diameter-synthesis>=0.5.2,<1",
    "morphio>=3.3.3,<4",
    "morph-tool[nrn]>=2.9,<3",
    "neuroc>=0.2.8,<1",
    "neurom>=3,<4",
    "neurots>=3.3.1,<4",
    "numpy>=1.22",
    "pandas>=1.5",
    "tqdm>=4.28.1",
    "urllib3>=1.26,<2; python_version < '3.9'",
    "voxcell>=3.1.1,<4",
]

mpi_extras = [
    "dask_mpi",
    "mpi4py>=3.0.3",
]

doc_reqs = [
    "m2r2",
    "sphinx<6",
    "sphinx-bluebrain-theme",
    "sphinx-jsonschema",
    "sphinx-click",
]

test_reqs = [
    "brainbuilder>=0.18.3",
    "dictdiffer>=0.9",
    "pytest>=6.2.5",
    "pytest-click>=1",
    "pytest-cov>=3",
    "pytest-html>=2",
    "pytest-mock>=3.5",
    "pytest-xdist>=3.0.2",
]

setup(
    name="region-grower",
    author="bbp-ou-cells",
    author_email="bbp-ou-cells@groupes.epfl.ch",
    description="Synthesize cells in a given spatial context.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://bbpteam.epfl.ch/documentation/projects/region-grower",
    project_urls={
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/CELLS/issues",
        "Source": "https://bbpgitlab.epfl.ch/neuromath/region-grower",
    },
    license="BBP-internal-confidential",
    packages=find_namespace_packages(include=["region_grower*"]),
    python_requires=">=3.8",
    version=VERSION,
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
