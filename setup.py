"""Setup for the region-grower package."""
import importlib.util
from pathlib import Path

from setuptools import find_packages
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
    "dask[dataframe, distributed]>=2.15.0,!=2021.8.1,!=2021.8.2",
    "diameter-synthesis>=0.2.5,<1",
    "morphio>=3,<4",
    "morph-tool[nrn]>=2.9,<3",
    "neuroc>=0.2.8,<1",
    "neurom>=3,<4",
    "neurots>=3.2,<4",
    "tqdm>=4.28.1",
    "voxcell>=2.7,<4",
]

mpi_extras = [
    "dask_mpi",
    "mpi4py>=3.0.3",
]

doc_reqs = [
    "m2r2",
    "sphinx",
    "sphinx-bluebrain-theme",
    "sphinx-jsonschema",
]

test_reqs = [
    "brainbuilder",
    "dictdiffer",
    "pytest",
    "pytest-cov",
    "pytest-html",
    "pytest-xdist",
    "pytest-mock",
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
    packages=find_packages(include=["region_grower"]),
    python_requires=">=3.8",
    version=VERSION,
    install_requires=reqs,
    extras_require={
        "docs": doc_reqs,
        "mpi": mpi_extras,
        "test": test_reqs,
    },
    entry_points={"console_scripts": ["region-grower=region_grower.cli:cli"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    include_package_data=True,
)
