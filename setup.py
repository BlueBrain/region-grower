"""Setup for the region-grower package."""
import imp

from setuptools import find_packages
from setuptools import setup

# Read the contents of the README file
with open("README.rst", encoding="utf-8") as f:
    README = f.read()


VERSION = imp.load_source("region_grower.version", "region_grower/version.py").VERSION

REQS = [
    "attrs>=19.3.0",
    "click>=7.0",
    "dask[dataframe, distributed]>=2.15.0,!=2021.8.1,!=2021.8.2",
    "diameter-synthesis>=0.2.5,<1",
    "morphio>=3,<4",
    "morph-tool[nrn]>=2.9,<3",
    "neuroc>=0.2.8,<1",
    "neurom>=3,<4",
    "neurots>=3,<4",
    "tqdm>=4.28.1",
    "voxcell>=2.7,<4",
]

MPI_EXTRAS = [
    "dask_mpi",
    "mpi4py>=3.0.3",
]

DOC_EXTRAS = [
    "m2r2",
    "sphinx",
    "sphinx-bluebrain-theme",
    "sphinx-jsonschema",
]

setup(
    name="region-grower",
    author="bbp-ou-cells",
    author_email="bbp-ou-cells@groupes.epfl.ch",
    version=VERSION,
    description="Synthesize cells in a given spatial context",
    long_description=README,
    long_description_content_type="text/x-rst",
    url="https://bbpteam.epfl.ch/documentation/projects/region-grower",
    project_urls={
        "Tracker": "https://bbpteam.epfl.ch/project/issues/projects/CELLS/issues",
        "Source": "https://bbpgitlab.epfl.ch/neuromath/region-grower",
    },
    license="BBP-internal-confidential",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.8",
    install_requires=REQS,
    extras_require={
        "mpi": MPI_EXTRAS,
        "docs": DOC_EXTRAS,
    },
    entry_points={"console_scripts": ["region-grower=region_grower.cli:cli"]},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
