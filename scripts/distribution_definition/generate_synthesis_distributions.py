"""Extract TNS distributions used by NeuroTS."""

# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of region-grower.
# See https://github.com/BlueBrain/region-grower for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import print_function

import argparse
import json
import multiprocessing
import os.path
import sys

import numpy as np
from neurots import extract_input

MTYPES = [
    "L1_DAC",
    "L1_HAC",
    "L1_LAC",
    "L1_NGC-DA",
    "L1_NGC-SA",
    "L1_SAC",
    "L23_BP",
    "L23_BTC",
    "L23_CHC",
    "L23_DBC",
    "L23_LBC",
    "L23_MC",
    "L23_NBC",
    "L23_NGC",
    "L23_SBC",
    "L2_IPC",
    "L2_TPC:A",
    "L2_TPC:B",
    "L3_TPC:A",
    "L3_TPC:B",
    "L4_BP",
    "L4_BTC",
    "L4_CHC",
    "L4_DBC",
    "L4_LBC",
    "L4_MC",
    "L4_NBC",
    "L4_NGC",
    "L4_SBC",
    "L4_SSC",
    "L4_TPC",
    "L4_UPC",
    "L5_BP",
    "L5_BTC",
    "L5_CHC",
    "L5_DBC",
    "L5_LBC",
    "L5_MC",
    "L5_NBC",
    "L5_NGC",
    "L5_SBC",
    "L5_TPC:A",
    "L5_TPC:B",
    "L5_TPC:C",
    "L5_UPC",
    "L6_BP",
    "L6_BPC",
    "L6_BTC",
    "L6_CHC",
    "L6_DBC",
    "L6_HPC",
    "L6_IPC",
    "L6_LBC",
    "L6_MC",
    "L6_NBC",
    "L6_NGC",
    "L6_SBC",
    "L6_TPC:A",
    "L6_TPC:C",
    "L6_UPC",
]

# Experimental cortical thicknesses from which
# the experimental cells were extracted
# Ordering is L1 to L6
THICKNESSES = {
    "mouse": [118.3, 93.01, 169.5, 178.6, 349.2, 420.5],
    "rat": [165, 149, 353, 190, 525, 700],
}


class NumpyEncoder(json.JSONEncoder):
    """The JSON encoder to handle Numpy objects."""

    def default(self, obj):
        """The default handler."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Worker:
    """The Worker to extract distributions."""

    def __init__(self, base_dir, feature):
        self.base_dir = base_dir
        self.feature = feature

    def check(self, mtype):
        """Check that a file exists for the given mtype."""
        if not os.path.exists(os.path.join(self.base_dir, mtype)):
            raise RuntimeError("No data for '%s'" % mtype)

    def __call__(self, mtype):
        """Extract the distributions."""
        print("%s..." % mtype, file=sys.stderr)
        result = extract_input.distributions(
            os.path.join(self.base_dir, mtype),
            neurite_types=["basal", "apical"],
            feature=self.feature,
        )
        print("%s...done" % mtype, file=sys.stderr)
        return result


def main(args):
    """The main process."""
    worker = Worker(args.base_dir, args.feature)
    for mtype in MTYPES:
        worker.check(mtype)

    if args.jobs is None:
        mapper = map
    else:
        pool = multiprocessing.Pool(args.jobs)
        mapper = pool.map
    result = {
        "mtypes": dict(zip(MTYPES, mapper(worker, MTYPES))),
        "metadata": {"cortical_thickness": THICKNESSES[args.species]},
    }

    with open(args.output, "w") as f:
        json.dump(result, f, sort_keys=True, indent=2, cls=NumpyEncoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract TNS distributions")
    parser.add_argument("base_dir", help="Base morphology folder")
    parser.add_argument(
        "--species",
        choices=["mouse", "rat"],
        help="Whether mouse or rat. This is needed to get the correct source cortical thicknesses",
    )
    parser.add_argument(
        "--feature",
        help="??? [default: %(default)s]",
        default="path_distances_2",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, help="Number of jobs to run in parallel", default=None
    )
    parser.add_argument("-o", "--output", help="Path to output JSON file", required=True)
    main(parser.parse_args())
