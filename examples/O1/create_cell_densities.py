# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of region-grower.
# See https://github.com/BlueBrain/region-grower for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

import json
import os
import sys

import yaml
from trimesh import ray

if __name__ == "__main__":
    if not ray.has_embree:
        print("you do not have pyembree installed, ray tracing will be slow")

    with open("cell_composition.yaml") as f:
        comp = yaml.safe_load(f)
    for data in comp["neurons"]:
        data["density"] = float(sys.argv[-1])

    with open("cell_composition.yaml", "w") as f:
        yaml.safe_dump(comp, f)

    mtypes = json.loads(os.environ["MTYPES"])
    red_comp = {"neurons": [], "version": "v2.0"}
    for data in comp["neurons"]:
        if data["traits"]["mtype"] in mtypes:
            red_comp["neurons"].append(data)

    with open("cell_composition_red.yaml", "w") as f:
        yaml.safe_dump(red_comp, f)
