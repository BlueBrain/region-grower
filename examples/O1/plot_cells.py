# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright (c) 2023-2024 Blue Brain Project, EPFL.
#
# This file is part of region-grower.
# See https://github.com/BlueBrain/region-grower for further info.
#
# SPDX-License-Identifier: Apache-2.0
#

"""Plot reconstructed cells."""

import json
import os

import matplotlib.pyplot as plt
import neurom as nm
from matplotlib.backends.backend_pdf import PdfPages
from morph_tool.morphdb import MorphDB
from neurom import view

if __name__ == "__main__":
    df = MorphDB.from_neurondb(
        "../rat_isocortex_release/neuronDB.xml", morphology_folder="../rat_isocortex_release"
    ).df
    selected = [
        "sm120429_2photon_a1-3_idA",
        "mtC261001B_idB",
        "rp101229_L5-2_idC",
        "rat_P16_S1_RH3_20140129",
    ]
    mtypes = json.loads(os.environ["MTYPES"])
    for mtype in mtypes:
        with PdfPages(f"bio_{mtype}.pdf") as pdf:
            if mtype == "L23_ChC":
                mtype = "L23_CHC"
            _df = df[df.mtype == mtype]
            _df = _df[_df.use_axon]
            for gid in _df.index:
                plt.figure()
                view.plot_morph(nm.load_morphology(df.loc[gid, "path"]))
                plt.suptitle("")

                plt.axis("equal")
                pdf.savefig()
                if _df.loc[gid, "name"] in selected:
                    plt.axis([-1000, 1000, -1000, 1300])
                    plt.axis("off")
                    plt.gca().set_title("")
                    plt.savefig(f"bio_{mtype}.png", dpi=2000, transparent=True)
                plt.close()
