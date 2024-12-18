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

import matplotlib.pyplot as plt
import neurom as nm
from matplotlib.backends.backend_pdf import PdfPages
from neurom import view
from voxcell.cell_collection import CellCollection

if __name__ == "__main__":
    df = CellCollection.load("nodes_synthesis.h5").as_dataframe()
    df["path"] = "morphologies/" + df["morphology"] + ".asc"
    mtypes = json.loads(os.environ["MTYPES"])
    for mtype in mtypes:
        plot = True
        with PdfPages(f"synth_{mtype}.pdf") as pdf:
            _df = df[df.mtype == mtype]
            print(_df)
            for gid in _df.index:
                plt.figure()
                view.plot_morph(
                    nm.load_morphology(df.loc[gid, "path"])
                )  # , realistic_diameters=True)
                plt.axis("equal")
                pdf.savefig()
                if plot:
                    plt.axis([-1000, 1000, -1000, 1300])
                    plt.axis("off")
                    plt.gca().set_title("")
                    plt.savefig(f"synth_{mtype}.png", dpi=2000, transparent=True)
                    plot = False
                plt.close()
