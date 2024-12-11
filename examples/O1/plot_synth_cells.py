import neurom as nm
from neurom import view
from matplotlib.backends.backend_pdf import PdfPages
import os
import matplotlib.pyplot as plt
import json
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
