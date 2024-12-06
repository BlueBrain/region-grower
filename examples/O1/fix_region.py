import json
from voxcell.cell_collection import CellCollection

if __name__ == "__main__":
    params = json.load(open("tmd_parameters.json"))
    distr = json.load(open("tmd_distributions.json"))
    try:
        params = {"O0": params["Isocortex"]}
        distr = {"O0": distr["Isocortex"]}
        json.dump(params, open("tmd_parameters.json", "w"), indent=4)
        json.dump(distr, open("tmd_distributions.json", "w"), indent=4)
    except:
        pass


    cells = CellCollection.load("nodes.h5")
    cells.properties["region"] = "O0"
    cells.save('nodes.h5')
