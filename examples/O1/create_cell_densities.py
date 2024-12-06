import yaml
import sys

if __name__ == "__main__":
    with open("cell_composition.yaml") as f:
        comp = yaml.safe_load(f)
    for data in comp["neurons"]:
        data["density"] = float(sys.argv[-1])

    with open("cell_composition.yaml", "w") as f:
        yaml.safe_dump(comp, f)

    mtype = "L5_TPC:A"
    for data in comp["neurons"]:
        if data["traits"]["mtype"] == mtype:
            comp = {"neurons": [data], "version": "v2.0"}

    with open("cell_composition_red.yaml", "w") as f:
        yaml.safe_dump(comp, f)
