"""Create the pia mesh for O1 atlas."""
import os

from neurocollage.mesh_helper import MeshHelper

if __name__ == "__main__":
    mesh_helper = MeshHelper({"atlas": "atlas", "structure": "region_structure.yaml"}, "O0")
    mesh = mesh_helper.get_pia_mesh()
    # slightly shift upwards to handle the fact that there are no outer voxcels in O1 atlas,
    # hence an artificial shift of the outer boundaries
    mesh.apply_translation([0, 2, 0])
    mesh.export("pia_mesh.obj")
