"""Create the pia mesh for O1 atlas."""
import numpy as np
from neurocollage.mesh_helper import MeshHelper

if __name__ == "__main__":
    mesh_helper = MeshHelper({"atlas": "atlas", "structure": "region_structure.yaml"}, "O0")
    mesh = mesh_helper.get_pia_mesh()
    # slightly shift upwards to handle the fact that there are no outer voxcels in O1 atlas,
    # hence an artificial shift of the outer boundaries
    mesh.apply_translation([0, 2, 0])
    mesh.export("pia_mesh.obj")

    meshes = mesh_helper.get_layer_meshes()
    for i, mesh in enumerate(meshes):
        mesh.export(f"L{i+1}_mesh.obj")
        # meshes to use with brayns, in world coordinate, not voxel ids
        mesh.vertices = mesh.vertices * 10.0 + np.array([-810.0, -10.0, -700.0])
        mesh.export(f"L{i+1}_mesh_viz.obj", include_color=False, include_texture=False)
