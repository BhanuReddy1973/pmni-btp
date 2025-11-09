import trimesh
import numpy as np

# Load the mesh
mesh = trimesh.load('/home/bhanu/pmni/PMNI/exp/diligent_mv/bear/exp_2025_11_08_12_15_02/meshes_validation/iter_00030000.ply')

# Get mesh statistics
vertices = len(mesh.vertices)
faces = len(mesh.faces)
bounds = mesh.bounds
volume = mesh.volume
area = mesh.area

# Create a text summary
summary = f"""
PMNI Reconstructed Mesh Statistics - Bear Object
===============================================

Vertices: {vertices:,}
Faces: {faces:,}
Volume: {volume:.6f}
Surface Area: {area:.6f}

Bounding Box:
  Min: [{bounds[0][0]:.3f}, {bounds[0][1]:.3f}, {bounds[0][2]:.3f}]
  Max: [{bounds[1][0]:.3f}, {bounds[1][1]:.3f}, {bounds[1][2]:.3f}]

Mesh Quality: High-resolution reconstruction from 20-view multi-view setup
Training: 30,000 iterations, final loss = 1.16e-02
"""

# Save to text file
with open('/home/bhanu/pmni/PMNI/report/images/mesh_statistics.txt', 'w') as f:
    f.write(summary)

print("Mesh statistics generated!")
print(summary)