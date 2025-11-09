import pyvista as pv
import numpy as np

# Start virtual framebuffer for headless rendering
pv.start_xvfb()

# Load the mesh
mesh = pv.read('/home/bhanu/pmni/PMNI/exp/diligent_mv/bear/exp_2025_11_08_12_15_02/meshes_validation/iter_00030000.ply')

# Create a plotter
plotter = pv.Plotter(off_screen=True, window_size=(800, 600))

# Add mesh
plotter.add_mesh(mesh, color='lightblue', show_edges=False)

# Set camera position for better view
plotter.camera_position = 'iso'

# Save screenshot
plotter.screenshot('/home/bhanu/pmni/PMNI/report/images/reconstructed_mesh.png')

# Different angles
plotter.camera_position = 'xy'
plotter.screenshot('/home/bhanu/pmni/PMNI/report/images/mesh_top_view.png')

plotter.camera_position = 'xz'
plotter.screenshot('/home/bhanu/pmni/PMNI/report/images/mesh_side_view.png')

plotter.close()

print("Mesh images generated!")