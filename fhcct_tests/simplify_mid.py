"""Simplified mid surface (same settings as split_blocks.py for top/bottom: epsilon = 0.05 dx)."""

import pyvista as pv

from emimesh.surface_simplification import simplify_surface

S = pv.read("../../emimesh/mid_0.15.vtk").clean()
print("input faces", S.n_cells)
S = simplify_surface(S, epsilon=20 * 0.05)
print("simplified faces", S.n_cells)
S.save("mid_dec.vtk")
