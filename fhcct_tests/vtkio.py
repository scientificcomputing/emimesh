"""Legacy (v2.0 ASCII) VTK I/O for the FHC_CT dt executable."""

import numpy as np


def write_surface(fname, points, tris):
    with open(fname, "w") as f:
        f.write("# vtk DataFile Version 2.0\nsurface\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {len(points)} double\n")
        np.savetxt(f, points, fmt="%.17g")
        f.write(f"POLYGONS {len(tris)} {4 * len(tris)}\n")
        np.savetxt(f, np.column_stack([np.full(len(tris), 3), tris]), fmt="%d")
