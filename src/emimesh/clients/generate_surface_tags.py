import argparse

import numpy as np
import pyvista as pv
from imagemesh.facets import _raw_faces, mark_boundary_facets, mark_interface_facets
from imagemesh.io import read_mesh, save_mesh

from emimesh.evaluate_mesh import lstr


def main():
    parser = argparse.ArgumentParser(
        description="Tag the cell membranes and the outer boundary of a labelled tet mesh."
    )
    parser.add_argument("--infile", help="input mesh with cell data 'label'", type=str)
    parser.add_argument("--output", help="output facet mesh, e.g. facets.xdmf", type=str)

    args = parser.parse_args()
    mesh = read_mesh(args.infile)
    max_label = int(np.max(mesh[lstr]))
    outer_offset = int(10 ** np.ceil(np.log10(max_label)))
    save_mesh(mark_interfaces(mesh, outer_offset), args.output)


def mark_interfaces(mesh, outer_offset, label_array=lstr):
    """
    Facet mesh (sharing the points of mesh) with cell data 'boundaries': the
    interface between two labels is tagged with the larger label, an outer
    boundary facet with the label of its tet + outer_offset.
    """
    interfaces = mark_interface_facets(mesh, label_array=label_array)
    boundaries = mark_boundary_facets(mesh, label_array=label_array)
    faces = np.vstack([_raw_faces(interfaces), _raw_faces(boundaries)])
    tags = np.concatenate(
        [interfaces["region_b"], np.asarray(boundaries["boundary"]) + outer_offset]
    )
    cells = np.column_stack([np.full(len(faces), 3), faces]).ravel()
    facets = pv.PolyData(mesh.points, faces=cells)
    facets.cell_data["boundaries"] = tags
    return facets


if __name__ == "__main__":
    main()
