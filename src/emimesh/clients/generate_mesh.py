import argparse
import time

import pyvista as pv
from imagemesh.curved import (
    adaptive_snap_boundaries,
    convert_to_quadratic,
    print_quality_stats,
    straighten_inverted_cells,
)
from imagemesh.tetmesh import mesh_surface

from emimesh.evaluate_mesh import lstr


def main():
    parser = argparse.ArgumentParser(
        description="Simplify a multi-label surface and tetrahedralize it with fTetWild."
    )
    parser.add_argument("--infile", help="multi-label surface with 'boundary_labels'", type=str)
    parser.add_argument(
        "--envelopsize",
        help="absolute size of the fTetWild surface envelope (in nm)",
        type=float,
    )
    parser.add_argument("--stopquality", help="fTetWild mesh quality score", type=float, default=10)
    parser.add_argument(
        "--simplify_eps",
        help="absolute distance tolerance of the surface simplification (in nm), 0 to skip, "
        "default 0.5 voxels",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--edge_length_fac",
        help="fTetWild target edge length relative to the bounding box diagonal",
        type=float,
        default=0.05,
    )
    parser.add_argument("--output", help="output filename", type=str)
    parser.add_argument(
        "--surface_output", help="optional output of the simplified surface", type=str
    )
    parser.add_argument("--max_threads", help="max number of threads", type=int, default=1)
    parser.add_argument(
        "--curved",
        help="curved (10-node) tets, with their boundary and interface nodes snapped onto "
        "the unsimplified surface (True or False)",
        type=str_to_bool,
        default=False,
    )

    args = parser.parse_args()
    surf = pv.read(args.infile)
    if args.simplify_eps is None:
        args.simplify_eps = 0.5 * float(surf.field_data["dx"][0])
    volmesh, dec = mesh_surface(
        surf,
        envelopsize=args.envelopsize,
        simplify_eps=args.simplify_eps,
        stop_energy=args.stopquality,
        edge_length_fac=args.edge_length_fac,
        num_threads=args.max_threads,
    )
    if args.curved:
        volmesh = curve_mesh(volmesh, surf)
    if args.surface_output:
        dec.save(args.surface_output)
    pv.save_meshio(args.output, volmesh)
    print(volmesh.array_names)


def str_to_bool(value):
    """Parse True/False as written by the Snakefile from a config value."""
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected True or False, got {value!r}")


def curve_mesh(mesh, surf):
    """
    Quadratic version of the labelled tet mesh, with the nodes of its boundary
    and interface facets snapped onto the sheets of surf ('boundary_labels')
    with the same label pair, without inverting any cell.
    """
    start = time.time()
    quad = convert_to_quadratic(mesh)
    for name in mesh.field_data.keys():
        quad.field_data[name] = mesh.field_data[name]
    adaptive_snap_boundaries(quad, surf, label_array=lstr)
    n_bad = straighten_inverted_cells(quad)
    if n_bad:
        print(f"warning: {n_bad} cells are still inverted")
    print_quality_stats(quad, "curved mesh")
    quad.field_data["runtime"] = mesh.field_data["runtime"] + time.time() - start
    return quad


if __name__ == "__main__":
    main()
