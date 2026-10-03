import argparse

import pyvista as pv

from emimesh.generate_mesh import mesh_surface


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

    args = parser.parse_args()
    surf = pv.read(args.infile)
    if args.simplify_eps is None:
        args.simplify_eps = 0.5 * float(surf.field_data["dx"][0])
    volmesh, surf = mesh_surface(
        surf,
        envelopsize=args.envelopsize,
        simplify_eps=args.simplify_eps,
        stop_quality=args.stopquality,
        edge_length_fac=args.edge_length_fac,
        max_threads=args.max_threads,
    )
    if args.surface_output:
        surf.save(args.surface_output)
    pv.save_meshio(args.output, volmesh)
    print(volmesh.array_names)


if __name__ == "__main__":
    main()
