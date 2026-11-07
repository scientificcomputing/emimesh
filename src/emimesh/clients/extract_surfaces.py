import argparse
from pathlib import Path

import pyvista as pv

from emimesh.extract_surfaces import extract_surface


def _float_or_none(v):
    return None if str(v).lower() == "none" else float(v)


def _int_or_none(v):
    return None if str(v).lower() == "none" else int(v)


def main():
    parser = argparse.ArgumentParser(
        description="Extract the smoothed multi-label surface (surface nets) of a label image."
    )
    parser.add_argument("--infile", help="path to the processed image data", type=str)
    parser.add_argument("--output", help="output surface, e.g. surf_smooth.vtk", type=str)
    parser.add_argument(
        "--smoothing_scale",
        help="maximal displacement of the surface points during smoothing, in voxels",
        type=float,
        default=1.2,
    )
    parser.add_argument(
        "--smoothing_iterations",
        help="number of smoothing iterations (default: 16 for laplace; fair: 100, or as "
        "many as influence_radius allows)",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--method",
        help="laplace: constrained Laplacian smoothing (VTK); "
        "fair: volume-preserving fairing bounded by the voxel surface",
        choices=["laplace", "fair"],
        default="laplace",
    )
    parser.add_argument(
        "--influence_radius",
        help="fair: range of influence in voxels: no point depends on a voxel further away "
        "('none': no limit)",
        type=_int_or_none,
        default=None,
    )
    parser.add_argument("--alpha", help="fair: smoothness weight", type=float, default=30.0)
    parser.add_argument(
        "--dead_zone",
        help="fair: free normal motion around the start, in voxels ('none': no anchor)",
        type=_float_or_none,
        default=0.5,
    )
    parser.add_argument(
        "--max_distance",
        help="fair: maximal distance from the voxel surface, in voxels ('none': no limit)",
        type=_float_or_none,
        default=1.0,
    )
    parser.add_argument(
        "--volume",
        help="fair: keep the volume of every label",
        type=lambda v: str(v).lower() in ("1", "true", "yes"),
        default=True,
    )
    args = parser.parse_args()

    print(f"reading file: {args.infile}")
    imggrid = pv.read(args.infile)
    kwargs = {}
    if args.method == "fair":
        kwargs = dict(
            method="fair",
            influence_radius=args.influence_radius,
            alpha=args.alpha,
            dead_zone=args.dead_zone,
            max_distance=args.max_distance,
            volume=args.volume,
        )
    surf = extract_surface(
        imggrid,
        smoothing_scale=args.smoothing_scale,
        iterations=args.smoothing_iterations,
        **kwargs,
    )
    print(f"surface: {surf.n_points} points, {surf.n_cells} faces")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    surf.save(args.output)


if __name__ == "__main__":
    main()
