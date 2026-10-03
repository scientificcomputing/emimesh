import argparse
from pathlib import Path

import pyvista as pv

from emimesh.extract_surfaces import extract_surface


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
        "--smoothing_iterations", help="number of smoothing iterations", type=int, default=16
    )
    args = parser.parse_args()

    print(f"reading file: {args.infile}")
    imggrid = pv.read(args.infile)
    surf = extract_surface(
        imggrid, smoothing_scale=args.smoothing_scale, iterations=args.smoothing_iterations
    )
    print(f"surface: {surf.n_points} points, {surf.n_cells} faces")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    surf.save(args.output)


if __name__ == "__main__":
    main()
