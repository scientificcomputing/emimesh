import argparse
import pyvista as pv
from plot_utils import get_screenshot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infile",
        help="path to mesh file",
        type=str,
    )
    parser.add_argument(
        "--output",
        help="outfile name",
        type=str,
    )

    args = parser.parse_args()
    mesh = pv.read(args.infile)
    bounds = list(mesh.bounds)
    bounds[1] = (bounds[0] + bounds[1]) * 0.5
    bounds[3] = (bounds[2] + bounds[3]) * 0.5
    bounds[4] = (bounds[4] + bounds[5]) * 0.5
    #mesh = mesh.clip_box(bounds)
    get_screenshot(mesh.threshold(1.5, scalars="label"), args.output,scalar="label", cmap="rainbow")