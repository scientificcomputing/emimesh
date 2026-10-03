import argparse

import pyvista as pv

from emimesh.ecs_share import adjust_ecs_share


def main():
    parser = argparse.ArgumentParser(
        description="Move the cell-ECS interfaces of a multi-label surface (with "
        "'boundary_labels') uniformly until the ECS takes a prescribed volume share. "
        "The bounding box is kept."
    )
    parser.add_argument("--infile", help="input surface, e.g. top_surf_smooth.vtk", type=str)
    parser.add_argument("--outfile", help="output surface", type=str)
    parser.add_argument("--ecs-share", help="target ECS volume share (0-1)", type=float)
    parser.add_argument(
        "--min-width",
        help="minimal thickness of cells and ECS gaps (length units of the surface)",
        type=float,
        default=10.0,
    )
    args = parser.parse_args()
    surf = pv.read(args.infile)
    out, d, share = adjust_ecs_share(surf, args.ecs_share, min_width=args.min_width)
    print(f"offset {d:.3f}, ECS share {share:.5f}")
    out.save(args.outfile)


if __name__ == "__main__":
    main()
