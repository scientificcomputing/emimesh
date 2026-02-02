import json
import pyvista as pv
import argparse
import numpy as np
from emimesh.generate_mesh import mesh_surfaces

def get_values(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from get_values(v)
        else:
            yield v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csgtree",
        help="path to csgtree file",
        type=str,
    )
    parser.add_argument(
        "--envelopsize",
        help="absolut size of fTetWild surface envelop (in nm)",
        type=float,
    )
    parser.add_argument(
        "--stopquality", help="fTetWild mesh quality score", type=float, default=10
    )
    parser.add_argument(
        "--output",
        help="output filename",
        type=str,
    )
    parser.add_argument(
        "--max_threads", help="max number of threads", type=int, default=1
    )

    args = parser.parse_args()
    with open(args.csgtree) as f:
        csgtree = json.load(f)
    surfs = [surf for surf in get_values(csgtree) if "ply" in surf]
    roifile = [s for s in surfs if "roi.ply" in s][0]
    roi = pv.read(roifile)
    diag = np.sqrt(3) * roi.volume ** (1 / 3)
    abs_eps = args.envelopsize    
    volmesh = mesh_surfaces(
        args.csgtree,
        eps=abs_eps / diag,
        stop_quality=args.stopquality,
        max_threads=args.max_threads,
    )
    pv.save_meshio(args.output, volmesh)
    print(volmesh.array_names)



if __name__ == "__main__":
    main()