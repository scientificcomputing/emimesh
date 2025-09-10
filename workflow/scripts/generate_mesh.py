import json
import meshio
import pyvista as pv
import argparse
import numpy as np
import time

def get_values(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from get_values(v)
        else:
            yield v

def mesh_surfaces(csg_tree, eps, stop_quality, max_threads):
    import wildmeshing as wm
    tetra = wm.Tetrahedralizer(
        epsilon=eps,
        edge_length_r=eps * 50,
        coarsen=True,
        stop_quality=stop_quality,
        max_threads=max_threads,
        skip_simplify=False,
    )
    tetra.load_csg_tree(json.dumps(csg_tree))
    start = time.time()
    tetra.tetrahedralize()
    point_array, cell_array, marker = tetra.get_tet_mesh()

    volmesh = pv.from_meshio(
        meshio.Mesh(
            point_array, [("tetra", cell_array)], cell_data={"label": [marker.ravel()]}
        )
    )
    volmesh.field_data['runtime'] = time.time() - start
    volmesh.field_data['threads'] = max_threads
    return volmesh

if __name__ == "__main__":
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
    es = args.envelopsize
    csgtree = {"operation":"intersection","right":csgtree, "left":roifile}
    volmesh = mesh_surfaces(
        csgtree,
        eps=es / diag,
        stop_quality=args.stopquality,
        max_threads=args.max_threads,
    )
    pv.save_meshio(args.output, volmesh)
