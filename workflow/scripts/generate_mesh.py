import wildmeshing as wm
import json
import meshio
import pyvista as pv
import argparse
import os
import numpy as np
from utils import get_bounding_box
from pathlib import Path
from functools import reduce
import operator


def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)


def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


def get_values(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from get_values(v)
        else:
            yield v


def mesh_surfaces(csg_tree, eps, stop_quality, max_threads):
    tetra = wm.Tetrahedralizer(
        epsilon=eps,
        edge_length_r=eps * 20,
        coarsen=True,
        stop_quality=stop_quality,
        max_threads=max_threads,
    )
    tetra.load_csg_tree(json.dumps(csg_tree))
    tetra.tetrahedralize()
    point_array, cell_array, marker = tetra.get_tet_mesh()

    volmesh = pv.from_meshio(
        meshio.Mesh(
            point_array, [("tetra", cell_array)], cell_data={"label": [marker.ravel()]}
        )
    )
    return volmesh


def get_screenshot(mesh, filename):
    pv.start_xvfb()
    p = pv.Plotter(off_screen=True)
    p.add_mesh(mesh, cmap="rainbow", show_scalar_bar=False)
    p.camera_position = "yz"
    p.camera.azimuth = 225
    p.camera.elevation = 20
    p.screenshot(filename, transparent_background=True)


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
        "--max_threads", help="max number of threads", type=int, default=0
    )

    args = parser.parse_args()
    with open(args.csgtree) as f:
        csgtree = json.load(f)
    surfs = [surf for surf in get_values(csgtree) if "stl" in surf]
    shrunk_bbox_file = Path(args.output).parent / "bbox.stl"
    bboxfile = [s for s in surfs if "bbox.stl" in s][0]
    bbox = pv.read(bboxfile)
    diag = np.sqrt(3) * bbox.volume ** (1 / 3)
    es = args.envelopsize
    shrunk_bbox = get_bounding_box([bbox], 1.5 * es)
    shrunk_bbox.save(shrunk_bbox_file)
    setInDict(csgtree, ["left"] * (len(surfs) - 1), str(shrunk_bbox_file))
    volmesh = mesh_surfaces(
        csgtree,
        eps=es / diag,
        stop_quality=args.stopquality,
        max_threads=args.max_threads,
    )
    os.remove("__tracked_surface.stl")
    pv.save_meshio(args.output, volmesh)

    screenshotfile = Path(args.output).parent / "mesh.png"

    get_screenshot(volmesh.threshold(1.5), screenshotfile)
