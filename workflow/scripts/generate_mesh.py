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
import time

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
    import wildmeshing as wm
    tetra = wm.Tetrahedralizer(
        epsilon=eps,
        edge_length_r=eps * 200,
        coarsen=True,
        stop_quality=stop_quality,
        max_threads=max_threads,
        skip_simplify=False,
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

def mesh_surfaces_os(csg_tree, eps, stop_quality, max_threads, outdir):

    csg_path = f"{outdir}/csgtree.json"
    with open(csg_path, "w") as f:
        f.write(json.dumps(csg_tree))
    start = time.time()
    cmd = (f"fTetWild/build/FloatTetwild_bin --csg {csg_path} " + 
           f"--max-threads {max_threads} -e {eps} " + 
           f"--stop-energy {stop_quality} --level 2 " +
           f"--output {outdir}/mesh.msh")
    import subprocess
    rtn = subprocess.run(cmd, shell=True)
    #from IPython import embed; embed()
    #if  rtn.returncode != 0:
        # sometimes the simpflication process fails during edge swapping (segmentation fault).
        # in that case, repeat without simplification
    #    subprocess.run(cmd + " --skip-simplify", shell=True)
    print("mesh generation finished...")
    mesh = pv.read(f"{outdir}/mesh.msh").clean()
    mesh.cell_data.set_array(mesh["gmsh:physical"], "label", deep_copy=True)
    mesh.cell_data.remove("gmsh:geometrical")
    mesh.cell_data.remove("gmsh:physical")
    mesh.cell_data.remove("color")
    mesh.field_data['runtime'] = time.time() - start
    mesh.field_data['threads'] = max_threads
    return mesh


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
    outdir = Path(args.output).parent
    roifile = [s for s in surfs if "roi.ply" in s][0]
    roi = pv.read(roifile)
    diag = np.sqrt(3) * roi.volume ** (1 / 3)
    es = args.envelopsize
    csgtree = {"operation":"intersection","right":csgtree, "left":roifile}
    volmesh = mesh_surfaces_os(
        csgtree,
        eps=es / diag,
        stop_quality=args.stopquality,
        max_threads=args.max_threads,
        outdir=str(outdir)
    )
    pv.save_meshio(args.output, volmesh)
