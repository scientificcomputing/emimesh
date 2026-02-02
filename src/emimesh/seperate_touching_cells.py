from fenics import *
import argparse
import numpy as np

def seperate_touching_cells(sm):
    cellids = []
    mesh = sm.mesh()
    ecsid = 1
    for v in vertices(mesh):
        current_id = None
        for c in cells(v):
            label = sm[c]
            if label != ecsid:
                if current_id is None:
                    current_id = label
                elif current_id != label:
                    sm[c] = ecsid
                    cellids.append(c.index())
    return cellids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infile",
        help="input mesh",
        type=str,
    )
    parser.add_argument(
        "--output",
        help="output filename",
        type=str,
    )

    args = parser.parse_args()
    mesh = Mesh()
    with XDMFFile(args.infile) as mf:
        mf.read(mesh)
        sm = MeshFunction("size_t", mesh, 3, 0)
        mf.read(sm, "label")
    max_label = sm.array().max()

    print("seperating touching cells...")
    seperate_touching_cells(sm)
    sm.rename("label", "label")
    with XDMFFile(args.output) as outfile:
        outfile.write(sm)