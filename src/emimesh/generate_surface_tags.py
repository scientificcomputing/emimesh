from fenics import *
import argparse
import numpy as np


def mark_interfaces(mesh, subdomains, outer_offset):
    bm = MeshFunction("size_t", mesh, 2, 0)
    bm.rename("boundaries", "")
    for f in facets(mesh):
        domains = []
        for c in cells(f):
            domains.append(subdomains[c])
        domains = list(set(domains))
        domains.sort()
        if f.exterior():
            bm[f] = domains[0] + outer_offset
            continue
        if len(domains) < 2:
            continue
        bm[f] = domains[1]
    return bm


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
    outer_offset = int(10 ** np.ceil(np.log10(max_label)))
    bm = mark_interfaces(mesh, sm, outer_offset)

    with XDMFFile(args.output) as outfile:
        outfile.write(bm)
