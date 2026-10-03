import argparse
import time
import pyvista as pv
from pathlib import Path
import numba
import numpy as np
import fastremap
import dask
import yaml
from emimesh.process_image_data import process_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", help="input data", type=str)
    parser.add_argument(
        "--output", help="output filename", type=str, default="processeddata.vti"
    )
    parser.add_argument("--nworkers", help="number of workers", type=int, default=1)
    parser.add_argument("--dx", help="target resolution", type=int, default=None)
    parser.add_argument("--ncells", help="number of cells", type=int, default=None)
    parser.add_argument('-o','--operation', nargs='+', action='append', help="operations to be performed on the segmented image")

    args = parser.parse_args()
    n_parallel = args.nworkers
    print(f"Using {n_parallel} workers...")
    start = time.time()

    # read image file
    imggrid = pv.read(args.infile)

    imggrid, combinedmap, cell_labels, cell_counts = process_image(imggrid, 
                                                                   dx=args.dx,
                                           operations=args.operation,
                                           ncells=args.ncells, 
                                           num_threads=args.nworkers)
    
    mesh_statistics = dict()
    mesh_statistics["cell_labels"] = cell_labels
    mesh_statistics["cell_counts"] = cell_counts
    mesh_statistics["mapping"] = combinedmap

    for k, v in mesh_statistics.items():
        mesh_statistics[k] = np.array(v).tolist()

    resdir = Path(args.output).parent

    with open(resdir / "imagestatistic.yml", "w") as mesh_stat_file:
        yaml.dump(mesh_statistics, mesh_stat_file)

    resdir.mkdir(parents=True, exist_ok=True)
    imggrid.save(args.output)

if __name__ == "__main__":
    main()