import argparse
import pyvista as pv
from emimesh.plot_utils import get_screenshot
import numpy as np
import matplotlib
import cmocean
import fastremap

hexcolor = lambda c: int(matplotlib.colors.to_hex(c)[1:], base=16)

def main():
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
    parser.add_argument(
        "--exclude",
        help="cell ids to exclude",
        type=int,
        nargs="+"
    )


    args = parser.parse_args()
    mesh = pv.read_meshio(args.infile)
    bounds = list(mesh.bounds)
    bounds[1] = (bounds[0] + bounds[1]) * 0.5
    bounds[3] = (bounds[2] + bounds[3]) * 0.5
    bounds[4] = (bounds[4] + bounds[5]) * 0.5
    #mesh = mesh.clip_box(bounds)

    cells = mesh.threshold(1.5, scalars="label")
    if args.exclude is not None:
        cells = cells.extract_cells(np.isin(cells["label"], args.exclude)==False)
    cells["label"] = cells["label"].astype(np.int32)
    cellids = fastremap.unique(cells["label"])
    shuffled = list(cellids)
    np.random.shuffle(shuffled)
    cells["label"] = fastremap.remap(cells["label"], {k:v for k,v in zip(cellids, shuffled)})
    
    newcmap = cmocean.tools.crop_by_percent(cmocean.cm.curl_r, 5, which='max', N=None)
    get_screenshot(cells, args.output, scalar="label", cmap="curl")
    
    #import k3d
    #ecs = mesh.extract_cells((mesh["label"]==1))

    #color_map =k3d.paraview_color_maps.Rainbow_Uniform
    #color_map = k3d.paraview_color_maps.Linear_Green_Gr4L
    #cells = cells.cell_data_to_point_data()
    #cells_k3d = k3d.vtk_poly_data(cells.extract_surface(), side="double",
                              #color_attribute=("label", 0, float(cells["label"].max())), 
                              #color_map=color_map, name="cells"
    #                          color=hexcolor("limegreen"))
    #ecs_k3d = k3d.vtk_poly_data(ecs.extract_surface(), side="double", color=hexcolor("white"),
    #                            opacity=0.8,
    #                            name="ecs", wireframe=True)

    #generate_screenshots([cells_k3d, ecs_k3d], args.output, fov=32)


    

if __name__ == "__main__":
    main()