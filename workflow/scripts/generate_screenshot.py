import argparse
import pyvista as pv
import k3d
from plot_utils import get_screenshot
from pathlib import Path
import numpy as np
import matplotlib

hexcolor = lambda c: int(matplotlib.colors.to_hex(c)[1:], base=16)

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
    parser.add_argument(
        "--exclude",
        help="cell ids to exclude",
        type=int,
        nargs="+"
    )



    args = parser.parse_args()
    mesh = pv.read(args.infile)
    bounds = list(mesh.bounds)
    bounds[1] = (bounds[0] + bounds[1]) * 0.5
    bounds[3] = (bounds[2] + bounds[3]) * 0.5
    bounds[4] = (bounds[4] + bounds[5]) * 0.5
    #mesh = mesh.clip_box(bounds)

    cells = mesh.threshold(1.5, scalars="label")
    if args.exclude is not None:
        cells = cells.extract_cells(np.isin(cells["label"], args.exclude)==False)
    get_screenshot(cells, args.output, scalar="label", cmap="rainbow")

    ecs = mesh.extract_cells((mesh["label"]==1))

    color_map =k3d.paraview_color_maps.Rainbow_Uniform
    color_map = k3d.paraview_color_maps.Linear_Green_Gr4L
    cells = cells.cell_data_to_point_data()
    cells_k3d = k3d.vtk_poly_data(cells.extract_surface(), side="double",
                              color_attribute=("label", 0, float(cells["label"].max())), 
                              color_map=color_map, name="cells")
    ecs_k3d = k3d.vtk_poly_data(ecs.extract_surface(), side="double", color=hexcolor("white"),
                                opacity=0.3, name="ecs", wireframe=True)

    pl = k3d.plot(
        camera_rotate_speed=3,
        camera_zoom_speed=5,
        screenshot_scale=1,
        background_color=0x000000,
        grid_visible=False,
        camera_auto_fit=True,
        axes_helper=False,
        lighting=2
        )
    pl += cells_k3d
    pl += ecs_k3d

    with open(Path(args.output).with_suffix(".html"), 'w') as f:
        f.write(pl.get_snapshot())
