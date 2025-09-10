import pyvista as pv
import numpy as np
import argparse
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

lstr = "label"
ecs_id = 1


def compute_local_width(mesh, ecs_id, cell_ids):
    ecs = mesh.extract_cells(np.isin(mesh.cell_data[lstr], ecs_id))
    distances = []
    for cid in cell_ids:
        cell = mesh.extract_cells(np.isin(mesh.cell_data[lstr], [cid]))
        dist = ecs.compute_implicit_distance(cell.extract_surface())
        distances.append(abs(dist.point_data["implicit_distance"]))

    dist = ecs.compute_implicit_distance(mesh.extract_surface())
    distances.append(abs(dist.point_data["implicit_distance"]))

    distances = np.array(distances).T
    distances.sort()
    min_dist = distances[:, 0] + distances[:, 1]
    ecs["local_width"] = min_dist
    ecs = ecs.point_data_to_cell_data()
    ecs = ecs.compute_cell_sizes()
    return ecs.cell_data["local_width"], abs(ecs.cell_data["Volume"])

def compute_surface_volume(mesh, cell_ids):
    mesh = mesh.compute_cell_sizes()
    mesh["Volume"] = np.abs(mesh["Volume"])
    assert (mesh["Volume"] > 0).all()
    volumes, surface_areas = [], []
    for cid in cell_ids:
        cell = mesh.extract_cells(np.isin(mesh.cell_data[lstr], [cid]))
        surf = cell.extract_surface()
        surface_areas.append(surf.compute_cell_sizes()["Area"].sum())
        volumes.append(cell["Volume"].sum())
    return volumes, surface_areas


def plot_local_width(width, volume, filename):
    plt.figure(dpi=300)
    sns.histplot(
        x=width,
        weights=volume,
        bins=40,
        kde=True,
        stat="percent",
        edgecolor="white",
        kde_kws={"bw_adjust": 1},
    )
    plt.xlabel("local width (nm)")
    plt.ylabel("relative frequency (%)")
    plt.tight_layout()
    plt.savefig(filename)

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

    args = parser.parse_args()
    mesh = pv.read_meshio(args.infile)
    cell_ids = list(np.unique(mesh[lstr]))
    cell_ids.remove(ecs_id)
    #ecs_width, ecs_cell_volume = compute_local_width(mesh, ecs_id, cell_ids=cell_ids)
    cell_volume, cell_surface = compute_surface_volume(mesh, cell_ids)
    ecs_volume, ecs_surface = compute_surface_volume(mesh, [ecs_id])
    ecs_volume, ecs_surface = ecs_volume[0], ecs_surface[0]
    mesh_volume = ecs_volume + sum(cell_volume)
    ecs_share = ecs_volume / mesh_volume
    ecs_mesh = mesh.extract_cells(mesh[lstr]==ecs_id)
    ecs_surf_mesh = ecs_mesh.extract_surface()
    boundary_mesh = mesh.extract_surface()
    cell_boundary_mesh = boundary_mesh.extract_cells(boundary_mesh[lstr]>ecs_id)
    ecs_boundary_mesh = boundary_mesh.extract_cells(boundary_mesh[lstr]==ecs_id)
    n_ecs_boundary_points = boundary_mesh.number_of_points - cell_boundary_mesh.number_of_points
    results = dict(npoints=mesh.number_of_points,
                  ncompcells=mesh.number_of_cells,
                  ecs_volume=ecs_volume, ecs_surface=ecs_surface,
                  cell_surface=cell_surface, cell_volume=cell_volume,
                  ecs_share=ecs_share,
                  npoints_membrane=ecs_surf_mesh.number_of_points - n_ecs_boundary_points,
                  npoints_boundary=boundary_mesh.number_of_points,
                  npoints_ecs=ecs_mesh.number_of_points,
                  ntets_ecs=ecs_mesh.number_of_cells,
                  ntets_ics=mesh.number_of_cells - ecs_mesh.number_of_cells,
                  nfacets_membrane = ecs_surf_mesh.number_of_cells - ecs_boundary_mesh.number_of_cells,
                  #ecs_width=ecs_width, ecs_cell_volume=ecs_cell_volume,
                  )
    if "runtime" in mesh.array_names:
        results["runtime"] = mesh["runtime"]
    if "threads" in mesh.array_names:
        results["threads"] = mesh["threads"]

    with open(args.output, "w") as outfile:
        yaml.dump(results, outfile)

    #plot_local_width(width, volume, Path(args.infile).parent / "local_width.png")