
import numpy as np
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
        surface_areas.append(float(surf.compute_cell_sizes()["Area"].sum()))
        volumes.append(float(cell["Volume"].sum()))
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

