import numpy as np
import argparse
import yaml
import dufte
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import matplotlib

for k, v in dufte.style.items():
    if "color" in k:
        dufte.style[k] = "black"

plt.style.use(dufte.style)


def plot_cell_sizes(mesh_statistics, filename):
    color_proc = "orange"
    color_orig = "teal"
    voxelcount = np.prod(mesh_statistics["size"])
    voxelvol = np.prod(mesh_statistics["resolution"])

    orig_vol_share = mesh_statistics["original_cell_counts"] / voxelcount
    vol_share = mesh_statistics["cell_counts"] / voxelcount
    orig_cell_vol_dict = dict(
        zip(mesh_statistics["original_cell_labels"], orig_vol_share)
    )

    cell_vol_dict = dict(zip(mesh_statistics["cell_labels"], vol_share))

    n_cells = len(mesh_statistics["cell_labels"])

    orig_cell_labels = list(mesh_statistics["original_cell_labels"])
    if 0 in orig_cell_labels:
        orig_cell_labels.remove(0)
    orig_n_cells = len(orig_cell_labels)
    x_largest_cells = np.flip(orig_cell_labels)[:n_cells]
    orig_largest_vols = [orig_cell_vol_dict[ci] for ci in x_largest_cells]
    largest_vols = [cell_vol_dict.get(ci, np.nan) for ci in x_largest_cells]
    rel_growth = (np.array(largest_vols) - np.array(orig_largest_vols)) / np.array(
        orig_largest_vols
    )

    sns.set_context("notebook")
    plt.figure(dpi=500)

    x = np.arange(len(x_largest_cells)) + 1
    sns.scatterplot(
        x=x,
        y=orig_largest_vols,
        marker="s",
        label=f"original \n ({orig_n_cells} cells)",
        color=color_orig,
    )
    sns.scatterplot(
        x=x,
        y=largest_vols,
        edgecolor="black",
        label=f"processed \n ({n_cells} cells)",
        color=color_proc,
    )

    plt.xlabel("cell number")
    plt.ylabel("cell volume share (%)")

    ax = plt.gca()
    ax.set_xmargin(0.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=2, frameon=False)

    ax2 = plt.twinx()
    sns.lineplot(
        x=x,
        y=np.cumsum(orig_largest_vols),
        markers=".-",
        color=color_orig,
    )
    sns.lineplot(
        x=x,
        y=np.cumsum(np.nan_to_num(largest_vols)),
        markers=".-",
        color=color_proc,
    )

    sns.scatterplot(
        x=0, y=[orig_cell_vol_dict[0]], edgecolor="black", color=color_orig, marker="D"
    )
    sns.scatterplot(
        x=0, y=[cell_vol_dict[0]], edgecolor="black", color=color_proc, marker="D"
    )
    ax2.yaxis.set_major_formatter(PercentFormatter(1))
    ax2.set_xmargin(0.05)
    ax.set_ylim((-0.01, 0.11))
    ax2.set_ylim((-0.1, 1.1))
    ax.set_yticks(np.linspace(0, 0.1, 6))
    ax2.set_yticks(np.linspace(0, 1.0, 6))
    ax2.set_ylabel("cumulative / ECS volume share (%)")

    ticks = np.arange(0, len(x), max(2, int(len(x) / 6)))
    ticklabels = list(ticks)
    ticklabels[0] = "ECS"
    ax.set_xticks(ticks, ticklabels)
    plt.tight_layout()
    plt.savefig(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infile",
        help="path to mesh statistics infile",
        type=str,
    )
    parser.add_argument(
        "--output",
        help="outfile name",
        type=str,
    )
    args = parser.parse_args()

    with open(args.infile) as infile:
        mesh_statistic = yaml.load(infile, Loader=yaml.FullLoader)

    plot_cell_sizes(mesh_statistic, args.output)
