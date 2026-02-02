import argparse
import yaml
from emimesh.generate_analysis_plots import plot_cell_sizes

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
