from cloudvolume import CloudVolume
import numpy as np
import pyvista as pv
import argparse
import pyvista as pv
from utils import np2pv


def download_img(cloud_path, mip, pos, physical_size):
    vol = CloudVolume(
        cloud_path, parallel=8, progress=True, mip=mip, cache=True, bounded=True
    )

    size = [ps / res for ps, res in zip(physical_size, vol.resolution)]
    size = np.array(size).astype("uint64")

    pos = np.array(pos, dtype=np.float32)
    pos[:2] /= 2  # account for different resolution online

    img = vol.download_point(pos, mip=mip, size=size).squeeze()
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cloudpath",
        help="path to cloud data",
        type=str,
        default="precomputed://gs://iarpa_microns/minnie/minnie65/seg",
    )
    parser.add_argument("--mip", help="resolution (0 is highest)", type=int, default=0)
    parser.add_argument(
        "--position",
        help="point position in x-y-z integer pixel position \
              (can be copied from neuroglancer)",
        type=str,
        default="225182-107314-19500",
    )
    parser.add_argument(
        "--size",
        help="cube side length of the volume to be downloaded (in nm)",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--output", help="output filename", type=str, default="data.xdmf"
    )
    args = parser.parse_args()

    position = args.position.split("-")
    size = [args.size] * 3
    img = download_img(args.cloudpath, args.mip, position, size)
    data = np2pv(img, img.resolution)
    data.save(args.output)
