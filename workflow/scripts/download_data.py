from cloudvolume import CloudVolume
import webknossos as wk
import numpy as np
import pyvista as pv
import argparse
import pyvista as pv
from utils import np2pv
import dask.array as da
from pathlib import Path

def download_webknossos(cloud_path, mip, pos, physical_size):
    target_mag=wk.Mag([2,2,1])
    voxel_size = np.array([11.24,11.24, 28])
    mag_voxel_size = (voxel_size * target_mag.to_np())
    mag = wk.Mag(1)
    size = [int(ps / vs) for ps, vs in zip(physical_size, voxel_size)]
    bbox = wk.BoundingBox(pos, size=size)
    bbox.align_with_mag(target_mag)
    ds = wk.Dataset.download(cloud_path, mags=[mag], path=f"data/{mip}_{physical_size}", exist_ok=True,
                             bbox=bbox, layers="segmentation")
    layer = ds.get_layer("segmentation")
    layer.downsample_mag(from_mag=mag, target_mag=target_mag, allow_overwrite=True)
    mag_view = layer.get_mag(target_mag)
    img = mag_view.read().squeeze()
    assert img.max() > 0
    return img, mag_voxel_size


def download_cloudvolume(cloud_path, mip, pos, physical_size):
    vol = CloudVolume(
        cloud_path, parallel=8, progress=True, mip=mip, cache=True, bounded=True
    )

    size = [ps / res for ps, res in zip(physical_size, vol.resolution)]
    size = np.array(size).astype("uint64")

    pos = np.array(pos, dtype=np.float32)
    pos[:2] /= 2  # account for different resolution online

    img = vol.download_point(pos, mip=mip, size=size).squeeze()
    return img, vol.resolution


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
    try:
        img,res = download_cloudvolume(args.cloudpath, args.mip, position, size)
    except:
        img,res = download_webknossos(args.cloudpath, args.mip, position, size)
    print(res)
    data = np2pv(img, res)
    Path(args.output).parent.mkdir(exist_ok=True)
    data.save(args.output)
