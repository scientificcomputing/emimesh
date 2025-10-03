import numpy as np
import pyvista as pv
import argparse
from utils import np2pv
from pathlib import Path

def download_webknossos(cloud_path, mip, pos, physical_size):
    import webknossos as wk
    target_mag=wk.Mag([2,2,1])
    voxel_size = np.array([11.24,11.24, 28])
    mag_voxel_size = (voxel_size * target_mag.to_np())
    mag = wk.Mag(1)
    size = [int(ps / vs) for ps, vs in zip(physical_size, voxel_size)]
    bbox = wk.BoundingBox(pos, size=size)
    bbox.align_with_mag(target_mag)
    ds = wk.Dataset.download(cloud_path, mags=[mag], path=f".cache/webknossos/{mip}_{physical_size}",
                             bbox=bbox, layers="segmentation")
    layer = ds.get_layer("segmentation")
    layer.downsample_mag(from_mag=mag, target_mag=target_mag, allow_overwrite=True)
    mag_view = layer.get_mag(target_mag)
    img = mag_view.read().squeeze()
    assert img.sum() > 0, "dataset empty!"
    return img, mag_voxel_size


def download_cloudvolume(cloud_path, mip, pos, physical_size):
    from cloudvolume import CloudVolume
    vol = CloudVolume(
        cloud_path, use_https=True, parallel=8, progress=True, mip=mip, cache=True, bounded=True
    )
    print(f"data resoltion: {vol.resolution}")
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
    )
    parser.add_argument(
        "--size",
        help="cube side length of the volume to be downloaded (in nm)",
        type=str,
        default=1000,
    )
    parser.add_argument(
        "--output", help="output filename", type=str, default="data.xdmf"
    )

    args = parser.parse_args()

    position = args.position.split("-")
    try:
        size = [int(args.size)] * 3
    except ValueError:
        size = [int(s) for s in args.size.split("-")]

    try:
        img,res = download_cloudvolume(args.cloudpath, args.mip, position, size)
    except:
        img,res = download_webknossos(args.cloudpath, args.mip, position, size)
        
    print(res)
    data = np2pv(img, res)
    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    data.save(args.output)
