import numpy as np

__all__ = ["download_webknossos", "download_cloudvolume"]

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
    from cloudvolume import Bbox, CloudVolume
    vol = CloudVolume(
        cloud_path, use_https=True, parallel=8, progress=True, mip=mip, cache=True, bounded=True
    )
    print(f"data resolution: {vol.resolution}")
    size = np.rint(np.array(physical_size) / vol.resolution).astype(np.int64)

    pos = np.array(pos, dtype=np.float32)
    pos[:2] /= 2  # account for different resolution online

    # download_point spans pt +- size//2 and so drops a voxel for odd sizes;
    # build the bbox ourselves to get exactly `size` voxels
    center = np.array(vol.point_to_mip(pos, mip=0, to_mip=mip)).astype(np.int64)
    minpt = center - size // 2
    bbox = Bbox(minpt, minpt + size)
    img = vol.download(bbox, mip=mip).squeeze()
    return img, vol.resolution
