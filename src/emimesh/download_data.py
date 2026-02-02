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
    from cloudvolume import CloudVolume
    vol = CloudVolume(
        cloud_path, use_https=True, parallel=8, progress=True, mip=mip, cache=True, bounded=True
    )
    print(f"data resolution: {vol.resolution}")
    size = [ps / res for ps, res in zip(physical_size, vol.resolution)]
    size = np.array(size).astype("uint64")

    pos = np.array(pos, dtype=np.float32)
    pos[:2] /= 2  # account for different resolution online

    img = vol.download_point(pos, mip=mip, size=size).squeeze()
    return img, vol.resolution
