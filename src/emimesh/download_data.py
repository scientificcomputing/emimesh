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


MICRONS_POSITION_RESOLUTION = (4, 4, 40)


def download_cloudvolume(
    cloud_path,
    mip,
    pos,
    physical_size,
    position_resolution=MICRONS_POSITION_RESOLUTION,
    return_origin=False,
):
    """
    Download the block of physical_size (nm) centered at pos, at mip.

    pos: center in voxels of size position_resolution (nm per axis), i.e. the
        coordinates neuroglancer shows for its `dimensions`. The default is that
        of the MICrONS explorer (4 x 4 x 40 nm; the minnie65 segmentation has
        8 x 8 x 40 nm at mip 0). The FlyEM male CNS uses (8, 8, 8).
    return_origin: also return the lower corner of the block (nm).

    Returns (img, resolution), or (img, resolution, origin) with return_origin.
    """
    from cloudvolume import Bbox, CloudVolume
    vol = CloudVolume(
        cloud_path, use_https=True, parallel=8, progress=True, mip=mip, cache=True, bounded=True
    )
    res = np.asarray(vol.resolution, dtype=np.float64)
    print(f"data resolution: {vol.resolution}")
    size = np.rint(np.array(physical_size) / res).astype(np.int64)

    # center in voxels of mip, floored as CloudVolume's point_to_mip
    center_nm = np.asarray(pos, dtype=np.float64) * np.asarray(position_resolution, np.float64)
    center = np.floor(center_nm / res).astype(np.int64)

    # download_point spans pt +- size//2 and so drops a voxel for odd sizes;
    # build the bbox ourselves to get exactly `size` voxels
    minpt = center - size // 2
    bbox = Bbox(minpt, minpt + size)
    img = vol.download(bbox, mip=mip).squeeze()
    if return_origin:
        return img, vol.resolution, minpt * res
    return img, vol.resolution
