import pyclesperanto_prototype as cle
from cloudvolume import CloudVolume
import numpy as np
import fastremap
cle.set_wait_for_kernel_finish(True)
import dask.array as da
import matplotlib.pyplot as plt
from functools import partial
import time

cloud_path = "precomputed://gs://iarpa_microns/minnie/minnie65/seg"
pos = (337400,138900,22519)
physical_size  = [2000,]*3
mip = 1
vol = CloudVolume(
    cloud_path, parallel=8, progress=True, mip=mip, cache=True, bounded=True
)

size = [ps / res for ps, res in zip(physical_size, vol.resolution)]
size = np.array(size).astype("uint64")

pos = np.array(pos, dtype=np.float32)
pos[:2] /= 2  # account for different resolution online

img = vol.download_point(pos, mip=mip, size=size).squeeze()
#img, remapping = fastremap.renumber(img)
img = da.from_array(img, chunks=1024)#, chunks=(500,500,500))
print("start unique")
cell_labels, cell_counts = fastremap.unique(img, return_counts=True)
print("end unique")
remapping = {c:i for i,c in enumerate(cell_labels)}
img = img.map_blocks(partial(fastremap.remap, table=remapping), dtype=img.dtype)
img = img.map_blocks(partial(fastremap.refit, value=len(cell_labels)))

cle.select_device(cle.available_device_names()[0])
print(f"Running processing on {cle.get_device()}")

def process_img(chunk, smoothr, smoothit):
    print("Processing image of size", chunk.shape)
    for i in range(2):
        chunk = cle.dilate_labels(chunk, radius=2)

    for i in range(smoothit):
        chunk = cle.opening_labels(chunk, radius=smoothr)
        chunk = cle.closing_labels(chunk, radius=smoothr)
        #chunk = cle.smooth_labels(chunk, radius=smoothr)

    for i in range(1):
        chunk = cle.erode_labels(chunk, radius=2)
    return np.array(chunk)

img = da.random.randint(low=0, high=1000, size=[1500]*3, chunks=500)
unsm = da.map_blocks(partial(process_img, smoothr=2, smoothit=2),
                     img, dtype=img.dtype)
mem_gb = 16       
chunk_mem_gb = np.prod(img.chunksize) * np.nbytes[img.dtype] / 1e9
max_workers = int(mem_gb / (chunk_mem_gb*2.5))
print(f"shape : {img.shape}")
print(f"chunksize : {chunk_mem_gb} GB")
print(f"nchunks : {img.npartitions}")
print(f"workers : {max_workers}")
unsm = unsm.compute(num_workers=max_workers)
print("computed successfully")
exit()

smooth1 = da.map_blocks(partial(process_img, smoothr=5, smoothit=5),
                     da.rechunk(img, chunks=-1), dtype=img.dtype)

smooth2 = da.map_blocks(partial(process_img, smoothr=10, smoothit=3),
                     da.rechunk(img, chunks=-1), dtype=img.dtype)

fig, ax = plt.subplots(ncols=3)
ax[0].imshow(unsm[:,:,3])
ax[1].imshow(smooth1[:,:,3])
ax[2].imshow(smooth2[:,:,3])
plt.savefig("smoothing.png")