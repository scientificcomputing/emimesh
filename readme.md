# EMI-Mesh: Generating high-quality extracellular-membrane-intracellular meshes from imaging data
![example workflow](https://github.com/scientificcomputing/emimesh/actions/workflows/test_conda.yml/badge.svg)

<img src='images/grand_challenge_mesh2.png' width='400'>

This repo provides a pipeline to generate high quality tetrahedral meshes of brain tissue on the cellular scale suitable for numerical simulations.

## Features:

- generates high quality meshes of dense reconstructions of the neuropil
- both extracellular and intracellular space included
- automated pipeline from segmentation to mesh
- image processing steps included, to account for e.g. missing ECS from chemically fixated tissue

## Getting started

First, install snakemake (using conda):

`conda create -c conda-forge -c bioconda -n snakemake snakemake snakemake-storage-plugin-http snakemake-executor-plugin-cluster-generic`

and activate it:
`conda activate snakemake`

Then, run snakemake on an example configuration file, with e.g.

`snakemake --configfile configfiles/cortical_mm3.yml --use-conda --cores 8`


## Intended Workflow
Emimesh works with `.yaml` configuration files (see `config_files/` for examples). In one file, you specify:

- the base dataset, position and size, using [webknossos](https://home.webknossos.org/publications) or [neuroglancer](https://github.com/google/neuroglancer/) (example: cortical MM^3 dataset at position [225182-107314-22000](https://ngl.microns-explorer.org/#!%7B%22dimensions%22:%7B%22x%22:%5B4e-9%2C%22m%22%5D%2C%22y%22:%5B4e-9%2C%22m%22%5D%2C%22z%22:%5B4e-8%2C%22m%22%5D%7D%2C%22position%22:%5B225182.5%2C107314.5%2C22000.5%5D%2C%22crossSectionScale%22:11.406101410482504%2C%22projectionOrientation%22:%5B0.1528419554233551%2C0.49656152725219727%2C0.39075320959091187%2C0.7598538994789124%5D%2C%22projectionScale%22:40961.499900183306%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%7B%22url%22:%22precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em%22%2C%22subsources%22:%7B%22default%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22annotationColor%22:%22#7d7d7d%22%2C%22shaderControls%22:%7B%22normalized%22:%7B%22range%22:%5B86%2C172%5D%7D%7D%2C%22name%22:%22img%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%7B%22url%22:%22precomputed://gs://iarpa_microns/minnie/minnie65/seg%22%2C%22subsources%22:%7B%22default%22:true%2C%22mesh%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22segments%22%2C%22annotationColor%22:%22#949494%22%2C%22selectedAlpha%22:0.3%2C%22segments%22:%5B%22864691134947427836%22%2C%22864691135337771494%22%2C%22864691135393949941%22%2C%22864691135462270365%22%2C%22864691135474669888%22%2C%22864691135617729935%22%2C%22864691135718476593%22%2C%22864691136024102713%22%2C%22864691136390364287%22%2C%22864691136436690846%22%5D%2C%22segmentQuery%22:%22864691136194301772%2C%20864691136814938734%22%2C%22colorSeed%22:3728349837%2C%22name%22:%22seg%22%7D%5D%2C%22showAxisLines%22:false%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%22seg%22%7D%2C%22layout%22:%7B%22type%22:%224panel%22%2C%22orthographicProjection%22:true%7D%2C%22selection%22:%7B%22layers%22:%7B%22seg%22:%7B%22annotationId%22:%22data-bounds%22%2C%22annotationSource%22:0%2C%22annotationSubsource%22:%22bounds%22%7D%7D%7D%7D))
- segmentation processing steps and meshing parameters

Then, emimesh will do the following steps for you:
- download segmented image data
- preprocess the image for meshing
- extract the surfaces of each cell
- generate a volumetric mesh of the extracted surfaces mesh and the extracellular space in between the cells with [fTetWild](https://github.com/wildmeshing/fTetWild)

A simple configuration file could look like this:
```yaml
name : cortical_mm3 # will only be used for naming the output folder
raw:
  cloudpath : "precomputed://gs://iarpa_microns/minnie/minnie65/seg" # cloud volume datapath
  mip : 2 # resolution level (usually 0 corresponds to finest, and higher mips provides upsampled data)
  position : "225182-107314-22000" # center of the dataset
  size : 5000  # side length of the dataset (in physical dimension, usually nm)
processing:
  dx : 20        # isotropic voxels size for resampling before processing
  operation : 
    - "smooth iterations=1 radius=1" # smooth the segmentation before meshing
meshing:
  envelopsize : [18] # maximum deviation of the tetrahdral mesh interfaces from the input surfaces (in physical dimension, usually nm).
```

## Configuration Reference

The configuration file is divided into `raw`, `processing`, and `meshing`.

### Raw Data (`raw`)
Controls data download and extent.

| Option | Description |
| :--- | :--- |
| **`cloudpath`** | URI to the data source. Supports `precomputed://` (cloud-volume) or webknossos URLs. |
| **`position`** | Center coordinates of the volume in `x-y-z` format (integer pixels). |
| **`mip`** | Resolution level. `0` is the highest resolution. Higher integers indicate downsampled versions. |
| **`size`** | Dimensions of the volume to download in physical units (nm). Can be a single integer `[5000]` for a cube, or `"dx-dy-dz"` for a bounding box (where `dx`,`dy` and `dz` are integer values). |

### Processing Settings (`processing`)
Controls general resampling and filtering.

| Option | Description |
| :--- | :--- |
| **`dx`** | Target isotropic resolution (nm). The downloaded data will be resampled to this resolution before operations are applied. |
| **`ncells`** | (Optional) Integer. If specified, only the largest `N` cells by volume are kept in the final mesh. |

### Meshing Settings (`meshing`)
Controls the `fTetWild` meshing engine.

| Option | Description |
| :--- | :--- |
| **`envelopsize`** | The "envelope" size (epsilon) for fTetWild, in nm. This defines how much the final tetrahedral mesh surface is allowed to deviate from the input surface. Larger values allow for coarser meshes with fewer elements. |
| **`stopquality`** | (Optional) fTetWild quality score (default: 10). Controls the trade-off between mesh quality and fidelity. Higher values stop optimization earlier (at the expense of mesh quality). |

## Image Processing Operations

A key part of the EMI-Mesh workflow is the `processing` block in your config file. You can define a sequential list of operations to clean, smooth, and manipulate the segmentation data before meshing. The operations are based on [nbmorph](https://github.com/MariusCausemann/nbmorph) and are multithreaded.

Operations are defined as a list of strings under `processing: operation`. Each string follows the format:
`"command arg1=value1 arg2=value2 ..."`

### Available Operations

| Operation | Arguments | Description |
| :--- | :--- | :--- |
| **`merge`** | `labels` (list) | Merges all provided labels into the first label in the list. Useful for combining fragments of the same cell. |
| **`ncells`** | `ncells` (int)<br>`keep_cell_labels` (list, optional) | Filters the image to keep only the *N* largest cells (by volume). Use `keep_cell_labels` to ensure specific IDs are not filtered out. |
| **`dilate`** | `radius` (int)<br>`labels` (list, optional) | Expands labels by the specified radius (in pixels). If `labels` is provided, only those specific cells are dilated. |
| **`erode`** | `radius` (int)<br>`labels` (list, optional) | Shrinks labels by the specified radius. If `labels` is provided, only those specific cells are eroded. |
| **`smooth`** | `radius` (int)<br>`iterations` (int)<br>`labels` (list, optional) | smooths the cell boundaries using morphological opening and closing. |
| **`removeislands`** | `minsize` (int) | Removes disconnected components (dust) smaller than the specified voxel count. |

### Region of Interest (ROI) Operations
You can generate a mask (ROI) based on specific cells and apply operations to that mask before applying it back to the image. This is useful for masking out areas surrounding specific cells.

| Operation | Arguments | Description |
| :--- | :--- | :--- |
| **`roigenerate`** | `labels` (list) | Creates a binary ROI mask containing the specified labels. |
| **`roiapply`** | *None* | Applies the current ROI mask to the image (sets everything outside the ROI to 0/ECS). |
| **`roi<op>`** | *Same as op* | Applies a standard operation to the *ROI mask* itself (e.g., `roidilate radius=10`, `roierode`). |

### Examples

**1. Basic Smoothing and Cleaning**
Standard cleanup for a cortical volume.
```yaml
processing:
  operation:
    - "removeislands minsize=5000"    # Remove small noise
    - "dilate radius=1"               # Close small gaps
    - "smooth iterations=1 radius=1"  # Smooth boundaries
    - "erode radius=1"                # Create gaps between cells
```

**2. Targeting a Specific Cell (Astrocyte)**
Filter for 1 cell, but ensure a specific ID is kept, then smooth.
```yaml
processing:
  operation:
    - "ncells ncells=1 keep_cell_labels='[864691136194301772]'"
    - "smooth iterations=1 radius=2"
    - "erode radius=1"
```

**3. Complex ROI Manipulation (Capillary)**
Merge fragments, clean noise, then create a large masked area around a specific capillary cell to exclude distant neighbors.
```yaml
processing:
  operation:
    # Merge fragmented labels into one ID
    - "merge labels='[864691136534887842, 864691135476973891]'"
    # Clean up
    - "removeislands minsize=5000"
    - "dilate radius=5"
    - "smooth iterations=2 radius=4"
    - "erode radius=1"
    # Create a mask based on the main capillary label
    - "roigenerate labels='[864691136534887842]'"
    # Expand the mask significantly
    - "roidilate radius=80"
    # Apply mask (delete everything outside the expanded capillary area)
    - "roiapply"
```

## Output
The output consists of the following directories:
* raw: The downloaded segmentation as in `.vti` format, suitable for e.g. paraview
* processed: The processed image in `.vti` format
* surfaces: The surfaces of the extracted cells in `.ply` format, again suitable for visualization with paraview or usage in other meshing software
* meshes: The generated volumetric meshes in `.xdmf` format, containing labels for the extracellular space (label 1) and increasing integer values (2,..., N) for all cells. A mapping between the labels and the original cell id in the base segmenation is provided in the `processed/.../imagestatistics.yml`file.