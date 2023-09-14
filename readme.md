# EMI-Mesh: Generating high-quality extracellular-membrane-intracellular meshes from imaging data


<img src='images/grand_challenge_mesh2.png' width='400'>

This repo provides a pipeline to generate high quality tetrahedral meshes of brain tissue on the cellular scale suitable for numerical simulations.

## Features:

- high quality meshes of dense reconstructions of the neuropil
- includes both extracellular and intracellular space
- automated pipeline from segmentation to mesh
- basic image processing steps included, to account for e.g. missing ECS from chemically fixated tissue


## Workflow
- download segmented image data
- preprocess the image for meshing
    - choose *N* largest cells
    - expand, apply morphological smoothing and shrinkage/erosion to each cell
- extract the surfaces of each cell
- generate a volumetric mesh of the extracted surfaces mesh and the extracellular space in between the cells with [fTetWild](https://github.com/wildmeshing/fTetWild)

## Reproducing results

The workflow is based on [Snakemake](https://snakemake.readthedocs.io/en/stable/index.html). To generate meshes, install snakemake (e.g. `conda install -c bioconda snakemake`), modify the `config.yml` file in this repo and run `snakemake --cores all --use-conda`. That's it!
`Snakemake` will install all required dependencies (specified in `workflow/envs/environment.yml`) and orchastrate the jobs. It also supports schedulers on HPC systems such as slurm.

## Output
The output consists of the following directories:
* raw: The downloaded segmentation as is in `.vtk` format, suitable for e.g. paraview
* processed: The processed image in `.vtk` format
* surfaces: The surfaces of the extracted cells in `.stl` format, again suitable for visualization with paraview or usage in other meshing software
* meshes: The generated volumetric meshes in `.xdmf` format, containing labels for the extracellular space (label 1) and increasing integer vaules (2,..., N) for all cells contained. There is currently no mapping to the cell ids of the segmentation. The file `_facet.xdmf`contains facet marker, where the label *l* corresponds to the boundary between ECS and cell *l*. The outer boundaries are marked as `l + offset`, where `offset` is the next higher power of ten of the cell numbers (`offset=int(10 ** np.ceil(np.log10(N_cells)))`)


## Limitations
* currently only supports data accessible via [cloud-volume](https://github.com/seung-lab/cloud-volume)
* assumes isotropic data
* does not handle intersecting cells
* agnostic to cell types - all cells are handled equal


