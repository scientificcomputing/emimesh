# Mid block with FHC_CT, conforming to tetwild top/bottom

The top and bottom blocks are meshed with pytetwild, as in `split_blocks.py`. The mid
block is meshed with the constrained tetrahedralization of
[FHC_CT](https://github.com/FHCCT/FHC_CT), so that it conforms exactly to the tetwild cut
faces. The mesh quality is not optimised here; that is left to a later facet-freezing
fTetWild optimisation.

## Run

```
./run.sh [d]        # on an x86_64 compute node, d = 5 by default
```

| step | script | output |
|---|---|---|
| tetwild top/bottom, cut-plane faces with the adjacent tet marker | `tetwild_blocks.py top\|bottom` | `{top,bottom}_tw.vtu`, `{top,bottom}_cut.vtk` |
| simplify the mid surface (`simplify_surface`, eps = 0.05 dx) | `simplify_mid.py` | `mid_dec.vtk` |
| dt input: mid surface, caps, stitching tubes, side walls | `build_mid.py [d] [surface]` | `mid_in.vtk`, `mid_stitch.vtk` |
| constrained tetrahedralization | `dt --input mid_in.vtk --output mid_out.vtk --out 1` | `mid_out.vtk` |
| checks and labels | `check_mid.py` | `mid_labelled.vtu`, `mid_interfaces.vtk` |
| optional: label error against the full-res mid surface | `label_error.py` | `mid_labelled_err.vtu` |

`extern/FHC_CT` (cloned, git-ignored) contains only a prebuilt binary, `bin/linux/dt`,
and a static library. Notes on it:
- Input must be legacy VTK 2.0 ASCII POLYDATA (`vtkio.write_surface`); VTK 5.1 makes it
  hang.
- The output points are reordered, so map them back by coordinates.
- All input triangles are kept exactly, including internal and open surfaces.
- It aborts on intersecting input.
- Planar quads split into two triangles give zero-volume tets.
- `--size` has no effect, and `--refine` only adds interior points.

## Method (`build_mid.py`)

1. **Mid surface:** the simplified mid surface, with its own labelled caps, is
   compressed affinely in z into [950+d, 1050-d]. An affine map keeps it closed and
   intersection free. Clipping or projecting it did not: projection produced
   zero-volume tets, and leaving a gap gave wrong labels near the caps.
2. **Caps:** the tetwild cut faces at z ≈ 950 and 1050. They are kept exactly; they
   deviate from the plane by up to 2.1 units but are height fields.
3. **Stitching tubes:** on both caps, cells are separated by ECS everywhere; no two cells
   touch directly. For each cell, every boundary cycle on the tetwild cap is joined to the
   matching cycle of the same cell on the mid cap. The joining triangulation minimises the
   total length of the connecting edges, and only vertices on that cell's own membrane are
   used.
   - **Matching:** cycles are paired by distance (Hungarian algorithm). Loops touching at
     a pinch vertex of the simplified mid cap are merged first.
   - **Collar:** each tube starts with a vertical strip that lifts the boundary vertices
     (not on the rim) to a flat plane at z = 950 + (max cap deviation + d)/2. This keeps
     the tube from cutting through the uneven tetwild cap.
   - **Side walls:** cycles touching the box side are split into interior chains, whose
     end points become anchors of the side wall strips. One chain against several is
     joined through the rim pieces between them.
   - **Mismatches:** regions that exist on only one side end flat at their cap.
4. **Side walls:** strips between the two cap rims, zipped piecewise between the anchors.

Labels (`check_mid.py`):
- Band tets take the winding number of the compressed mid surface, which is exact
  because the surface is closed.
- The tubes close every cell region in the slab between the caps. Each connected slab
  region takes the label of the cap faces it touches.

## Result (d = 5)

- **Size:** 43k input triangles, 103k tets, 27 Steiner points; dt takes about 1 s.
- **Validity:**
  - all input triangles are tet faces;
  - 0 tets with volume ≤ 0;
  - the mesh boundary is exactly caps + side walls.
- **Labels:**
  - the caps equal the tetwild markers on every face;
  - labels change only across constraint faces, and every tube face separates its cell
    from ECS;
  - only one slab region touches two labels: a mid-only piece of label 34 (area 3.9k),
    which ends flat at the mid cap.
- **Label error:** compared with the full-resolution mid surface, 1.8% of the volume is
  labelled differently (0.9% in the interior). This comes from the compression (up to d
  in z) and the stitching. Without simplification it is about the same, but with 8× the
  tets.

## Possible alternative

Mesh every block with dt on shared caps. That is conforming by construction and needs no
stitching, but it requires identical cap triangulations in neighbouring blocks and
intersection-free block surfaces. The decimated `top_surf_dec.vtk` currently
self-intersects.
