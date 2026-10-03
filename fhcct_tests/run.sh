#!/bin/bash
# Mid block meshed with FHC_CT dt, conforming to the tetwild top/bottom blocks.
# Run on a compute node (x86_64) from this directory.
set -e
P=${PYTHON:-/global/D1/homes/mariusca/emimesh/.pixi/envs/default/bin/python}
DT=../extern/FHC_CT/bin/linux/dt
D=${1:-5}
[ -f top_cut.vtk ] || $P tetwild_blocks.py top
[ -f bottom_cut.vtk ] || $P tetwild_blocks.py bottom
[ -f mid_dec.vtk ] || $P simplify_mid.py
$P build_mid.py $D mid_dec.vtk
$DT --input mid_in.vtk --output mid_out.vtk --out 1 > mid_dt.log 2>&1
$P check_mid.py
