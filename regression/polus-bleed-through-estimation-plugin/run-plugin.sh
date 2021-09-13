#!/bin/bash

version=$(<VERSION)
data_path=$(readlink --canonicalize ../../data)

# Must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
POLUS_LOG=INFO

# Change to .ome.zarr to save output images as zarr files.
POLUS_EXT=".ome.tif"

# Inputs
inpDir=/data/input
filePattern="S1_R1_C1-C11_A1_y0(00-14)_x0(00-21)_c0{cc}.ome.tif"
groupBy="c"
kernelSize=3

# Output paths
outDir=/data/output

echo $[data_path]

docker run --mount type=bind,source=${data_path},target=/data/ \
            --user "$(id -u)":"$(id -g)" \
            --env POLUS_LOG=${POLUS_LOG} \
            --env POLUS_EXT=${POLUS_EXT} \
            polusai/bleed-through-estimation-plugin:"${version}" \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --kernelSize ${kernelSize} \
            --groupBy ${groupBy} \
            --outDir ${outDir}
