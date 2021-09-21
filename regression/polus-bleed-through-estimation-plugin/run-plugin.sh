#!/bin/bash

version=$(<VERSION)
data_path=$(readlink --canonicalize ../../data)

# Must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
POLUS_LOG=INFO

# Change to .ome.zarr to save output images as zarr files.
POLUS_EXT=".ome.tif"

# Inputs
inpDir=/data/images/MaricRatBrain2019/standard/intensity
filePattern="S1_R1_C1-C11_A1_c00{c}.ome.tif"
groupBy="c"
kernelSize="3x3"

# Output paths
outDir=/data/output/images/
csvDir=/data/output/csvs

docker run --mount type=bind,source="${data_path}",target=/data/ \
            --user "$(id -u)":"$(id -g)" \
            --env POLUS_LOG=${POLUS_LOG} \
            --env POLUS_EXT=${POLUS_EXT} \
            polusai/bleed-through-estimation-plugin:"${version}" \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --groupBy ${groupBy} \
            --kernelSize ${kernelSize} \
            --outDir ${outDir} \
            --csvDir ${csvDir}
