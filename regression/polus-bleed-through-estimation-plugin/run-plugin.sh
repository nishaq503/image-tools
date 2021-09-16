#!/bin/bash

version=$(<VERSION)
data_path=$(readlink --canonicalize ../../data/bleed_through_estimation)

# Must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
POLUS_LOG=INFO

# Change to .ome.zarr to save output images as zarr files.
POLUS_EXT=".ome.tif"

# Inputs
inpDir=/data/input
filePattern="r{rrr}_c{ccc}_z{zzz}.ome.tif"
groupBy="c"

# Output paths
outDir=/data/output/images
csvDir=/data/output/csvs

docker run --mount type=bind,source="${data_path}",target=/data/ \
            --user "$(id -u)":"$(id -g)" \
            --env POLUS_LOG=${POLUS_LOG} \
            --env POLUS_EXT=${POLUS_EXT} \
            polusai/bleed-through-estimation-plugin:"${version}" \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --groupBy ${groupBy} \
            --outDir ${outDir} \
            --csvDir ${csvDir}
