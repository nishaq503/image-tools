#!/bin/bash

version=$(<VERSION)
echo "Version: ${version}"

data_path=$(readlink -f ./data)
echo "Data path: ${data_path}"

docker run polusai/stitching-vector-to-microjson-plugin:"${version}"

# Inputs
inpDir=/data/input
filePattern="*.txt"

# Output paths
outDir=/data/output

# Must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
POLUS_LOG=INFO

docker run --mount type=bind,source="${data_path}",target=/data/ \
            --user "$(id -u)":"$(id -g)" \
            --env POLUS_LOG=${POLUS_LOG} \
            --env POLUS_IMG_EXT=${POLUS_IMG_EXT} \
            polusai/stitching-vector-to-microjson-plugin:"${version}" \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --outDir ${outDir}
