#!/bin/bash

version=$(<VERSION)
data_path=$(readlink --canonicalize ../../../data/smp-training)

# Inputs
#pretrainedModel=/data/pretrained-model
modelName="Linknet"
encoderBase="ResNet"
encoderVariant="resnet34"
encoderWeights="imagenet"
optimizerName="Adam"
batchSize=8

imagesTrainDir=/data/input/train/intensity
imagesTrainPattern="p0_y1_r{r}_c0.ome.tif"
labelsTrainDir=/data/input/train/labels
labelsTrainPattern="p0_y1_r{r}_c0.ome.tif"

imagesValidDir=/data/input/val/intensity
imagesValidPattern="p0_y1_r{r}_c0.ome.tif"
labelsValidDir=/data/input/val/labels
labelsValidPattern="p0_y1_r{r}_c0.ome.tif"

device='cuda'
checkpointFrequency=5

lossName="MCCLoss"
#lossName="DiceLoss"
#lossName="SoftBCEWithLogitsLoss"
maxEpochs=5
patience=2
minDelta=1e-4

# Output paths
outputDir=/data/output

#            --pretrainedModel ${pretrainedModel} \

# Remove the --gpus all to test on CPU
docker run --mount type=bind,source="${data_path}",target=/data \
            --user "$(id -u)":"$(id -g)" \
            --rm \
            --gpus "all" \
            --privileged -v /dev:/dev \
            labshare/polus-smp-training-plugin:"${version}" \
            --modelName ${modelName} \
            --encoderBase ${encoderBase} \
            --encoderVariant ${encoderVariant} \
            --encoderWeights ${encoderWeights} \
            --optimizerName ${optimizerName} \
            --batchSize ${batchSize} \
            --imagesTrainDir ${imagesTrainDir} \
            --imagesTrainPattern ${imagesTrainPattern} \
            --labelsTrainDir ${labelsTrainDir} \
            --labelsTrainPattern ${labelsTrainPattern} \
            --imagesValidDir ${imagesValidDir} \
            --imagesValidPattern ${imagesValidPattern} \
            --labelsValidDir ${labelsValidDir} \
            --labelsValidPattern ${labelsValidPattern} \
            --device ${device} \
            --checkpointFrequency ${checkpointFrequency} \
            --lossName ${lossName} \
            --maxEpochs ${maxEpochs} \
            --patience ${patience} \
            --minDelta ${minDelta} \
            --outputDir ${outputDir}
