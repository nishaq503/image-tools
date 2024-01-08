# Stitching Vector to MicroJson (v0.1.0)

Here we provide the stitching-vector-to-microjson plugin to convert stitching vectors from the text format to the microjson format.

This plugin is part of the [WIPP](https://isg.nist.gov/deepzoomweb/software/wipp) ecosystem.

Contact [Najib Ishaq](mailto:najib.ishaq@nih.gov) for more information.

## Usage

This plugin takes three parameters:

1. `inpDir`: Input stitching-vector collection.
2. `filePattern`: File-name pattern to use when selecting files to process.
3. `outDir`: Output stitching-vector collection.

## Building

Run `./build-docker.sh` to build the docker image.

## Install WIPP Plugins

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

The vector-to-label plugin takes two input parameters and one output parameter:

| Name            | Description                                              | I/O    | Type            | Default |
| --------------- | -------------------------------------------------------- | ------ | --------------- | ------- |
| `--inpDir`      | Input stitching-vector collection                        | Input  | stitchingVector | N/A     |
| `--filePattern` | File-name pattern to use when selecting files to process | Input  | string          | "*.txt" |
| `--outDir`      | Output stitching-vector collection                       | Output | genericData     | N/A     |
