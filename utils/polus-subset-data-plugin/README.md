# Subset Data

This WIPP uses a filename pattern to select a subset of data from an image collection and stores the subset as a new image collection. The filename pattern can be a [regular expression](https://en.wikipedia.org/wiki/Regular_expression) and can include elements of a filename pattern described below.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Input Filename Pattern

This plugin uses [filename patterns](https://github.com/USNISTGOV/MIST/wiki/User-Guide#input-parameters) similar to that of what MIST uses. In particular, defining a filename variable is surrounded by `{}`, and the variable name and number of spaces dedicated to the variable are denoted by repeated characters for the variable. For example, if all filenames follow the structure `filename_TTT.ome.tif`, where TTT indicates the timepoint the image was captured at, then the filename pattern would be `filename_{ttt}.ome.tif`.

In addition to the position variables (both `x` and `y`, or `p`), the only other variables that can be used are `z`, `c`, and `t`.

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name            | Description                                           | I/O    | Type       |
| --------------- | ----------------------------------------------------- | ------ | ---------- |
| `--filePattern` | Filename pattern used to separate data                | Input  | string     |
| `--inpDir`      | Input image collection to be processed by this plugin | Input  | collection |
| `--outDir`      | Output collection                                     | Output | collection |

