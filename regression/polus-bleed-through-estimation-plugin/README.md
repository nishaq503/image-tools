# Bleed-Through Estimation

This WIPP plugin estimates the bleed-through in a collection of 2d images.
The process is described in detail in [this paper](https://doi.org/10.1038/s41467-021-21735-x) and is implemented in [this repo](https://github.com/RoysamLab/whole_brain_analysis).

The `filePattern` and `groupBy` parameters can be used to group images into subsets.
This plugin will apply bleed-through correction to each subset.

TODO: We have noticed a memory leak that occurs when too many models are being trained.
This is probably from somewhere in the compiled and optimized code for sklearn.
We are investigating the source but for now, images with dozens of channels might not be a good idea.
During testing, this plugin was stable on 5 rounds of 11 images each for the MaricRatBrain dataset.

## Sample Images

TODO: Add sample images of inputs and outputs.

## File Patterns

This plugin uses [file-patterns](https://filepattern.readthedocs.io/en/latest/Examples.html#what-is-filepattern) to create subsets of an input collection.
In particular, defining a filename variable is surrounded by `{}`, and the variable name and number of spaces dedicated to the variable are denoted by repeated characters for the variable.
For example, if all filenames follow the structure `prefix_tTTT.ome.tif`, where `TTT` indicates the time-point of capture of the image, then the file-pattern would be `prefix_t{ttt}.ome.tif`.

The available variables for filename patterns are `x`, `y`, `p`, `z`, `c` (channel), `t` (time-point), and `r` (replicate).
For position variables, either `x` and `y` grid positions or a sequential position `p` may be present, but not both.
This differs from MIST in that `r` and `c` are used to indicate grid row and column rather than `y` and `x` respectively.
This change was made to remain consistent with Bioformats dimension names and to permit the use of `c` as a channel variable.

## Other Parameters

### --groupBy

Each round of staining should be its own group.
This suggests that `'r'` and `'c'` should be among the grouping variables.
You can also use `'x'`, `'y'`, `'z'` and/or `'p'` to further subgroup tiles if the fields-of-view have not yet been stitched together.

TODO: More testing around proper grouping of images by positional variables...

### --tileSelectionCriterion

Which method to use to rank and select tiles in images.
The available options are:

* `'HighMeanIntensity'`: Select tiles with the highest mean pixel intensity.
                         This is the default.
* `'HighEntropy'`: Select tiles with the highest entropy.
* `'HighMedianIntensity'`: Select tiles with the highest median pixel intensity.
* `'LargeIntensityRange'`: Select tiles with the largest difference in intensity of the brightest and dimmest pixels.

We rank-order all tiles based on one of these criteria and then select the 10 best tiles from each channel.
A tiles chosen for any channel is used for all channels.

TODO: Add more...

### --model

Which model (from `sklearn.linear_model`) to train for estimating bleed-through.
The available options are:

* `'Lasso'`: This was used for the paper linked above and is the default.
* `'ElasticNet'`: A generalization of `'Lasso'`.
* `'PoissonGLM'`: A `GLM` with the `PoissonRegressor` because pixel intensities follow a Poisson distribution.

For each channel, we train a separate instance of this model (on the selected tiles) using adjacent channels to estimate bleed-through.

TODO: Add more...

TODO: ICA is probably the correct model for this task.
However, I know of no way to train ICA with batches of data (with each set of tiles making a single batch).
We could probably run ICA if we had a small enough number of tiles but the memory-scaling of ICA generally makes it unsuitable for big-data applications. 

### --channelOverlap

The number of adjacent channels to consider for estimating bleed-through.

We assume that the channels in the input images are numbered `0, 1, 2, ... num_channels` and that the channels are ordered by their target wavelength.
We use a default value of `1`.
Higher values will look for bleed-through from farther channels but will cause the models to take longer to train.

### --computeComponents

Whether to compute and write the bleed-through component for each image.
These components can be directly subtracted from the source images to correct for bleed-through.

TODO: Some mixing coefficients from the models are large and could produce negative pixel-values after subtraction. Investigate and fix...

## Build the plugin

To build the Docker image for the conversion plugin, run `./build-docker.sh`.

## Install WIPP Plugin

In WIPP, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 3 input arguments and 1 output argument:

| Name            | Description                                   | I/O    | Type    | Default |
|-----------------|-----------------------------------------------|--------|---------|---------|
| `--inpDir`      | Path to input images.                         | Input  | String  | N/A |
| `--filePattern` | File pattern to subset images.                | Input  | String  | N/A |
| `--groupBy`     | Variables to group together.                  | Input  | String  | N/A |
| `--tileSelectionCriterion`  | Method to use for selecting tiles. | Input  | Enum  | HighMeanIntensity |
| `--model`  | Model to train for estimating bleed-through. | Input  | Enum  | Lasso |
| `--channelOverlap`  | Number of adjacent channels to consider. | Input  | Integer  | 1 |
| `--computeComponents`  | Whether to compute and write the bleed-through component for each image. | Input  | Boolean  | true |
| `--outDir`      | Output image collection,                      | Output | String  | N/A |
