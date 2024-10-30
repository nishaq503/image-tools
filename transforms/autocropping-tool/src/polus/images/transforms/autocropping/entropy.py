"""Helpers for entropy calculations."""

import numpy


def row_wise_entropy(tile: numpy.ndarray) -> numpy.ndarray:
    """Calculate the entropy of the rows in a tile.

    Args:
        tile: A numpy array (at least 2D) representing a tile.

    Returns:
        A 1D numpy array representing the entropy of the rows.
    """
    dtype = tile.dtype

    if numpy.issubdtype(dtype, numpy.floating):
        return row_entropy_float(tile)

    if numpy.issubdtype(dtype, numpy.integer):
        return row_entropy_int(tile)

    if numpy.issubdtype(dtype, numpy.bool_):
        tile = tile.astype(numpy.uint8)
        return row_entropy_int(tile)

    msg = f"Unsupported dtype: {dtype}"
    raise ValueError(msg)


def col_wise_entropy(tile: numpy.ndarray) -> numpy.ndarray:
    """Calculate the entropy of the columns in a tile.

    Args:
        tile: A numpy array (at least 2D) representing a tile.

    Returns:
        A 1D numpy array representing the entropy of the columns.
    """
    dtype = tile.dtype
    tile = tile.T

    if numpy.issubdtype(dtype, numpy.floating):
        return row_entropy_float(tile)

    if numpy.issubdtype(dtype, numpy.integer):
        return row_entropy_int(tile)

    if numpy.issubdtype(dtype, numpy.bool_):
        tile = tile.astype(numpy.uint8)
        return row_entropy_int(tile)

    msg = f"Unsupported dtype: {dtype}"
    raise ValueError(msg)


def row_entropy_float(tile: numpy.ndarray) -> numpy.ndarray:
    """Calculate the entropy of the rows in a tile.

    Args:
        tile: A numpy array (at least 2D) with float values representing a tile.

    Returns:
        A numpy array representing the entropy of the rows.
    """
    # Normalize the rows so that they sum to 1
    tile = tile / tile.sum(axis=1, keepdims=True)

    # Calculate the entropy of the rows
    return -numpy.sum(tile * numpy.log2(tile), axis=1)


def row_entropy_int(tile: numpy.ndarray) -> numpy.ndarray:
    """Calculate the entropy of the rows in a tile.

    Args:
        tile: A numpy array (at least 2D) with integer values representing a tile.

    Returns:
        A numpy array representing the entropy of the rows.
    """
    # Use one more bin than the maximum value in the tile
    nbins = numpy.max(tile) + 1

    # Get counts of each value in the rows
    counts = numpy.vstack(numpy.bincount(row, minlength=nbins) for row in tile)

    # Divide by the number of columns to get the probability of each value
    probs = counts.astype(numpy.float32) / float(tile.shape[1])

    # Calculate the entropy of the rows
    return -numpy.sum(probs * numpy.log2(probs), axis=1)


__all__ = ["row_wise_entropy", "col_wise_entropy"]
