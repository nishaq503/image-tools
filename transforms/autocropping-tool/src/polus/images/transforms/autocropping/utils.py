"""Miscellaneous utility functions for the tool."""


import logging
import os
import typing

import bfio

POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))

POLUS_IMG_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.tif")
POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".csv")

TILE_SIZE = 2_048


def make_logger(name: str, level: typing.Optional[str] = None) -> logging.Logger:
    """Create a logger with the given name.

    Args:
        name: The name of the logger.
        level: The logging level.

    Returns:
        The logger.
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(getattr(logging, level))
    else:
        logger.setLevel(POLUS_LOG)
    return logger


def iter_tiles_bfio(
    bfio_obj: typing.Union[bfio.BioReader, bfio.BioWriter],
    tile_size: int = TILE_SIZE,
) -> typing.Generator[tuple[int, int, int, int], None, None]:
    """Iterate over 2D tiles in an image.

    Args:
        bfio_obj: The BioReader or BioWriter over the image.
        tile_size: The size of the tiles.

    Yields:
        The x and y coordinates of the top-left corner and the bottom-right corner
        of each tile.
    """
    for y_min in range(0, bfio_obj.Y, tile_size):
        y_max = min(y_min + tile_size, bfio_obj.Y)
        for x_min in range(0, bfio_obj.X, tile_size):
            x_max = min(x_min + tile_size, bfio_obj.X)
            yield x_min, y_min, x_max, y_max


def iter_tiles_bbox(
    bbox: tuple[int, int, int, int],
    tile_size: int = TILE_SIZE,
) -> typing.Generator[tuple[int, int, int, int], None, None]:
    """Iterate over 2D tiles in a bounding box.

    Args:
        bbox: The bounding box to iterate over.
        tile_size: The size of the tiles.

    Yields:
        The x and y coordinates of the top-left corner and the bottom-right corner
        of each tile.
    """
    x_min, y_min, x_max, y_max = bbox
    for y in range(y_min, y_max, tile_size):
        y_end = min(y + tile_size, y_max)
        for x in range(x_min, x_max, tile_size):
            x_end = min(x + tile_size, x_max)
            yield x, y, x_end, y_end
