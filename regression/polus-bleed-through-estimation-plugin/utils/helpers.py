from typing import Generator
from typing import Union

from bfio import BioReader
from bfio import BioWriter

from . import constants


def replace_extension(name: str, new_extension: str = None) -> str:
    """ Replaces the extension in the name of an input image with `POLUS_EXT`
     for writing corresponding output images. """
    new_extension = constants.POLUS_EXT if new_extension is None else new_extension
    return (
        name
        .replace('.ome.tif', new_extension)
        .replace('.ome.zarr', new_extension)
    )


def count_tiles(reader_or_writer: Union[BioReader, BioWriter]) -> int:
    """ Returns the number of tiles in a BioReader/BioWriter """
    tile_size = constants.TILE_SIZE_2D if reader_or_writer.Z == 1 else constants.TILE_SIZE_3D
    num_tiles = (
        len(range(0, reader_or_writer.Z, tile_size)) *
        len(range(0, reader_or_writer.Y, tile_size)) *
        len(range(0, reader_or_writer.X, tile_size))
    )
    return num_tiles


def tile_indices(
        reader_or_writer: Union[BioReader, BioWriter],
) -> Generator[tuple[int, int, int, int, int, int], None, None]:
    """ A generator for the indices of all tiles in a BioReader/BioWriter. """
    tile_size = constants.TILE_SIZE_2D if reader_or_writer.Z == 1 else constants.TILE_SIZE_3D

    for z_min in range(0, reader_or_writer.Z, tile_size):
        z_max = min(reader_or_writer.Z, z_min + tile_size)

        for y_min in range(0, reader_or_writer.Y, tile_size):
            y_max = min(reader_or_writer.Y, y_min + tile_size)

            for x_min in range(0, reader_or_writer.X, tile_size):
                x_max = min(reader_or_writer.X, x_min + tile_size)

                yield z_min, z_max, y_min, y_max, x_min, x_max


def count_tiles_2d(reader_or_writer: Union[BioReader, BioWriter]) -> int:
    """ Returns the number of 2d tiles in a BioReader/BioWriter """
    tile_size = constants.TILE_SIZE_2D if reader_or_writer.Z == 1 else constants.TILE_SIZE_3D
    num_tiles = (
        reader_or_writer.Z *
        len(range(0, reader_or_writer.Y, tile_size)) *
        len(range(0, reader_or_writer.X, tile_size))
    )
    return num_tiles


def tile_indices_2d(
        reader_or_writer: Union[BioReader, BioWriter],
) -> Generator[tuple[int, int, int, int, int], None, None]:
    """ A generator for the indices of all 2d tiles in a BioReader/BioWriter. """
    tile_size = constants.TILE_SIZE_2D if reader_or_writer.Z == 1 else constants.TILE_SIZE_3D

    for z in range(reader_or_writer.Z):

        for y_min in range(0, reader_or_writer.Y, tile_size):
            y_max = min(reader_or_writer.Y, y_min + tile_size)

            for x_min in range(0, reader_or_writer.X, tile_size):
                x_max = min(reader_or_writer.X, x_min + tile_size)

                yield z, y_min, y_max, x_min, x_max
