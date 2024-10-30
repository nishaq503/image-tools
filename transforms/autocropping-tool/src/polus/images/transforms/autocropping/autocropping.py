"""Autocropping Tool."""


import pathlib

import bfio

from . import bounding_box
from . import utils

logger = utils.make_logger(__name__)


def autocrop_group(  # noqa: PLR0913
    inp_files: list[pathlib.Path],
    crop_x: bool,
    crop_y: bool,
    crop_individually: bool,
    threshold: float,
    out_dir: pathlib.Path,
) -> None:
    """Crop a group of images to a bounding box.

    Args:
        inp_files: The list of input files to crop.
        crop_x: Whether to crop the images in the X dimension.
        crop_y: Whether to crop the images in the Y dimension.
        crop_individually: Whether to crop each image individually to its own
            bounding box, or to crop all images to the same bounding box.
        threshold: The threshold to use when finding spikes in the entropy
            gradients.
        out_dir: The path to save the cropped images to.

    Raises:
        ValueError: If `inp_files` is empty.
        FileNotFoundError: If any of the input files do not exist.
        FileNotFoundError: If `out_dir` does not exist.
        ValueError: If any bounding box is invalid.
    """
    if len(inp_files) == 0:
        msg = "No input files provided."
        raise ValueError(msg)

    readers = [(inp_path, bfio.BioReader(inp_path)) for inp_path in inp_files]
    box: bounding_box.BoundingBox

    if crop_individually:
        logger.info("Cropping images individually...")

        for inp_path, reader in readers:
            logger.info(f"Individually processing {inp_path.name}...")

            box = bounding_box.BoundingBox.from_image(
                reader=reader,
                rows=crop_y,
                cols=crop_x,
                threshold=threshold,
            )
            logger.info(f"Bounding box for {inp_path.name}: {box}")
            box.crop_image(
                reader=reader,
                out_path=out_dir / inp_path.name,
            )
    else:
        logger.info("Cropping images collectively...")
        boxes: list[bounding_box.BoundingBox] = []
        for inp_path, reader in readers:
            logger.info(f"Finding bounding box for {inp_path.name}...")
            box = bounding_box.BoundingBox.from_image(
                reader=reader,
                rows=crop_y,
                cols=crop_x,
                threshold=threshold,
            )
            logger.info(f"Bounding box for {inp_path.name}: {box}")
            boxes.append(box)

        box = sum(boxes[1:], boxes[0])
        logger.info(f"Bounding box for all images: {box}")

        for inp_path, reader in readers:
            logger.info(f"Collectively cropping {inp_path.name}...")
            box.crop_image(
                reader=reader,
                out_path=out_dir / inp_path.name,
            )

    for _, reader in readers:
        reader.close()

    logger.info("Autocropping complete.")
