"""CLI for the Autocropping Tool tool."""

import logging
import os
import pathlib

import typer
from polus.images.transforms.autocropping.autocropping import autocropping

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("polus.images.transforms.autocropping")
logger.setLevel(POLUS_LOG)

POLUS_IMG_EXT = os.environ.get("POLUS_IMG_EXT", ".ome.tif")
POLUS_TAB_EXT = os.environ.get("POLUS_TAB_EXT", ".csv")
app = typer.Typer()


@app.command()
def main(  # noqa: PLR0913
    crop_individually: bool = typer.Option(
        False,
        help="Whether to crop each image individually to its own bounding box, or to crop all images to the same bounding box.",  # noqa: E501
        default=False,
    ),
    crop_x: bool = typer.Option(
        False,
        help="Whether to crop the images in the X dimension.",
        default=False,
    ),
    crop_y: bool = typer.Option(
        False,
        help="Whether to crop the images in the Y dimension.",
        default=False,
    ),
    file_pattern: str = typer.Option(
        ".*",
        help="The pattern for image names.",
    ),
    group_by: str = typer.Option(
        "",
        help="Each group can be cropped to the same bounding box.",
    ),
    inp_dir: pathlib.Path = typer.Option(
        ...,
        help="Path to the input directory containing the image collection.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        readable=True,
    ),
    out_dir: pathlib.Path = typer.Option(
        ...,
        help="Path to the output directory where the cropped images will be saved.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        readable=True,
    ),
) -> None:
    """CLI for the Autocropping Tool tool."""
    logger.info("Starting Autocropping Tool")

    logger.info(f"cropIndividually: {crop_individually}")
    logger.info(f"cropX: {crop_x}")
    logger.info(f"cropY: {crop_y}")
    logger.info(f"filePattern: {file_pattern}")
    logger.info(f"groupBy: {group_by}")
    logger.info(f"inpDir: {inp_dir}")
    logger.info(f"outDir: {out_dir}")

    autocropping()

    pass


if __name__ == "__main__":
    app()
