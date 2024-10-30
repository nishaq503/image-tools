"""CLI for the Autocropping Tool tool."""

import logging
import pathlib

import typer
from polus.images.transforms.autocropping import utils

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = utils.make_logger("polus.images.transforms.autocropping")
app = typer.Typer()


@app.command()
def main(  # noqa: PLR0913
    crop_individually: bool = typer.Option(
        False,
        "--cropIndividually",
        help="Whether to crop each image individually to its own bounding box, or to crop all images to the same bounding box.",  # noqa: E501
    ),
    crop_x: bool = typer.Option(
        True,
        "--cropX",
        help="Whether to crop the images in the X dimension.",
    ),
    crop_y: bool = typer.Option(
        True,
        "--cropY",
        help="Whether to crop the images in the Y dimension.",
    ),
    gradient_threshold: float = typer.Option(
        0.1,
        "--gradientThreshold",
        help="The threshold to use when finding spikes in the entropy gradients.",
    ),
    file_pattern: str = typer.Option(
        ".*",
        "--filePattern",
        help="The pattern for image names.",
    ),
    group_by: str = typer.Option(
        "",
        "--groupBy",
        help="Each group can be cropped to the same bounding box.",
    ),
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Path to the input directory containing the image collection.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        readable=True,
    ),
    out_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
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
    logger.info(f"gradientThreshold: {gradient_threshold}")
    logger.info(f"filePattern: {file_pattern}")
    logger.info(f"groupBy: {group_by}")
    logger.info(f"inpDir: {inp_dir}")
    logger.info(f"outDir: {out_dir}")

    pass


if __name__ == "__main__":
    app()
