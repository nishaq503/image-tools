"""CLI for the Autocropping Tool tool."""

import json
import logging
import pathlib

import filepattern
import typer
from polus.images.transforms.autocropping import autocrop_group
from polus.images.transforms.autocropping import utils

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = utils.make_logger("polus.images.transforms.autocropping")
app = typer.Typer()


@app.command()
def main(  # noqa: PLR0913
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
    crop_individually: bool = typer.Option(
        False,
        "--cropIndividually",
        help=(
            "Whether to crop each image individually to its own bounding box,"
            " or to crop all images in a group to the same bounding box."
        ),
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
    out_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
        help="Path to the output directory where the cropped images will be saved.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        writable=True,
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        help=(
            "If True, return the paths to the images that would be saved "
            "without actually performing any other computation."
        ),
    ),
) -> None:
    """CLI for the Autocropping Tool tool."""
    logger.info("Starting Autocropping Tool")

    logger.info(f"inpDir: {inp_dir}")
    logger.info(f"filePattern: {file_pattern}")
    logger.info(f"groupBy: {group_by}")
    logger.info(f"cropIndividually: {crop_individually}")
    logger.info(f"cropX: {crop_x}")
    logger.info(f"cropY: {crop_y}")
    logger.info(f"gradientThreshold: {gradient_threshold}")
    logger.info(f"outDir: {out_dir}")

    fp = filepattern.FilePattern(file_pattern)

    groups: list[list[pathlib.Path]] = []

    for group, files in fp(group_by=list(group_by)):
        logger.info(f"Group: {group}")
        logger.info(f"Files: {files}")
        inp_files: list[pathlib.Path] = [p for _, [p] in files]
        logger.info(f"inp_files: {inp_files}")

        groups.append(inp_files)

    if preview:
        preview_json: dict[str, list[str]] = {"files": []}
        for inp_files in groups:
            for inp_file in inp_files:
                preview_json["files"].append(str(inp_file))

        with (out_dir / "preview.json").open("w") as f:
            json.dump(preview_json, f, indent=2)

    else:
        for inp_files in groups:
            autocrop_group(
                inp_files=inp_files,
                crop_x=crop_x,
                crop_y=crop_y,
                crop_individually=crop_individually,
                threshold=gradient_threshold,
                out_dir=out_dir,
            )

    logger.info("Autocropping Tool completed successfully.")


if __name__ == "__main__":
    app()
