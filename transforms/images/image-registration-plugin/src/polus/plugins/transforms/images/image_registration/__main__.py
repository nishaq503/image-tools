"""CLI for the Image Registration plugin."""

import logging
import pathlib
import shutil

import typer
from polus.plugins.transforms.images import image_registration
from polus.plugins.transforms.images.image_registration import register
from polus.plugins.transforms.images.image_registration.utils import constants
from polus.plugins.transforms.images.image_registration.utils.helpers import (
    parse_collection,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger(
    "polus.plugins.transforms.images.image_registration",
)
logger.setLevel(constants.POLUS_LOG)

app = typer.Typer()


@app.command()
def main(  # noqa: PLR0913
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        help="Input image collection",
        exists=True,
        readable=True,
        file_okay=False,
        resolve_path=True,
    ),
    pattern: str = typer.Option(
        ...,
        "--filePattern",
        help="Filename pattern used to separate data",
    ),
    registration_variable: str = typer.Option(
        ...,
        "--registrationVariable",
        help=(
            "Variable to help identify which images "
            "need to be registered to each other"
        ),
    ),
    transformation_variable: str = typer.Option(
        ...,
        "--transformationVariable",
        help="Variable to help identify which images have similar transformation",
    ),
    template: str = typer.Option(
        ...,
        "--template",
        help="Template image to be used for image registration",
    ),
    method: register.Method = typer.Option(
        "Projective",
        "--method",
        help="Image registration method",
    ),
    out_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
        help="Output image collection",
        exists=True,
        writable=True,
        file_okay=False,
        resolve_path=True,
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        help="Preview the outputs without running any computation",
    ),
) -> None:
    """CLI for the Image Registration plugin."""
    if inp_dir.joinpath("images").exists():
        inp_dir = inp_dir.joinpath("images")

    out_dir = out_dir.resolve()

    logger.info(f"--inpDir = {inp_dir}")
    logger.info(f"--filePattern = {pattern}")
    logger.info(f"--registrationVariable = {registration_variable}")
    logger.info(f"--transformationVariable = {transformation_variable}")
    logger.info(f"--template = {template}")
    logger.info(f"--method = {method.value}")
    logger.info(f"--outDir = {out_dir}")

    template_image_path = inp_dir.joinpath(template)

    # parse the input collection
    logger.info("Parsing the input collection and getting registration_dictionary")
    registration_dictionary = parse_collection(
        inp_dir,
        pattern,
        registration_variable,
        transformation_variable,
        template_image_path,
    )

    logger.info("Iterating over registration_dictionary....")
    for registration_set, similar_transformation_set in registration_dictionary.items():
        # registration_dictionary consists of set of already registered images as well
        if registration_set[0] == registration_set[1]:
            similar_transformation_list = similar_transformation_set.tolist()
            similar_transformation_list.append(registration_set[0])
            for image_path in similar_transformation_list:
                image_name = image_path.name
                logger.info(f"Copying image {image_name} to output directory")
                shutil.copy2(image_path, str(out_dir.joinpath(image_name)))
            continue

        # concatenate lists into a string to pass as an argument to argparse
        registration_string = " ".join(map(str, registration_set))
        similar_transformation_string = " ".join(map(str, similar_transformation_list))

        image_registration.main(
            registration_string,
            similar_transformation_string,
            out_dir,
            template,
            method,
        )

    if preview:
        logger.info("--preview = True")
        raise NotImplementedError


if __name__ == "__main__":
    app()


"""
python -m polus.plugins.transforms.images.image_registration \
    --inpDir ~/Documents/axle/data/ratbrain/subset/slides \
    --filePattern "r{r}_c{c+}.ome.tif" \
    --registrationVariable "c" \
    --transformationVariable "r" \
    --template "r1_c0.ome.tif" \
    --method "Projective" \
    --outDir ~/Documents/axle/data/ratbrain/demo-out/registered
"""
