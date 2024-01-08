"""CLI for the plugin."""

import logging
import os
import pathlib

import typer

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
logger = logging.getLogger("polus.plugins.formats.stitching_vector_to_microjson")
logger.setLevel(POLUS_LOG)

logging.getLogger("bfio").setLevel(logging.CRITICAL)

app = typer.Typer(help="Stitching Vector to MicroJSON Plugin.")


def generate_preview(
    inp_dir: pathlib.Path,  # noqa: ARG001
    out_dir: pathlib.Path,  # noqa: ARG001
) -> None:
    """Generate preview of the plugin outputs."""
    msg = "This function is not implemented yet."
    raise NotImplementedError(msg)


@app.command()
def main(
    inp_dir: pathlib.Path = typer.Option(
        ...,
        "--inpDir",
        "-i",
        help="Input stitching vector directory.",
        exists=True,
        readable=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    pattern: str = typer.Option(
        "*.txt",
        "--filePattern",
        "-p",
        help="Pattern for input stitching vector files.",
    ),
    out_dir: pathlib.Path = typer.Option(
        ...,
        "--outDir",
        "-o",
        help="Path to the output directory.",
        exists=True,
        writable=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    preview: bool = typer.Option(
        False,
        "--preview",
        "-p",
        help="Generate preview of expected outputs.",
    ),
) -> None:
    """Assemble images from a single stitching vector."""
    logger.info("Starting stitching vector to MicroJSON plugin.")

    logger.info(f"inpDir = {inp_dir}")
    logger.info(f"filePattern = {pattern}")
    logger.info(f"outDir = {out_dir}")
    logger.info(f"preview = {preview}")

    if preview:
        generate_preview(inp_dir, out_dir)
        return

    msg = "This function is not implemented yet."
    raise NotImplementedError(msg)


if __name__ == "__main__":
    app()
