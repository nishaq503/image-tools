"""Provides the essential function for the plugin."""


import pathlib


def sv2mj(
    inp_path: pathlib.Path,  # noqa: ARG001
    out_dir: pathlib.Path,  # noqa: ARG001
) -> None:
    """Convert stitching vector to MicroJSON.

    Args:
        inp_path: Path to the stitching vector file.
        out_dir: Path to the output directory.
    """
    msg = "This function is not implemented yet."
    raise NotImplementedError(msg)


def output_path(
    inp_path: pathlib.Path,  # noqa: ARG001
    out_dir: pathlib.Path,  # noqa: ARG001
) -> pathlib.Path:
    """Generate the output path for the stitching vector file."""
    msg = "This function is not implemented yet."
    raise NotImplementedError(msg)


__all__ = [
    "sv2mj",
    "output_path",
]
