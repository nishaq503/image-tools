"""Testing the CLI for the plugin."""

import json
import pathlib

from polus.plugins.formats.stitching_vector_to_microjson.__main__ import app
from typer.testing import CliRunner


def test_cli(local_data: tuple[pathlib.Path, pathlib.Path, pathlib.Path]):
    """Test the command line."""
    runner = CliRunner()

    inp_dir, out_dir, _ = local_data

    result = runner.invoke(
        app,
        [
            "--inpPath",
            str(inp_dir),
            "--filePattern",
            "*.txt",
            "--outDir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0


def test_cli_preview(local_data: tuple[pathlib.Path, pathlib.Path, pathlib.Path]):
    """Test the preview option."""
    runner = CliRunner()

    inp_dir, out_dir, _ = local_data

    result = runner.invoke(
        app,
        [
            "--imgPath",
            str(inp_dir),
            "--filePattern",
            "*.txt",
            "--outDir",
            str(out_dir),
            "--preview",
        ],
    )

    print(result.exception)
    print(result.stdout)
    assert result.exit_code == 0

    with out_dir.joinpath("preview.json").open("r") as file:
        plugin_json = json.load(file)

    # verify we generate the preview file
    result = plugin_json["outputDir"]
    assert len(result) == 1
    assert pathlib.Path(result[0]).name == "img_r00(1-2)_c00(1-2).ome.tif"


def test_cli_bad_input(local_data: tuple[pathlib.Path, pathlib.Path, pathlib.Path]):
    """Test bad inputs."""
    runner = CliRunner()

    inp_dir, out_dir, _ = local_data
    inp_dir = pathlib.Path("does_not_exists")

    result = runner.invoke(
        app,
        [
            "--imgPath",
            str(inp_dir),
            "--filePattern",
            "*.txt",
            "--outDir",
            str(out_dir),
        ],
    )

    assert result.exc_info[0] is FileNotFoundError
