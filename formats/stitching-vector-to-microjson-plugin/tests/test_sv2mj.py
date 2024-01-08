"""Test the plugin."""

import os
import pathlib

import pytest

from polus.plugins.formats import stitching_vector_to_microjson as plugin


def test_image_assembler(
    local_data: tuple[pathlib.Path, pathlib.Path, pathlib.Path]
) -> None:
    """Test correctness of the image assembler plugin in a basic case."""
    inp_dir, out_dir, gt_dir = local_data
    pattern = "*.txt"

    inp_files = list(inp_dir.glob(pattern))
    for inp_path in inp_files:
        plugin.sv2mj(inp_path, out_dir)

    assert len(os.listdir(gt_dir)) == len(inp_files)
    assert len(os.listdir(out_dir)) == len(inp_files)

    expected_out_paths = [
        plugin.output_path(inp_path, out_dir) for inp_path in inp_files
    ]
    for out_path in expected_out_paths:
        assert out_path.exists()
        test_correctness(inp_path, out_path)


@pytest.mark.skipif("not config.getoption('downloads')")
def test_image_assembler(nist_data: tuple[pathlib.Path, pathlib.Path]) -> None:
    """Test conversion of NIST stitching vector to MicroJSON."""

    inp_dir, out_dir = nist_data
    inp_path = inp_dir / "img-global-positions-1.txt"

    plugin.sv2mj(inp_path, out_dir)

    expected_out_path = out_dir / "img-global-positions-1.json"
    assert expected_out_path.exists()

    test_correctness(inp_path, expected_out_path)


def test_correctness(
    inp_path: pathlib.Path,
    out_path: pathlib.Path,
) -> None:
    """Test correctness of the output file."""
    out_name = inp_path.stem + ".json"

    assert out_path.name == out_name

    # TODO: add more tests
