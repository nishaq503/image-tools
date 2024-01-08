"""Test fixtures.

Set up all data used in tests.
"""

import io
import pathlib
import random
import shutil
import tempfile
import zipfile

import pytest
import requests


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add options to pytest."""
    parser.addoption(
        "--downloads",
        action="store_true",
        dest="downloads",
        default=False,
        help="run tests that download large data files",
    )


@pytest.fixture()
def local_data() -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:  # type: ignore
    """Generate test data for local testing."""
    # create temporary directory for all data
    data_dir = pathlib.Path(tempfile.mkdtemp(suffix="test_data"))

    inp_dir = data_dir / "inp-dir"
    inp_dir.mkdir()

    out_dir = data_dir / "out_dir"
    out_dir.mkdir()

    gt_dir = data_dir / "gt-dir"
    gt_dir.mkdir()

    # generate the image data
    tile_size = 1024
    offset_x = 1392 - tile_size
    offset_y = 1040 - tile_size

    sv_path = inp_dir / "img-global-positions-1.txt"

    # stitching data
    offsets = [
        {"grid": (0, 0), "file": "img_r001_c001.ome.tif", "position": (0, 0)},
        {
            "grid": (0, 1),
            "file": "img_r001_c002.ome.tif",
            "position": (tile_size - offset_x, 0),
        },
        {
            "grid": (1, 0),
            "file": "img_r002_c001.ome.tif",
            "position": (0, tile_size - offset_y),
        },
        {
            "grid": (1, 1),
            "file": "img_r002_c002.ome.tif",
            "position": (tile_size - offset_x, tile_size - offset_y),
        },
    ]
    for offset in offsets:
        offset["corr"] = round(random.uniform(-1, 1), 10)  # type: ignore[arg-type]  # noqa: S311 E501

    # create stitching vector
    stitching_data = [
        "file: img_r001_c001.ome.tif; corr: -0.0864568939; position: (0, 0); grid: (0, 0);",  # noqa: E501
        "file: img_r001_c002.ome.tif; corr: -0.657176744; position: (656, 0); grid: (0, 1);",  # noqa: E501
        "file: img_r002_c001.ome.tif; corr: 0.7119831612; position: (0, 1008); grid: (1, 0);",  # noqa: E501
        "file: img_r002_c002.ome.tif; corr: 0.2078192665; position: (656, 1008); grid: (1, 1);",  # noqa: E501
    ]

    with sv_path.open("w") as f:
        for row in stitching_data:
            f.write(f"{row}\n")

    # TODO: Generate the Ground Truth data

    yield (inp_dir, out_dir, gt_dir)

    # remove the temporary directory
    shutil.rmtree(data_dir)


@pytest.fixture()
def nist_data() -> tuple[pathlib.Path, pathlib.Path]:  # type: ignore
    """Download the NIST MIST dataset."""
    http_request_timeout = 10

    data_dir = pathlib.Path(tempfile.mkdtemp(suffix="nist_data"))

    # download the stitching vector
    raw_dir = data_dir / "raw-dir"
    if not raw_dir.exists():
        r = requests.get(
            url="https://github.com/usnistgov/MIST/wiki/testdata/Small_Phase_Test_Dataset_Example_Results.zip",
            timeout=http_request_timeout,
        )
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(raw_dir)

    assert raw_dir.exists(), "Could not download stitching vector"

    raw_dir = raw_dir / "Small_Phase_Test_Dataset_Example_Results"
    assert raw_dir.exists(), "downloaded stitching vector is malformed"

    inp_dir = data_dir / "inp-dir"
    inp_dir.mkdir(exist_ok=True)

    out_dir = data_dir / "out-dir"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    if not inp_dir.exists():
        msg = "could not successfully download nist_mist_dataset stitching vector"
        raise FileNotFoundError(msg)

    yield (inp_dir, out_dir)

    # remove the temporary directory
    shutil.rmtree(data_dir)
