"""Test image assembler plugin."""

import os
<<<<<<< HEAD
<<<<<<< HEAD
from pathlib import Path

from bfio import BioReader
=======
from bfio import BioReader

>>>>>>> de6ea1d (Update: update to new plugin standard:)
=======
from pathlib import Path

from bfio import BioReader
>>>>>>> 254a68a (update : update to new standards.)
from polus.plugins.transforms.images.image_assembler.image_assembler import (
    assemble_images,
)
BACKEND = "python"

<<<<<<< HEAD
<<<<<<< HEAD
from tests.fixtures import data, plugin_dirs, ground_truth_dir

def test_image_assembler(
    data: None,
    plugin_dirs: tuple[Path, Path, Path],
    ground_truth_dir: Path,
) -> None:
=======
from fixtures import (
    data,
    plugin_dirs,
    ground_truth_dir
)

def test_image_assembler(data, plugin_dirs, ground_truth_dir):
>>>>>>> de6ea1d (Update: update to new plugin standard:)
=======
from tests.fixtures import data, plugin_dirs, ground_truth_dir

def test_image_assembler(
    data: None,
    plugin_dirs: tuple[Path, Path, Path],
    ground_truth_dir: Path,
) -> None:
>>>>>>> 254a68a (update : update to new standards.)
    """Test correctness of the image assembler plugin in a basic case."""
    ground_truth_path = ground_truth_dir
    img_path, stitch_path, out_dir = plugin_dirs

    assemble_images(img_path, stitch_path, out_dir, False)

    assert len(os.listdir(ground_truth_path)) == 1
    assert len(os.listdir(out_dir)) == 1

    ground_truth_file = ground_truth_path / os.listdir(ground_truth_path)[0]
    assembled_image_file = out_dir / os.listdir(out_dir)[0]

    # check assembled image against ground truth
    with BioReader(ground_truth_file, backend=BACKEND) as ground_truth:
        with BioReader(assembled_image_file, backend=BACKEND) as image:
            assert ground_truth.shape == image.shape
            assert (ground_truth[:] == image[:]).all()
