import pathlib

import bfio
import numpy


def main():
    data_root = pathlib.Path(__file__).parent.parent.joinpath("data")
    assert data_root.exists(), f"Path not found: {data_root}"

    input_dir = data_root.joinpath("input")
    assert input_dir.exists(), f"Path not found: {input_dir}"

    test_image_path = input_dir.joinpath("t1_x1.ome.tif")
    assert test_image_path.exists(), f"Image not found: {test_image_path}"
    with bfio.BioReader(test_image_path) as reader:
        image = numpy.zeros(shape=(1, reader.Y, reader.X), dtype=reader.dtype)
        image[0, :] = reader[:, :, 0, 0, 0]
    print(f"Read image with shape {image.shape} and dtype {image.dtype} ...")
    # shape: (14892, 13056), dtype: numpy.uint8

    out_path = input_dir.joinpath("t1_x1.npy")
    numpy.save(
        file=str(out_path),
        arr=image,
        allow_pickle=False,
        fix_imports=False,
    )
    print(f"Saved image to {out_path}.")

    return


if __name__ == "__main__":
    main()
