""""Provides the image_registration functions."""

import logging
import pathlib
import typing
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path

import bfio
import cv2
import numpy
import numpy as np
from bfio import BioReader
from bfio import BioWriter

from ..utils import constants

logger = logging.getLogger(__name__)
logger.setLevel(constants.POLUS_LOG)


def corr2(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """corr2 Calculate correlation between 2 images.

    Inputs:
        a: A 2-dimensional numpy array
        b: A 2-dimensional numpy array

    Outputs:
        the correlation between a and b
    """
    c = np.sum(a) / np.size(a)
    d = np.sum(b) / np.size(b)

    c = a - c
    d = b - d

    return (c * d).sum() / np.sqrt((c * c).sum() * (d * d).sum())


def get_transform(
    moving_image: numpy.ndarray,
    reference_image: numpy.ndarray,
    max_val: float,
    min_val: float,
    method: str,
) -> numpy.ndarray:
    """Calculate homography matrix transform.

    This function registers the moving image with reference image

    Inputs:
        moving_image = Image to be transformed
        reference_image=  reference Image
    Outputs:
        homography= transformation applied to the moving image
    """
    # max number of features to be calculated using ORB
    max_features = 500_000
    # initialize orb feature matcher
    orb = cv2.ORB_create(max_features)

    # Normalize images and convert to appropriate type
    moving_image_norm = cv2.GaussianBlur(moving_image, (3, 3), 0)
    moving_image_norm = (moving_image_norm - min_val) / (max_val - min_val)
    moving_image_norm = (moving_image_norm * 255).astype(np.uint8)

    reference_image_norm = cv2.GaussianBlur(reference_image, (3, 3), 0)
    reference_image_norm = (reference_image_norm - min_val) / (max_val - min_val)
    reference_image_norm = (reference_image_norm * 255).astype(np.uint8)

    # find keypoints and descriptors in moving and reference image
    keypoints1, descriptors1 = orb.detectAndCompute(moving_image_norm, None)
    keypoints2, descriptors2 = orb.detectAndCompute(reference_image_norm, None)

    # Escape if one image does not have descriptors
    if not (
        isinstance(descriptors1, np.ndarray) and isinstance(descriptors2, np.ndarray)
    ):
        return None

    # match and sort the descriptos using hamming distance
    flann_params = {
        "algorithm": 6,
        "table_number": 6,
        "key_size": 12,
        "multi_probe_level": 1,  # FLANN_INDEX_LSH
    }
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    matches = matcher.match(descriptors1, descriptors2, None)
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    # extract top 25% of matches
    good_match_percent = 0.25
    num_good_matches = max(1, int(len(matches) * good_match_percent))
    matches = matches[:num_good_matches]

    # extract the point coordinates from the keypoints
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # If no matching points, return None
    if points1.shape[0] == 0 or points2.shape[0] == 0:
        return None

    # calculate the homography matrix
    if method == "Projective":
        homography, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
    elif method == "Affine":
        homography, _ = cv2.estimateAffine2D(points1, points2, method=cv2.RANSAC)
    elif method == "PartialAffine":
        homography, _ = cv2.estimateAffinePartial2D(points1, points2, method=cv2.RANSAC)

    return homography


def get_scale_factor(height: int, width: int) -> int:
    """This function returns the appropriate scale factor w.r.t to a target size.

    Target size has been fixed to 5 megapixels.

    Args:
        height: Image height
        width: Image width

    Returns:
        scale factor
    """
    target_size = 5_000_000  # 5 megapixels
    scale_factor = np.sqrt((height * width) / target_size)
    return int(scale_factor) if scale_factor > 1 else 1


def get_scaled_down_images(
    image: bfio.BioReader,
    scale_factor: int,
    get_max: bool = False,
) -> numpy.ndarray:
    """This function returns the scaled down version of an image.

    Args:
        image: A BioReader object
        scale_factor: the factor by which the image needs to be scaled down
        get_max: If True, return the max and min values of the image.

    Returns:
        scaled down version of the input image.
    """
    # Calculate scaling variables
    stride = int(scale_factor * np.floor(4096 / scale_factor))
    width = np.ceil(image.Y / scale_factor).astype(int)
    height = np.ceil(image.X / scale_factor).astype(int)

    # Initialize the output
    rescaled_image = np.zeros((width, height), dtype=image.dtype)

    # If max value is requested, initialize the variables
    if get_max:
        max_val = 0
        min_val = np.inf

    def load_and_scale(  # noqa: PLR0913
        x_,  # noqa: ANN001
        y_,  # noqa: ANN001
        x,  # noqa: ANN001
        y,  # noqa: ANN001
        get_max=get_max,  # noqa: ANN001
        reader=image,  # noqa: ANN001
        scale_factor=scale_factor,  # noqa: ANN001
        rescaled_image=rescaled_image,  # noqa: ANN001
    ) -> typing.Optional[tuple[float, float]]:
        """load_and_scale Load a section of an image and downscale.

        This is a transient method, and only works within the get
        scaled_down_images method.
        It's used to thread out loading and downscaling of large images.

        """
        # Read an image tile
        tile = reader[y_[0] : y_[1], x_[0] : x_[1], 0, 0, 0]

        # Average the image for scaling
        blurred_image = cv2.boxFilter(tile, -1, (scale_factor, scale_factor))

        # Collect pixels for downscaled image
        rescaled_image[y[0] : y[1], x[0] : x[1]] = blurred_image[
            ::scale_factor,
            ::scale_factor,
        ]

        if get_max:
            return np.max(tile), np.min(tile)

        return None

    # Load and downscale the image
    threads = []
    with ThreadPoolExecutor(max([cpu_count() // 2, 1])) as executor:
        for x in range(0, image.X, stride):
            x_max = min(x + stride, image.X)  # max x to load
            xi = int(x // scale_factor)  # initial scaled x-index
            xe = int(np.ceil(x_max / scale_factor))  # ending scaled x-index
            for y in range(0, image.Y, stride):
                y_max = min(y + stride, image.Y)  # max y to load
                yi = int(y // scale_factor)  # initial scaled y-index
                ye = int(np.ceil(y_max / scale_factor))  # ending scaled y-index

                threads.append(
                    executor.submit(
                        load_and_scale,
                        [x, x_max],
                        [y, y_max],
                        [xi, xe],
                        [yi, ye],
                    ),
                )

    # Return max and min values if requested
    if get_max:
        results = [thread.result() for thread in threads]
        max_val = max(
            result[0] for result in results  #  type: ignore[index,assignment]
        )
        min_val = min(result[1] for result in results)  #  type: ignore[index]
        return rescaled_image, max_val, min_val

    return rescaled_image


def register_image(  # noqa: PLR0913
    br_ref,  # noqa: ANN001
    br_mov,  # noqa: ANN001
    bw,  # noqa: ANN001
    xt,  # noqa: ANN001
    yt,  # noqa: ANN001
    xm,  # noqa: ANN001
    ym,  # noqa: ANN001
    x,  # noqa: ANN001
    y,  # noqa: ANN001
    x_crop,  # noqa: ANN001
    y_crop,  # noqa: ANN001
    max_val,  # noqa: ANN001
    min_val,  # noqa: ANN001
    method,  # noqa: ANN001
    rough_homography_upscaled,  # noqa: ANN001
) -> numpy.ndarray:
    """register_image Register one section of two images.

    This method is designed to be used within a thread. It registers
    one section of two different images, saves the output, and
    returns the homography matrix used to transform the image.

    """
    # Load a section of the reference and moving images
    ref_tile = br_ref[
        yt[0] : yt[1],
        xt[0] : xt[1],
        0,
        0,
        0,
    ]
    mov_tile = br_mov[
        ym[0] : ym[1],
        xm[0] : xm[1],
        0,
        0,
        0,
    ]

    # Get the transformation matrix
    projective_transform = get_transform(mov_tile, ref_tile, max_val, min_val, method)

    # Use the rough transformation matrix if no matrix was returned
    is_rough = False
    if not isinstance(projective_transform, np.ndarray):
        is_rough = True
        projective_transform = rough_homography_upscaled

    # Transform the moving image
    if method == "Projective":
        transformed_image = cv2.warpPerspective(
            mov_tile,
            projective_transform,
            (xt[1] - xt[0], yt[1] - yt[0]),
        )
    else:
        transformed_image = cv2.warpAffine(
            mov_tile,
            projective_transform,
            (xt[1] - xt[0], yt[1] - yt[0]),
        )

    # Determine the correlation between the reference and transformed moving image
    corr = corr2(ref_tile, transformed_image)

    # If the correlation is bad, try using the rough transform instead
    if corr < 0.4 and not is_rough:  # noqa: PLR2004
        if method == "Projective":
            transformed_image = cv2.warpPerspective(
                mov_tile,
                rough_homography_upscaled,
                (xt[1] - xt[0], yt[1] - yt[0]),
            )
        else:
            transformed_image = cv2.warpAffine(
                mov_tile,
                rough_homography_upscaled,
                (xt[1] - xt[0], yt[1] - yt[0]),
            )
        projective_transform = rough_homography_upscaled

    # Write the transformed moving image
    y_max = y + y_crop[1] - y_crop[0]
    x_max = x + x_crop[1] - x_crop[0]
    bw[y:y_max, x:x_max, 0, 0, 0] = transformed_image[
        y_crop[0] : y_crop[1],
        x_crop[0] : x_crop[1],
    ]

    return projective_transform


def apply_transform(  # noqa: PLR0913
    br_mov: bfio.BioReader,
    bw: bfio.BioWriter,
    tiles: tuple[list[int], list[int], list[int], list[int]],
    shape: tuple[int, int, list[int], list[int]],
    transform: np.ndarray,
    method: str,
) -> None:
    """Apply a transform to an image.

    This method is designed to be used within a thread. It loads
    a section of an image, applies a transform, and saves the
    transformed image to file.

    Args:
        br_mov: A BioReader object for the moving image.
        bw: A BioWriter object for the output image.
        tiles: A tuple containing the tile indices for the moving and
            reference images.
        shape: A tuple containing the image coordinates and shape.
        transform: The transformation matrix to apply to the moving image.
        method: The method used to calculate the transformation matrix.
    """
    # Get the tile indices
    xm, ym, xt, yt = tiles

    # Read the moving image tile
    mov_tile = br_mov[
        ym[0] : ym[1],
        xm[0] : xm[1],
        0,
        0,
        0,
    ]

    # Get the image coordinates and shape
    x, y, x_crop, y_crop = shape

    # Transform the moving image
    if method == "Projective":
        transformed_image = cv2.warpPerspective(
            mov_tile,
            transform,
            (xt[1] - xt[0], yt[1] - yt[0]),
        )
    else:
        transformed_image = cv2.warpAffine(
            mov_tile,
            transform,
            (xt[1] - xt[0], yt[1] - yt[0]),
        )

    # Write the transformed image to the output file
    y_max = y + y_crop[1] - y_crop[0]
    x_max = x + x_crop[1] - x_crop[0]
    bw[y:y_max, x:x_max, 0, 0, 0] = transformed_image[
        y_crop[0] : y_crop[1],
        x_crop[0] : x_crop[1],
    ]


def main(  # noqa: PLR0915, C901
    registration_string: str,
    similar_transformation_string: str,
    out_dir: pathlib.Path,
    template: str,
    method: str,
) -> None:
    """The main function for image_registration.

    This function registers a set of images to a template image.

    Args:
        registration_string: A string containing the paths to the reference and
            moving images.
        similar_transformation_string: A string containing the paths to the
            images that have similar transformation as the moving image.
        out_dir: The output directory.
        template: The name of the template image.
        method: The method to use for registration. Options are "Projective",
            "Affine", and "PartialAffine".
    """
    # Set up the number of threads for each task
    read_workers = max(cpu_count() // 3, 1)
    write_workers = max(cpu_count() - 1, 2)
    loop_workers = max(3 * cpu_count() // 4, 2)

    # extract filenames from registration_string and similar_transformation_string
    registration_set = registration_string.split()
    similar_transformation_set = similar_transformation_string.split()

    filename_len = len(template)

    # separate the filename of the moving image from the complete path
    moving_image_name = registration_set[1][-1 * filename_len :]

    # read and downscale reference image
    br_ref = BioReader(registration_set[0], max_workers=write_workers)
    scale_factor = get_scale_factor(br_ref.Y, br_ref.X)
    logger.info(f"Scale factor: {scale_factor}")

    # initialize the scale factor and scale matrix(to be used to upscale the
    # transformation matrices)
    if method == "Projective":
        scale_matrix = np.array(
            [
                [1, 1, scale_factor],
                [1, 1, scale_factor],
                [1 / scale_factor, 1 / scale_factor, 1],
            ],
        )
    else:
        scale_matrix = np.array(
            [
                [1 / scale_factor, 1 / scale_factor, 1],
                [1 / scale_factor, 1 / scale_factor, 1],
            ],
        )

    logger.info(
        "Reading and downscaling reference image: {}".format(
            Path(registration_set[0]).name,
        ),
    )
    reference_image_downscaled, max_val, min_val = get_scaled_down_images(
        br_ref,
        scale_factor,
        get_max=True,
    )
    br_ref.max_workers = read_workers

    # read moving image
    logger.info(
        "Reading and downscaling moving image: {}".format(
            Path(registration_set[1]).name,
        ),
    )
    br_mov = BioReader(registration_set[1], max_workers=write_workers)
    moving_image_downscaled = get_scaled_down_images(br_mov, scale_factor)
    br_mov.max_workers = read_workers

    # calculate rough transformation between scaled down reference and moving image
    logger.info("calculating rough homography...")
    rough_homography_downscaled = get_transform(
        moving_image_downscaled,
        reference_image_downscaled,
        max_val,
        min_val,
        method,
    )

    # upscale the rough homography matrix
    logger.info("Inverting homography...")
    if method == "Projective":
        rough_homography_upscaled = (
            numpy.eye(scale_matrix.shape[0])
            if rough_homography_downscaled is None
            else rough_homography_downscaled * scale_matrix
        )
        homography_inverse = np.linalg.inv(rough_homography_upscaled)
    else:
        rough_homography_upscaled = rough_homography_downscaled
        homography_inverse = cv2.invertAffineTransform(rough_homography_downscaled)

    # Initialize the output file
    bw = BioWriter(
        out_dir.joinpath(Path(registration_set[1]).name),
        metadata=br_mov.metadata,
        max_workers=write_workers,
    )
    bw.X = br_ref.X
    bw.Y = br_ref.Y
    bw.Z = 1
    bw.C = 1
    bw.T = 1

    # transformation variables
    reg_shape = []
    reg_tiles = []
    reg_homography = []

    # Loop through image tiles and start threads
    logger.info("Starting threads...")
    threads = []
    first_tile = True
    with ThreadPoolExecutor(max_workers=loop_workers) as executor:
        for x in range(0, br_ref.X, 2048):
            for y in range(0, br_ref.Y, 2048):
                # Get reference/template image coordinates
                xt = [max(0, x - 1024), min(br_ref.X, x + 2048 + 1024)]
                yt = [max(0, y - 1024), min(br_ref.Y, y + 2048 + 1024)]

                # Use the rough homography to get coordinates in the moving image
                coords = np.array(
                    [
                        [xt[0], xt[0], xt[1], xt[1]],
                        [yt[0], yt[1], yt[1], yt[0]],
                        [1, 1, 1, 1],
                    ],
                    dtype=np.float64,
                )

                coords = np.matmul(homography_inverse, coords)

                mins = np.min(coords, axis=1)
                maxs = np.max(coords, axis=1)

                xm = [
                    int(np.floor(max(mins[0], 0))),
                    int(np.ceil(min(maxs[0], br_mov.X))),
                ]
                ym = [
                    int(np.floor(max(mins[1], 0))),
                    int(np.ceil(min(maxs[1], br_mov.Y))),
                ]

                reg_tiles.append((xm, ym, xt, yt))

                # Get cropping dimensions
                x_crop = [1024 if xt[0] > 0 else 0]
                x_crop.append(
                    2048 + x_crop[0]
                    if xt[1] - xt[0] >= 3072  # noqa: PLR2004
                    else xt[1] - xt[0] + x_crop[0],
                )
                y_crop = [1024 if yt[0] > 0 else 0]
                y_crop.append(
                    2048 + y_crop[0]
                    if yt[1] - yt[0] >= 3072  # noqa: PLR2004
                    else yt[1] - yt[0] + y_crop[0],
                )
                reg_shape.append((x, y, x_crop, y_crop))

                # Start a thread to register the tiles
                threads.append(
                    executor.submit(
                        register_image,
                        br_ref,
                        br_mov,
                        bw,
                        xt,
                        yt,
                        xm,
                        ym,
                        x,
                        y,
                        x_crop,
                        y_crop,
                        max_val,
                        min_val,
                        method,
                        rough_homography_upscaled,
                    ),
                )

                # Bioformats require the first tile be written before any other tile
                if first_tile:
                    logger.info("Waiting for first_tile to finish...")
                    first_tile = False
                    threads[0].result()

        # Wait for threads to finish, track progress
        for thread_num in range(len(threads)):
            if thread_num % 10 == 0:
                logger.info(
                    "Registration progress: {:6.2f}%".format(
                        100 * thread_num / len(threads),
                    ),
                )
            reg_homography.append(threads[thread_num].result())

    # Close the image
    bw.close()
    logger.info(f"Registration progress: {100.0:6.2f}%")

    # iterate across all images which have the similar transformation as
    # the moving image above
    for moving_image_path in similar_transformation_set:
        # separate image name from the path to it
        moving_image_name = moving_image_path[-1 * filename_len :]

        logger.info(f"Applying registration to image: {moving_image_name}")

        br_mov = BioReader(moving_image_path, max_workers=read_workers)

        bw = BioWriter(
            out_dir.joinpath(moving_image_name),
            metadata=br_mov.metadata,
            max_workers=write_workers,
        )
        bw.X = br_ref.X
        bw.Y = br_ref.Y
        bw.Z = 1
        bw.C = 1
        bw.T = 1

        # Apply transformation to remaining images
        logger.info(f"Transformation progress: {0.0:5.2f}%")
        threads = []
        with ThreadPoolExecutor(loop_workers) as executor:
            first_tile = True
            for tile, shape, transform in zip(reg_tiles, reg_shape, reg_homography):
                # Start transformation threads
                threads.append(
                    executor.submit(
                        apply_transform,
                        br_mov,
                        bw,
                        tile,
                        shape,
                        transform,
                        method,
                    ),
                )

                # The first tile must be written before all other tiles
                if first_tile:
                    first_tile = False
                    threads[0].result()

            # Wait for threads to finish and track progress
            for thread_num in range(len(threads)):
                if thread_num % 10 == 0:
                    logger.info(
                        "Transformation progress: {:6.2f}%".format(
                            100 * thread_num / len(threads),
                        ),
                    )
                threads[thread_num].result()
        logger.info(f"Transformation progress: {100.0:6.2f}%")

        bw.close()

    br_ref.close()
