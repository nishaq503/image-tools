from pathlib import Path

import numpy
from bfio import BioReader
from bfio import BioWriter

if __name__ == '__main__':
    test_dir = (
        Path(__file__)
        .parent  # tests
        .parent  # plugin
        .parent  # regression
        .parent  # repo-root
        .joinpath('data')
        .joinpath('bleed_through_estimation')
        .joinpath('temp')
    )
    assert test_dir.exists()

    with BioReader(test_dir.joinpath('c0_input.ome.tif')) as reader_0, \
         BioReader(test_dir.joinpath('c1_input.ome.tif')) as reader_1:

        with BioWriter(test_dir.joinpath('c0_mixed.ome.tif'), metadata=reader_0.metadata) as writer:
            image = (0.9 * numpy.asarray(reader_0[:], dtype=numpy.float32) +
                     0.1 * numpy.asarray(reader_1[:], dtype=numpy.float32))
            writer[:] = numpy.asarray(image, dtype=writer.dtype)

        with BioWriter(test_dir.joinpath('c1_mixed.ome.tif'), metadata=reader_0.metadata) as writer:
            image = (0.5 * numpy.asarray(reader_0[:], dtype=numpy.float32) +
                     0.5 * numpy.asarray(reader_1[:], dtype=numpy.float32))
            writer[:] = numpy.asarray(image, dtype=writer.dtype)

        with BioWriter(test_dir.joinpath('c2_mixed.ome.tif'), metadata=reader_0.metadata) as writer:
            image = (0.1 * numpy.asarray(reader_0[:], dtype=numpy.float32) +
                     0.9 * numpy.asarray(reader_1[:], dtype=numpy.float32))
            writer[:] = numpy.asarray(image, dtype=writer.dtype)
