import abc
import logging
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import filepattern
import numpy
from bfio import BioReader
from bfio import BioWriter
from skimage import img_as_float32
from sklearn import linear_model

from utils import constants
from utils import helpers
from utils import types

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger('models')
logger.setLevel(constants.POLUS_LOG)


class Model(abc.ABC):
    """ Base class for models that can be trained to estimate bleed-through. """

    __slots__ = '__files', '__selected_tiles', '__channel_overlap', '__coefficients'

    def __init__(
            self,
            files: list[Path],
            selected_tiles: types.TileIndices,
            channel_overlap: int,
    ):
        """ Trains a model on the given files.

        Args:
            files: Paths to images on which the model will be trained.
            selected_tiles: A list of indices of the tiles on which to train the model.
            channel_overlap: The number of adjacent channels to consider for bleed-through estimation.
        """

        self.__files = files
        self.__selected_tiles = selected_tiles
        self.__channel_overlap = min(len(self.__files) - 1, max(1, channel_overlap))

        self.__coefficients = self._fit()

    @abc.abstractmethod
    def _init_model(self):
        """ Initialize a model. """
        pass

    @property
    def coefficients(self) -> numpy.ndarray:
        """ Returns the matrix of mixing coefficients from the trained model. """
        return self.__coefficients

    def _get_neighbors(self, source_index: int) -> list[int]:
        """ Get the neighboring channels for the given source-channel. """
        neighbor_indices = [source_index - j - 1 for j in range(self.__channel_overlap)]
        neighbor_indices.extend(source_index + j + 1 for j in range(self.__channel_overlap))
        neighbor_indices = list(filter(lambda j: 0 <= j < len(self.__files), neighbor_indices))
        return neighbor_indices

    def _fit_thread(self, source_index: int) -> list[float]:
        """ Trains a single model on a single source-channel and returns the mixing coefficients with the adjacent channels.

        This function can be run inside a thread in a ProcessPoolExecutor.

        Args:
            source_index: Index of the source channel.

        Returns:
            A list of the mixing coefficient with each neighboring channel within self.__channel_overlap of the source channel.
        """
        with BioReader(self.__files[source_index]) as source_reader:
            neighbor_readers = [
                BioReader(self.__files[i])
                for i in self._get_neighbors(source_index)
            ]

            logger.info(f'Fitting {self.__class__.__name__} {source_index} on {len(self.__selected_tiles)} tiles...')
            model = self._init_model()
            for i, (z_min, z_max, y_min, y_max, x_min, x_max) in enumerate(self.__selected_tiles):
                logger.info(f'Fitting {self.__class__.__name__} {source_index}: Progress: {100 * i / len(self.__selected_tiles):6.2f} %')

                tiles: list[numpy.ndarray] = [
                    numpy.squeeze(source_reader[y_min:y_max, x_min:x_max, z_min:z_max, 0, 0]).flatten()
                ]
                tiles.extend(
                    numpy.squeeze(reader[y_min:y_max, x_min:x_max, z_min:z_max, 0, 0]).flatten()
                    for reader in neighbor_readers
                )
                tiles: numpy.ndarray = img_as_float32(numpy.asarray(tiles).T)

                model.fit(tiles[:, 1:], tiles[:, 0])

            coefficients = list(map(float, model.coef_))
            del model
            [reader.close() for reader in neighbor_readers]

        return coefficients

    def _fit(self) -> numpy.ndarray:
        """ Fits the model on the images and returns a matrix of mixing coefficients. """

        with ProcessPoolExecutor() as executor:
            coefficients_list: list[Future[list[float]]] = [executor.submit(self._fit_thread, i) for i in range(len(self.__files))]
            coefficients_list: list[list[float]] = [future.result() for future in coefficients_list]

        coefficients_matrix = numpy.zeros(
            shape=(len(self.__files), len(self.__files)),
            dtype=numpy.float32,
        )
        for i, coefficients in enumerate(coefficients_list):
            neighbor_indices = self._get_neighbors(i)
            coefficients_matrix[i, neighbor_indices] = coefficients

        return coefficients_matrix

    def _get_writer_paths(self, destination_dir: Path) -> list[Path]:
        """ Returns an output path for each image in the list of input paths. """
        return [destination_dir.joinpath(helpers.replace_extension(input_path.name)) for input_path in self.__files]

    def coefficients_to_csv(
            self,
            destination_dir: Path,
            pattern: str,
            group: list[types.FPFileDict],
    ):
        """ Write the matrix of mixing coefficients to a csv.

        TODO: Learn how to use `filepattern` better and cut down on the input
         params. Ideally, we would only need the `destination_dir` as an input
         and the name for the csv would be generated from `self.__files`.

        Args:
            destination_dir: Directory were to write the csv.
            pattern: The pattern used for grouping images.
            group: The group of images for which the csv will be generated.
        """
        # noinspection PyTypeChecker
        name = filepattern.output_name(pattern, group, dict())

        metadata_path = destination_dir.joinpath(f'{name}_coefficients.csv')
        with open(metadata_path, 'w') as outfile:
            header = ','.join(f'{c}' for c in range(len(self.__files)))
            outfile.write(f'channel,{header}\n')

            for channel, row in enumerate(self.coefficients):
                row = ','.join(f'{c:.6e}' for c in row)
                outfile.write(f'{channel},{row}\n')

        return

    def write_components(self, destination_dir: Path):
        """ Write out the estimated bleed-through components.

        These bleed-through components can be subtracted from the original
        images to achieve bleed-through correction.

        Args:
            destination_dir: Path to the directory where the output images will
                              be written.
        """
        readers = list(map(BioReader, self.__files))

        with ProcessPoolExecutor() as executor:
            for i, writer_path in enumerate(self._get_writer_paths(destination_dir)):
                neighbor_indices = numpy.nonzero(self.__coefficients[i, :])[0]
                coefficients: list[float] = self.__coefficients[i, :][neighbor_indices]
                neighbor_paths = [self.__files[j] for j in neighbor_indices]

                coefficients: list[tuple[float, Path]] = [
                    (float(c), p)
                    for c, p in zip(coefficients, neighbor_paths)
                ]

                executor.submit(
                    self._write_components_thread,
                    writer_path,
                    readers[i].metadata,
                    coefficients,
                )

        [reader.close() for reader in readers]
        return

    @staticmethod
    def _write_components_thread(
            writer_path: Path,
            metadata: Any,
            coefficients: list[tuple[float, Path]],
    ):
        """ Writes the bleed-through components for a single image.

        This function can be run in a single thread in a ProcessPoolExecutor.

        Args:
            writer_path: Path for the output image.
            metadata: The metadata from the source image.
            coefficients: A list of tuples of the mixing coefficient and a path
                           to the corresponding image in a neighboring channel.
        """
        neighbor_readers = [BioReader(neighbor) for _, neighbor in coefficients]
        coefficients: list[float] = [c for c, _ in coefficients]

        with BioWriter(writer_path, metadata=metadata) as writer:

            logger.info(f'Writing components for {writer_path.name}...')
            num_tiles = helpers.count_tiles_2d(writer)
            for i, (z, y_min, y_max, x_min, x_max) in enumerate(helpers.tile_indices_2d(writer)):
                component = numpy.zeros(
                    shape=(y_max - y_min, x_max - x_min),
                    dtype=writer.dtype,
                )
                if i % 10 == 0:
                    logger.info(f'Writing {writer_path.name}: Progress {100 * i / num_tiles:6.2f} %')

                for c, reader in zip(coefficients, neighbor_readers):
                    tile = numpy.squeeze(reader[y_min:y_max, x_min:x_max, z:z + 1, 0, 0])
                    component += numpy.asarray((c * tile), dtype=writer.dtype)

                writer[y_min:y_max, x_min:x_max, z:z + 1, 0, 0] = component

        [reader.close() for reader in neighbor_readers]
        return


class Lasso(Model):
    """ Uses sklearn.linear_model.Lasso

    This is the model used by the paper and source code (linked below) which we
     used as the seed for this plugin.

    https://doi.org/10.1038/s41467-021-21735-x
    https://github.com/RoysamLab/whole_brain_analysis
    """
    def _init_model(self):
        return linear_model.Lasso(alpha=1e-4, copy_X=True, positive=True, warm_start=True, max_iter=10)


class ElasticNet(Model):
    """ Uses sklearn.linear_model.ElasticNet """
    def _init_model(self):
        return linear_model.ElasticNet(alpha=1e-4, warm_start=True)


class PoissonGLM(Model):
    """ Uses sklearn.linear_model.PoissonRegressor """
    def _init_model(self):
        return linear_model.PoissonRegressor(alpha=1e-4, warm_start=True)


""" A dictionary to let us use a model by name. """
MODELS = {
    'Lasso': Lasso,
    'PoissonGLM': PoissonGLM,
    'ElasticNet': ElasticNet,
}
