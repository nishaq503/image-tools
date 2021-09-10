import abc
import logging
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

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

    __slots__ = '__files', '__channel_overlap', '__coefficients'

    def __init__(
            self,
            files: list[Path],
            selected_tiles: types.TileIndices,
            image_mins,
            image_maxs,
            channel_overlap: int,
    ):
        """ Trains a model on the given files.

        Args:
            files: Paths to images on which the model will be trained.
            selected_tiles: A list of indices of the tiles on which to train the model.
            channel_overlap: The number of adjacent channels to consider for bleed-through estimation.
        """

        self.__files = files
        self.__channel_overlap = min(len(self.__files) - 1, max(1, channel_overlap))

        self.__coefficients = self._fit(selected_tiles,image_mins,image_maxs)

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
        neighbor_indices = [source_index - i - 1 for i in range(self.__channel_overlap)]
        neighbor_indices.extend(source_index + i + 1 for i in range(self.__channel_overlap))
        neighbor_indices = list(filter(lambda i: 0 <= i < len(self.__files), neighbor_indices))
        return neighbor_indices

    def _fit_thread(self, source_index: int, selected_tiles: types.TileIndices, image_mins, image_maxs) -> list[float]:
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
            
            mins = [image_mins[source_index]]
            mins.extend([image_mins[i] for i in self._get_neighbors(source_index)])
            maxs = [image_maxs[source_index]]
            maxs.extend([image_maxs[i] for i in self._get_neighbors(source_index)])

            logger.info(f'Fitting {self.__class__.__name__} {source_index} on {len(selected_tiles)} tiles...')
            model = self._init_model()
            for i, (z_min, z_max, y_min, y_max, x_min, x_max) in enumerate(selected_tiles):
                logger.info(f'Fitting {self.__class__.__name__} {source_index}: Progress: {100 * i / len(selected_tiles):6.2f} %')

                tiles: list[numpy.ndarray] = [
                    numpy.squeeze(source_reader[y_min:y_max, x_min:x_max, z_min:z_max, 0, 0]).flatten()
                ]
                tiles.extend(
                    numpy.squeeze(reader[y_min:y_max, x_min:x_max, z_min:z_max, 0, 0]).flatten()
                    for reader in neighbor_readers
                )
                
                tiles: numpy.ndarray = numpy.asarray(tiles).T.astype(numpy.float32)
                tiles = (tiles - numpy.asarray(mins)) / (numpy.asarray(maxs) - numpy.asarray(mins))
                tiles[tiles < 0] = 0

                source, neighbors = tiles[:, 0], tiles[:, 1:]
                interactions = numpy.sqrt(numpy.expand_dims(source, axis=1) * neighbors)
                neighbors = numpy.concatenate([neighbors, interactions], axis=1)

                model.fit(neighbors, source)

            coefficients = list(map(float, model.coef_))
            del model
            [reader.close() for reader in neighbor_readers]

        return coefficients

    def _fit(self, selected_tiles: types.TileIndices, image_mins, image_maxs) -> numpy.ndarray:
        """ Fits the model on the images and returns a matrix of mixing coefficients. """

        with ProcessPoolExecutor() as executor:
            coefficients_list: list[Future[list[float]]] = [
                executor.submit(self._fit_thread, source_index, selected_tiles, image_mins, image_maxs)
                for source_index in range(len(self.__files))
            ]
            coefficients_list: list[list[float]] = [future.result() for future in coefficients_list]

        coefficients_matrix = numpy.zeros(
            shape=(len(self.__files), 2 * len(self.__files)),
            dtype=numpy.float32,
        )
        for i, coefficients in enumerate(coefficients_list):
            neighbor_indices = self._get_neighbors(i)
            interaction_indices = [j + len(self.__files) for j in neighbor_indices]

            neighbor_indices = neighbor_indices + interaction_indices
            coefficients_matrix[i, neighbor_indices] = coefficients

        return coefficients_matrix

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
            header_1 = ','.join(f'c{c}' for c in range(len(self.__files)))
            header_2 = ','.join(f'i{c}' for c in range(len(self.__files)))
            outfile.write(f'channel,{header_1},{header_2}\n')

            for channel, row in enumerate(self.coefficients):
                row = ','.join(f'{c:.6e}' for c in row)
                outfile.write(f'c{channel},{row}\n')

        return

    def write_components(self, destination_dir: Path, image_maxs,image_mins):
        """ Write out the estimated bleed-through components.

        These bleed-through components can be subtracted from the original
        images to achieve bleed-through correction.

        Args:
            destination_dir: Path to the directory where the output images will
                              be written.
        """
        
        with ProcessPoolExecutor() as executor:
            processes = []
            for source_index, input_path in enumerate(self.__files):
                writer_name = helpers.replace_extension(input_path.name)
                processes.append(executor.submit(
                    self._write_components_thread,
                    destination_dir,
                    writer_name,
                    source_index,
                    image_maxs,
                    image_mins
                ))
                
            for process in processes:
                process.result()
        return

    def _write_components_thread(
            self,
            output_dir: Path,
            image_name: str,
            source_index: int,
            image_maxs,
            image_mins
    ):
        """ Writes the bleed-through components for a single image.

        This function can be run in a single thread in a ProcessPoolExecutor.

        Args:
            output_dir: Path for the directory of the bleed-through components.
            image_name: name of the source image.
            source_index: index of the source channel.
        """
        neighbor_indices = self._get_neighbors(source_index)

        neighbor_readers = [BioReader(self.__files[i]) for i in neighbor_indices]
        neighbor_coefficients = [float(c) for c in self.__coefficients[source_index, [neighbor_indices]][0]]
        neighbor_maxs = [image_maxs[i] for i in neighbor_indices]
        neighbor_mins = [image_mins[i] for i in neighbor_indices]

        interaction_indices = [j + len(self.__files) for j in neighbor_indices]
        interaction_coefficients = [float(c) for c in self.__coefficients[source_index, [interaction_indices]][0]]

        with BioReader(self.__files[source_index]) as reader:
            metadata = reader.metadata
            num_tiles = helpers.count_tiles_2d(reader)
            tile_indices = list(helpers.tile_indices_2d(reader))

            original_writer = BioWriter(output_dir.joinpath(image_name}), metadata=metadata)

            logger.info(f'Writing components for {image_name}...')
            for i, (z, y_min, y_max, x_min, x_max) in enumerate(tile_indices):
                tile = numpy.squeeze(reader[y_min:y_max, x_min:x_max, z:z + 1, 0, 0])

                original_component = numpy.zeros_like(tile)

                if i % 10 == 0:
                    logger.info(f'Writing {image_name}: Progress {100 * i / num_tiles:6.2f} %')

                for neighbor_reader, original_c, interaction_c, maxs, mins in zip(neighbor_readers, neighbor_coefficients, interaction_coefficients,neighbor_maxs,neighbor_mins):
                    neighbor_tile = numpy.squeeze(neighbor_reader[y_min:y_max, x_min:x_max, z:z + 1, 0, 0]).astype(numpy.float32)
                    neighbor_tile = (neighbor_tile - mins) / (maxs - mins)
                    if original_c != 0:
                        
                        # apply the coefficient
                        current_component = original_c * neighbor_tile
                        current_component[current_component < 0] = 0
                        
                        # Rescale, but do not add in the minimum value offset.
                        current_component *= (maxs - mins)
                        original_component += current_component.astype(tile.dtype)

                original_writer[y_min:y_max, x_min:x_max, z:z + 1, 0, 0] = original_component

        original_writer.close()
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
