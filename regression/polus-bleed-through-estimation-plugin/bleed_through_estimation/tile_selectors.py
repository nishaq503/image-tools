import abc
import logging
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor
from operator import itemgetter
from pathlib import Path

import numpy
import scipy.stats
from bfio import BioReader

from utils import constants
from utils import helpers
from utils import types

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger('selectors')
logger.setLevel(constants.POLUS_LOG)


class Selector(abc.ABC):
    """ Base class for tile-selection methods. """

    __slots__ = '__files', '__is_high_better', '__num_tiles_per_channel', '__scores', '__selected_tiles', '__min', '__max'

    # noinspection PyTypeChecker
    def __init__(self, files: list[Path], num_tiles_per_channel: int, is_high_better: bool):
        """ Scores all tiles in each image and selects the best few from each image for training a model.
        
        Args:
            files: List of paths to images from which tiles will be selected.
            num_tiles_per_channel: How many tiles to select from each channel.
            is_high_better: Whether higher scoring tiles are better.
        """
        self.__files = files
        self.__is_high_better = is_high_better
        self.__num_tiles_per_channel = num_tiles_per_channel
        self.__min = []
        self.__max = []

        with ProcessPoolExecutor() as executor:
            scores: list[Future[types.ScoresDict]] = [
                executor.submit(self._score_tiles_thread, file_path)
                for file_path in self.__files
            ]
            self.__scores: list[types.ScoresDict] = []
            for future in scores:
                score,image_min,image_max =  future.result()
                self.__scores.append(score)
                self.__min.append(image_min)
                self.__max.append(image_max)

        self.__selected_tiles: types.TileIndices = self._select_best_tiles()

    @property
    def selected_tiles(self) -> types.TileIndices:
        return self.__selected_tiles
    
    @property
    def image_mins(self) -> list:
        return self.__min
    
    @property
    def image_maxs(self) -> list:
        return self.__max

    @abc.abstractmethod
    def _score_tile(self, tile: numpy.ndarray) -> float:
        raise NotImplementedError(f'Any subclass of Criterion must implement the \'score_tile\' method.')

    def _score_tiles_thread(self, file_path: Path) -> types.ScoresDict:
        """ This method runs in a single thread and scores all tiles for a single file.

        Args:
            file_path: Path to image for which the tiles need to be scored.

        Returns:
            A Dictionary of tile-scores. See `utils/types.py`
        """
        with BioReader(file_path) as reader:

            scores_dict: types.ScoresDict = dict()
            logger.info(f'Ranking tiles in {file_path.name}...')
            num_tiles = helpers.count_tiles(reader)
            image_min = numpy.iinfo(reader.dtype).max
            image_max = -numpy.iinfo(reader.dtype).min
            for i, (z_min, z_max, y_min, y_max, x_min, x_max) in enumerate(helpers.tile_indices(reader)):
                if i % 10 == 0:
                    logger.info(f'Ranking tiles in {file_path.name}. Progress {100 * i / num_tiles:6.2f} %')

                tile = numpy.squeeze(reader[y_min:y_max, x_min:x_max, z_min:z_max, 0, 0])
                scores_dict[(z_min, z_max, y_min, y_max, x_min, x_max)] = self._score_tile(tile)
                
                image_min = tile[tile > 0].min(initial=image_min)
                image_max = tile.max(initial=image_max)

        return scores_dict, image_min, image_max

    def _select_best_tiles(self) -> types.TileIndices:
        """ Sort the tiles based on their scores and select the best few from each channel

        Returns:
            List of indices of the best tiles. See `utils.types.py`
        """
        return list(set(
            coordinates for scores_dict in self.__scores
            for coordinates, _ in list(sorted(
                scores_dict.items(),
                key=itemgetter(1),
                reverse=self.__is_high_better,
            ))[:self.__num_tiles_per_channel]
        ))


class HighEntropy(Selector):
    """ Select tiles with the highest entropy. """
    def _score_tile(self, tile: numpy.ndarray) -> float:
        # TODO: Test if results change much if the tile is converted using something like skimage.img_to_uint
        counts, _ = numpy.histogram(tile.flat, bins=128, density=True)
        return float(scipy.stats.entropy(counts))

    def __init__(self, files: list[Path], num_tiles_per_channel: int):
        super().__init__(files, num_tiles_per_channel, True)


class HighMeanIntensity(Selector):
    """ Select tiles with the highest mean intensity. """
    def _score_tile(self, tile: numpy.ndarray) -> float:
        return float(numpy.mean(tile))

    def __init__(self, files: list[Path], num_tiles_per_channel: int):
        super().__init__(files, num_tiles_per_channel, True)


class HighMedianIntensity(Selector):
    """ Select tiles with the highest median intensity. """
    def _score_tile(self, tile: numpy.ndarray) -> float:
        return float(numpy.median(tile))

    def __init__(self, files: list[Path], num_tiles_per_channel: int):
        super().__init__(files, num_tiles_per_channel, True)


class LargeIntensityRange(Selector):
    """ Select tiles with the largest difference between the 90th and 10th percentile intensities. """
    def _score_tile(self, tile: numpy.ndarray) -> float:
        return float(numpy.percentile(tile, 90) - numpy.percentile(tile, 10))

    def __init__(self, files: list[Path], num_tiles_per_channel: int):
        super().__init__(files, num_tiles_per_channel, True)


""" A Dictionary to let us use a Selector by name. """
SELECTORS = {
    'HighEntropy': HighEntropy,
    'HighMeanIntensity': HighMeanIntensity,
    'HighMedianIntensity': HighMedianIntensity,
    'LargeIntensityRange': LargeIntensityRange,
}
