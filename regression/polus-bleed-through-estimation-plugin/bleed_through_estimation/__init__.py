import logging
from pathlib import Path
from typing import Optional

from utils import constants
from utils import types
from . import models
from . import tile_selectors

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger('estimation')
logger.setLevel(constants.POLUS_LOG)


def estimate_bleed_through(
        *,
        group: list[types.FPFileDict],
        pattern: str,
        selector_name: str,
        model_name: str,
        channel_overlap: int,
        output_dir: Optional[Path],
        metadata_dir: Path,
        kernel_size: int
):
    """ Estimates the bleed-through across adjacent channels among a group of files.

    Args:
        group: A filepattern group containing all tiles and all channels in one round of imaging.
        pattern: The filepattern used for selecting the group.
        selector_name: The method to use for selecting tiles. See `tile_selectors.py`
        model_name: The model to train for estimating bleed-through coefficients. See `models.py`
        channel_overlap: The number of adjacent channels that could cause bleed-through.
        output_dir: If a Path is passed, bleed-through components will be saved in this directory.
        metadata_dir: The bleed-through coefficients for each round will be saved in this directory.
    """
    files = [file['file'] for file in group]

    logger.info(f'selecting tiles...')
    selector = tile_selectors.SELECTORS[selector_name](files, num_tiles_per_channel=10)

    logger.info(f'training models...')
    model = models.MODELS[model_name](files, selector.selected_tiles, selector.image_mins, selector.image_maxs, kernel_size, channel_overlap)

    logger.info(f'exporting coefficient matrix...')
    model.coefficients_to_csv(metadata_dir, pattern, group)
    
    print(model.coefficients)

    # if output_dir is not None:
    #     logger.info('writing bleed-through components...')
    #     model.write_components(output_dir,selector.image_maxs, selector.image_mins)

    return
