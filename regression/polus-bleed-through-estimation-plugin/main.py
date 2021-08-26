import argparse
import logging
from pathlib import Path

from filepattern import FilePattern

from bleed_through_estimation import estimate_bleed_through
from bleed_through_estimation.models import MODELS
from bleed_through_estimation.tile_selectors import SELECTORS
from utils import constants

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger('main')
logger.setLevel(constants.POLUS_LOG)


if __name__ == '__main__':
    logger.info("Parsing arguments...")
    _parser = argparse.ArgumentParser(
        prog='main',
        description='Perform bleed-through correction on an image collection.',
    )

    """ Define the arguments """
    _parser.add_argument(
        '--inpDir',
        dest='inpDir',
        type=str,
        help='Path to input images.',
        required=True,
    )

    _parser.add_argument(
        '--filePattern',
        dest='filePattern',
        type=str,
        help='Input file name pattern.',
        required=True,
    )

    _parser.add_argument(
        '--groupBy',
        dest='groupBy',
        type=str,
        help='Which variables to use when grouping images. '
             'Each group should contain all tiles and all channels in one round of imaging.',
        required=True,
    )

    _parser.add_argument(
        '--tileSelectionCriterion',
        dest='tileSelectionCriterion',
        type=str,
        help='What method to use for selecting tiles. These tiles will be used to estimate bleed-through.',
        required=False,
        default='HighMeanIntensity',
    )

    _parser.add_argument(
        '--model',
        dest='model',
        type=str,
        help='Which model to train for estimating bleed-through.',
        required=False,
        default='Lasso',
    )

    _parser.add_argument(
        '--channelOverlap',
        dest='channelOverlap',
        type=int,
        help='Number of adjacent channels to consider for estimating bleed-through.',
        required=False,
        default=1,
    )

    _parser.add_argument(
        '--computeComponents',
        dest='computeComponents',
        type=str,
        help='Whether to compute and write the bleed-through component for each image. '
             'A component can be subtracted from the corresponding image to remove bleed-through.',
        required=False,
        default='true',
    )

    _parser.add_argument(
        '--outDir',
        dest='outDir',
        type=str,
        help='Output directory for the processed images and metadata.',
        required=True,
    )

    _args = _parser.parse_args()

    _input_dir = Path(_args.inpDir).resolve()
    if _input_dir.joinpath('images').is_dir():
        _input_dir = _input_dir.joinpath('images')
    logger.info(f'inpDir = {_input_dir}')

    _pattern = _args.filePattern
    logger.info(f'filePattern = {_pattern}')

    _group_by = _args.groupBy
    if 'c' not in _group_by:
        _message = f'Grouping Variables must contain \'c\'. Got {_group_by} instead.'
        logger.error(_message)
        raise ValueError(_message)
    logger.info(f'groupBy = {_group_by}')

    _selector_name = _args.tileSelectionCriterion
    if _selector_name not in SELECTORS.keys():
        _message = f'--tileSelectionCriterion {_selector_name} not found. Must be one of {list(SELECTORS.keys())}.'
        logger.error(_message)
        raise ValueError(_message)
    logger.info(f'tileSelectionCriterion = {_selector_name}')

    _model_name = _args.model
    if _model_name not in MODELS.keys():
        _message = f'--model {_model_name} not found. Must be one of {list(MODELS.keys())}.'
        logger.error(_message)
        raise ValueError(_message)
    logger.info(f'model = {_model_name}')

    _channel_overlap = _args.channelOverlap
    logger.info(f'channelOverlap = {_channel_overlap}')

    _compute_components = _args.computeComponents
    if _compute_components not in ('true', 'false'):
        _message = f'The value {_compute_components} for --computeComponents is not valid. Must be either \'true\' or \'false\'.'
        logger.error(_message)
        raise ValueError(_message)
    _compute_components = (_compute_components == 'true')
    logger.info(f'computeComponents = {_compute_components}')

    _output_dir = Path(_args.outDir).resolve()
    _metadata_dir = _output_dir.joinpath('metadata')
    _metadata_dir.mkdir(parents=False, exist_ok=True)
    _output_dir = _output_dir.joinpath('images')
    _output_dir.mkdir(parents=False, exist_ok=True)

    logger.info(f'outDir = {_output_dir}')
    logger.info(f'metadataDir = {_metadata_dir}')

    _fp = FilePattern(_input_dir, _pattern)
    for _group in _fp(list(_group_by)):
        estimate_bleed_through(
            group=_group,
            pattern=_pattern,
            selector_name=_selector_name,
            model_name=_model_name,
            channel_overlap=_channel_overlap,
            output_dir=_output_dir if _compute_components else None,
            metadata_dir=_metadata_dir,
        )
