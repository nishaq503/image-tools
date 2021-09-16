import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path

from filepattern import FilePattern

import utils
from models import MODELS
from tile_selectors import SELECTORS

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger('main')
logger.setLevel(utils.POLUS_LOG)


def estimate_bleed_through(
        *,
        group: list[utils.FPFileDict],
        pattern: str,
        selector_name: str,
        model_name: str,
        channel_overlap: int,
        kernel_size: int,
        output_dir: Path,
        metadata_dir: Path,
):
    """ Estimates the bleed-through across adjacent channels among a group of files.

    Args:
        group: A filepattern group containing all tiles and all channels in one round of imaging.
        pattern: The filepattern used for selecting the group.
        selector_name: The method to use for selecting tiles. See `tile_selectors.py`
        model_name: The model to train for estimating bleed-through coefficients. See `models.py`
        channel_overlap: The number of adjacent channels that could cause bleed-through.
        kernel_size: Size of convolutional kernel to use for bleed through.
        output_dir: If a Path is passed, bleed-through components will be saved in this directory.
        metadata_dir: The bleed-through coefficients for each round will be saved in this directory.
    """
    files = [file['file'] for file in group]

    logger.info('selecting tiles...')
    selector = SELECTORS[selector_name](files, num_tiles_per_channel=10)

    logger.info('training models...')
    model = MODELS[model_name](
        files=files,
        selected_tiles=selector.selected_tiles,
        image_mins=selector.image_mins,
        image_maxs=selector.image_maxs,
        channel_overlap=channel_overlap,
        kernel_size=kernel_size,
    )

    logger.info('exporting coefficients...')
    model.coefficients_to_csv(metadata_dir, pattern, group)

    logger.info('writing bleed-through components...')
    model.write_components(output_dir)

    return


if __name__ == '__main__':
    logger.info("Parsing arguments...")
    _parser = argparse.ArgumentParser(
        prog='main',
        description='Perform bleed-through correction on an image collection.',
    )

    """ Define the arguments """
    _parser.add_argument(
        '--inpDir', dest='inpDir', type=str, required=True,
        help='Path to input images.',
    )

    _parser.add_argument(
        '--filePattern', dest='filePattern', type=str, required=True,
        help='Input file name pattern.',
    )

    _parser.add_argument(
        '--groupBy', dest='groupBy', type=str, required=True,
        help='Which variables to use when grouping images. '
             'Each group should contain all tiles and all channels in one round of imaging.',
    )

    _parser.add_argument(
        '--selectionCriterion', dest='selectionCriterion', type=str, required=False, default='HighMeanIntensity',
        help='What method to use for selecting tiles. These tiles will be used to estimate bleed-through.',
    )

    _parser.add_argument(
        '--model', dest='model', type=str, required=False, default='Lasso',
        help='Which model to train for estimating bleed-through.',
    )

    _parser.add_argument(
        '--channelOverlap', dest='channelOverlap', type=int, required=False, default=1,
        help='Number of adjacent channels to consider for estimating bleed-through.',
    )

    _parser.add_argument(
        '--kernelSize', dest='kernelSize', type=str, required=False, default='3x3',
        help='Size of convolutional kernel to learn.',
    )

    _parser.add_argument(
        '--outDir', dest='outDir', type=str, required=True,
        help='Output directory for the bleed-through components.',
    )

    _parser.add_argument(
        '--csvDir', dest='csvDir', type=str, required=True,
        help='Output directory for the coefficients of the learned kernels.',
    )

    _args = _parser.parse_args()
    _error_messages = list()

    _input_dir = Path(_args.inpDir).resolve()
    assert _input_dir.exists()
    if _input_dir.joinpath('images').is_dir():
        _input_dir = _input_dir.joinpath('images')

    _pattern = _args.filePattern

    _group_by = _args.groupBy
    if 'c' not in _group_by:
        _error_messages.append(f'Grouping Variables must contain \'c\'. Got {_group_by} instead.')

    _selector_name = _args.selectionCriterion
    if _selector_name not in SELECTORS.keys():
        _error_messages.append(f'--tileSelectionCriterion {_selector_name} not found. '
                               f'Must be one of {list(SELECTORS.keys())}.')

    _model_name = _args.model
    if _model_name not in MODELS.keys():
        _error_messages.append(f'--model {_model_name} not found. Must be one of {list(MODELS.keys())}.')

    _channel_overlap = _args.channelOverlap

    _kernel_size = _args.kernelSize
    if _kernel_size not in ('1x1', '3x3', '5x5'):
        _error_messages.append(f'--kernelSize must be one of \'1x1\', \'3x3\', \'5x5\'. '
                               f'Got {_kernel_size} instead.')
    _kernel_size = int(_kernel_size.split('x')[0])

    _output_dir = Path(_args.outDir).resolve()
    assert _output_dir.exists()
    _csv_dir = Path(_args.csvDir).resolve()
    assert _csv_dir.exists()

    if len(_error_messages) > 0:
        _message = '\n'.join((
            'Oh no! Something went wrong:',
            *_error_messages,
            'See the README for more details.',
        ))
        logger.error(_message)
        raise ValueError(_message)

    logger.info(f'inpDir = {_input_dir}')
    logger.info(f'filePattern = {_pattern}')
    logger.info(f'groupBy = {_group_by}')
    logger.info(f'selectionCriterion = {_selector_name}')
    logger.info(f'model = {_model_name}')
    logger.info(f'channelOverlap = {_channel_overlap}')
    logger.info(f'kernelSize = {_kernel_size}x{_kernel_size}')
    logger.info(f'outDir = {_output_dir}')
    logger.info(f'csvDir = {_csv_dir}')

    _fp = FilePattern(_input_dir, _pattern)
    _kwargs = dict(
        pattern=_pattern,
        selector_name=_selector_name,
        model_name=_model_name,
        channel_overlap=_channel_overlap,
        kernel_size=_kernel_size,
        output_dir=_output_dir,
        metadata_dir=_csv_dir,
    )
    with ProcessPoolExecutor(max_workers=utils.NUM_THREADS) as _executor:
        _processes = list()
        for _group in _fp(list(_group_by)):
            _kwargs['group'] = _group
            _processes.append(_executor.submit(estimate_bleed_through, **_kwargs))

        for _process in as_completed(_processes):
            _process.result()
