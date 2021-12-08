import argparse
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import numpy
import torch

import training
import utils

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("main")
logger.setLevel(utils.POLUS_LOG)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='main', description='Segmentation models training plugin')

    # Input arguments
    parser.add_argument('--pretrainedModel', dest='pretrainedModel', type=str, required=False,
                        help='Path to a model that was previously trained with this plugin. '
                             'If starting fresh, you must provide: '
                             '\'modelName\', '
                             '\'encoderBaseVariantWeights\', and'
                             '\'optimizerName\'.'
                             'See the README for available options.')

    parser.add_argument('--modelName', dest='modelName', type=str, required=False, default='Unet',
                        help='Which model architecture to use.')

    parser.add_argument('--encoderBase', dest='encoderBase', type=str, required=False, default='ResNet',
                        help='Base encoder to use.')
    parser.add_argument('--encoderVariant', dest='encoderVariant', type=str, required=False, default='resnet34',
                        help='Encoder variant to use.')
    parser.add_argument('--encoderWeights', dest='encoderWeights', type=str, required=False,
                        help='Name of dataset with which the model was pretrained.')

    parser.add_argument('--optimizerName', dest='optimizerName', type=str, required=False, default='Adam',
                        help='Name of optimization algorithm to use for training the model.')

    parser.add_argument('--batchSize', dest='batchSize', type=int, required=False,
                        help='Size of each batch for training. If left unspecified, we will automatically use '
                             'the largest possible size based on the model architecture and GPU memory.')

    parser.add_argument('--imagesTrainDir', dest='imagesTrainDir', type=str, required=True,
                        help='Collection containing images.')
    parser.add_argument('--labelsTrainDir', dest='labelsTrainDir', type=str, required=True,
                        help='Collection containing labels, i.e. the ground-truth, for the images.')
    parser.add_argument('--trainPattern', dest='trainPattern', type=str, required=False, default='.*',
                        help='Filename pattern for images.')

    parser.add_argument('--imagesValidDir', dest='imagesValidDir', type=str, required=True,
                        help='Collection containing images.')
    parser.add_argument('--labelsValidDir', dest='labelsValidDir', type=str, required=True,
                        help='Collection containing labels, i.e. the ground-truth, for the images.')
    parser.add_argument('--validPattern', dest='validPattern', type=str, required=False, default='.*',
                        help='Filename pattern for images.')

    parser.add_argument('--device', dest='device', type=str, required=False, default='cpu',
                        help='Device to run process on')
    parser.add_argument('--checkpointFrequency', dest='checkFreq', type=int, required=False, default=1,
                        help="How often to update the checkpoints")

    parser.add_argument('--lossName', dest='lossName', type=str, required=False, default='JaccardLoss',
                        help='Name of loss function to use.')
    # TODO: Handle downstream consequences of removing this
    # parser.add_argument('--metricName', dest='metricName', type=str, required=False, default='IoU',
    #                     help='Name of performance metric to track.')
    parser.add_argument('--maxEpochs', dest='maxEpochs', type=int, required=False, default=100,
                        help='Maximum number of epochs for which to continue training the model.')
    parser.add_argument('--patience', dest='patience', type=int, required=False, default=10,
                        help='Maximum number of epochs to wait for model to improve.')
    parser.add_argument('--minDelta', dest='minDelta', type=float, required=False,
                        help='Minimum improvement in loss to reset patience.')

    parser.add_argument('--outputDir', dest='outputDir', type=str, required=True,
                        help='Location where the model and the final checkpoint will be saved.')

    # Parse the arguments
    args = parser.parse_args()

    # Location to save model and checkpoint
    output_dir = Path(args.outputDir).resolve()
    assert output_dir.exists()

    """ Argument parsing """
    logger.info("Parsing arguments...")
    error_messages = list()

    # Model Configuration/Compilation
    # Input Arguments
    loss_name = args.lossName
    max_epochs = args.maxEpochs
    patience = args.patience
    min_delta = args.minDelta

    device = args.device
    checkpoint_frequency = args.checkFreq

    batch_size = args.batchSize

    images_train_dir = Path(args.imagesTrainDir).resolve()
    labels_train_dir = Path(args.labelsTrainDir).resolve()
    train_pattern: str = args.trainPattern

    images_valid_dir = Path(args.imagesValidDir).resolve()
    labels_valid_dir = Path(args.labelsValidDir).resolve()
    valid_pattern: str = args.validPattern

    config_path = os.path.join(output_dir, "config.json")

    # Model Creation/Specification via checkpoint dictionary
    pretrained_model: Optional[Path] = args.pretrainedModel
    if pretrained_model is None:

        encoder_base = args.encoderBase
        encoder_variant = args.encoderVariant
        encoder_weights = args.encoderWeights
        if encoder_weights == 'random':
            encoder_weights = None

        checkpoint: Dict[str, Any] = {
            'model_name': args.modelName,
            'encoder_variant': encoder_variant,
            'encoder_weights': encoder_weights,
            'optimizer_name': args.optimizerName,
            'final_epoch': 0,
            'model_state_dict': None,
            'optimizer_state_dict': None
        }
        with open(config_path, 'w') as config_file:
            json.dump(args.__dict__, config_file)

    else:
        encoder_base = None
        pretrained_model = Path(pretrained_model).resolve()
        checkpoint = torch.load(pretrained_model.joinpath('checkpoint.pth').resolve())

        if os.path.exists(config_path):
            json_obj = open(config_path, 'r')
            config_dict = json.load(json_obj)

    # Dataset
    if images_train_dir.joinpath('images').is_dir():
        images_train_dir = images_train_dir.joinpath('images')
    assert images_train_dir.exists()

    if labels_train_dir.joinpath('labels').is_dir():
        labels_train_dir = labels_train_dir.joinpath('labels')
    assert labels_train_dir.exists()

    if checkpoint["model_name"] not in utils.MODELS:
        error_messages.append(
            f'modelName must be one of {list(utils.MODELS.keys())}. '
            f'Got {checkpoint["model_name"]} instead.'
        )

    if encoder_base is not None:
        if encoder_base not in utils.ENCODERS:
            error_messages.append(
                f'encoderBase must be one of {list(utils.ENCODERS.keys())}. '
                f'Got {encoder_base} instead.'
            )
        else:
            available_variants = utils.ENCODERS[encoder_base]
            if checkpoint["encoder_variant"] not in available_variants:
                error_messages.append(
                    f'encoderVariant for {encoder_base} must be one of {list(available_variants.keys())}. '
                    f'Got {checkpoint["encoder_variant"]} instead.'
                )
            else:
                available_weights = available_variants[checkpoint["encoder_variant"]]

                if (
                        (checkpoint["encoder_weights"] is not None) and
                        (checkpoint["encoder_weights"] not in available_weights)
                ):
                    error_messages.append(
                        f'encoderWeights for {checkpoint["encoder_variant"]} must be one of {available_weights}. '
                        f'Got {checkpoint["encoder_weights"]} instead.'
                    )

    if checkpoint["optimizer_name"] not in utils.OPTIMIZERS:
        error_messages.append(
            f'optimizerName must be one of {list(utils.OPTIMIZERS.keys())}. '
            f'Got {checkpoint["optimizer_name"]} instead.'
        )

    if loss_name not in utils.LOSSES:
        error_messages.append(
            f'lossName must be one of {list(utils.LOSSES.keys())}. '
            f'Got {loss_name} instead.\n'
        )

    if len(error_messages) > 0:
        error_messages = ['Oh no! Something went wrong'] + error_messages + ['See the README for details.']
        error_message = '\n'.join(error_messages)
        logger.error(error_message)
        raise ValueError(error_message)

    logger.info(f'Using input arguments:')
    for arg in sorted(list(args.__dict__.keys())):
        logger.info(f'\t{arg} = {args.__dict__[arg]}')

    # TODO: Add support for multiple GPUs
    device: str = device if torch.cuda.is_available() else 'cpu'
    device: torch.device = torch.device(device if 'cuda' in device else 'cpu')
    logger.info(f'Using device: {device}...')

    model, optimizer, starting_epoch = training.initialize_model(checkpoint, device)

    logger.info('Determining maximum possible batch size...')
    num_trainable_params = (utils.TILE_STRIDE ** 2) + sum(
        numpy.prod(param.size())
        for name, param in model.named_parameters()
        if param.requires_grad
    )
    free_memory = utils.get_device_memory(device)
    logger.info(f'found {free_memory} bytes of free memory on device {device}...')
    max_batch_size = int(max(1, free_memory // (2 * 8 * num_trainable_params)))

    batch_size = max_batch_size if batch_size is None else min(batch_size, max_batch_size)
    logger.info(f'Using batch size: {batch_size}...')

    train_loader = training.initialize_dataloader(
        images_dir=images_train_dir,
        labels_dir=labels_train_dir,
        pattern=train_pattern,
        batch_size=batch_size,
    )
    valid_loader = training.initialize_dataloader(
        images_dir=images_valid_dir,
        labels_dir=labels_valid_dir,
        pattern=valid_pattern,
        batch_size=batch_size,
    )

    loss_class = utils.LOSSES[loss_name]
    loss_params = inspect.signature(loss_class.__init__).parameters
    loss_kwargs = dict()
    if 'mode' in loss_params:
        loss_kwargs['mode'] = 'binary'
    elif 'smooth_factor' in loss_params:
        loss_kwargs['smooth_factor'] = 0.1

    loss = loss_class(**loss_kwargs)
    loss.__name__ = loss_name
    epoch_iterators = training.initialize_epoch_iterators(
        model=model,
        loss=loss,
        metrics=list(metric() for metric in utils.METRICS.values()),
        device=device,
        optimizer=optimizer,
    )

    if not os.path.exists(os.path.join(output_dir, "trainlogs.csv")):
        f = open(os.path.join(output_dir, "trainlogs.csv"), 'w+')
        f.close()
    if not os.path.exists(os.path.join(output_dir, "validlogs.csv")):
        f = open(os.path.join(output_dir, "validlogs.csv"), 'w+')
        f.close()

    final_epoch = training.train_model(
        dataloaders=(train_loader, valid_loader),
        epoch_iterators=epoch_iterators,
        early_stopping=(max_epochs, patience, min_delta),
        starting_epoch=starting_epoch,
        output_dir=output_dir,
        checkpoint=checkpoint,
        checkpoint_frequency=checkpoint_frequency,
    )
