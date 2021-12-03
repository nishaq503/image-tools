import this
import logging, os, json
import logging.config
import argparse
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import numpy
import torch

import training
import utils

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

    parser.add_argument('--imagesTrainDir', dest='imagesDir', type=str, required=True,
                        help='Collection containing images.')
    parser.add_argument('--labelsTrainDir', dest='labelsDir', type=str, required=True,
                        help='Collection containing labels, i.e. the ground-truth, for the images.')
    parser.add_argument('--imagesValidDir', dest='imagesDir', type=str, required=True,
                        help='Collection containing images.')
    parser.add_argument('--labelsValidDir', dest='labelsDir', type=str, required=True,
                        help='Collection containing labels, i.e. the ground-truth, for the images.')
    parser.add_argument('--imagesPattern', dest='imagesPattern', type=str, required=False, default='.*',
                        help='Filename pattern for images.')
    parser.add_argument('--labelsPattern', dest='labelsPattern', type=str, required=False, default='.*',
                        help='Filename pattern for labels.')
    parser.add_argument('--trainFraction', dest='trainFraction', type=float, required=False, default=0.7,
                        help='Fraction of dataset to use for training.')
    parser.add_argument('--segmentationMode', dest='segmentationMode', type=str, required=True,
                        help='The kind of segmentation to perform.'
                             'Must be one of \'binary\', \'multilabel\', or \'multiclass\'')

    parser.add_argument('--device', dest='device', type=str, required=False, default='cpu',
                        help='Device to run process on')
    parser.add_argument('--create_checkpointDirectory', dest='create_checkpointDirectory', type=bool, required=False, default=False,
                        help="Create separate files for all the checkpoints?")
    parser.add_argument('--checkpointFrequency', dest='checkFreq', type=int, required=False, default=1,
                        help="How often to update the checkpoints")

    parser.add_argument('--trainAlbumentations', dest='trainAlbumentations', type=str, required=False, 
                        default='', help='The list of albumentations to apply to training data')
    parser.add_argument('--validAlbumentations', dest='validAlbumentations', type=str, required=False,
                        default='', help='The list of albumentations to apply to validation data')
    parser.add_argument('--lossName', dest='lossName', type=str, required=False, default='JaccardLoss',
                        help='Name of loss function to use.')
    parser.add_argument('--metricName', dest='metricName', type=str, required=False, default='IoU',
                        help='Name of performance metric to track.')
    parser.add_argument('--maxEpochs', dest='maxEpochs', type=int, required=False, default=100,
                        help='Maximum number of epochs for which to continue training the model.')
    parser.add_argument('--patience', dest='patience', type=int, required=False, default=10,
                        help='Maximum number of epochs to wait for model to improve.')
    parser.add_argument('--minDelta', dest='minDelta', type=float, required=False, default=1e-4,
                        help='Minimum improvement in loss to reset patience.')

    parser.add_argument('--outputDir', dest='outputDir', type=str, required=True,
                        help='Location where the model and the final checkpoint will be saved.')


    # Parse the arguments
    args = parser.parse_args()
    # Location to save model and checkpoint
    output_dir = Path(args.outputDir).resolve()
    assert output_dir.exists()
    
    os.environ["LOGFILE"] = os.path.join(str(output_dir), "logs.log")
    # Initialize Logging
    logging.basicConfig(filename='logs.log', \
                        encoding ='utf-8',\
                        format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s', \
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(utils.POLUS_LOG)

    
    
    """ Argument parsing """
    logger.info("Parsing arguments...")
    error_messages = list()

    # Model Configuration/Compilation
    # Output Arguments 
    loss_name = args.lossName
    metric_name = args.metricName
    max_epochs = args.maxEpochs
    patience = args.patience
    min_delta = args.minDelta
    device = args.device
    trainAlbumentations = args.trainAlbumentations
    validAlbumentations = args.validAlbumentations
    train_fraction: float = args.trainFraction
    segmentation_mode = args.segmentationMode
    batch_size = args.batchSize
    images_dir = Path(args.imagesDir).resolve()
    labels_dir = Path(args.labelsDir).resolve()
    images_pattern: str = args.imagesPattern
    labels_pattern: str = args.labelsPattern
    create_checkpointDirectory: bool = args.create_checkpointDirectory


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
        config_dict = checkpoint
        config_dict["imagesDir"] = str(images_dir)
        config_dict["imagesPattern"] = images_pattern
        config_dict["labelsDir"] = str(labels_dir)
        config_dict["labelsPattern"] = labels_pattern
        config_dict["trainFraction"] = train_fraction
        config_dict["segmentationMode"] = segmentation_mode
        config_dict["batchSize"] = batch_size
        config_dict["loss_name"] = loss_name
        config_dict["metric_name"] = metric_name
        config_dict["patience"] = patience
        config_dict["min_delta"] = min_delta
        config_dict["device"] = device
        config_dict["trainAlbumentations"] = trainAlbumentations
        config_dict["validAlbumentations"] = validAlbumentations
        config_dict["override_checkpoint"] = create_checkpointDirectory
        print(config_dict)
        with open(config_path, 'w') as config_file:
            json.dump(config_dict, config_file)
        
    else:
        encoder_base = None
        pretrained_model = Path(pretrained_model).resolve()
        checkpoint = torch.load(pretrained_model.joinpath('checkpoint.pth').resolve())

        if os.path.exists(config_path):
            config_dict = json.load(config_path)
    
    trainAlbumentations = trainAlbumentations.split("_")
    validAlbumentations = validAlbumentations.split("_")
    
    # Dataset
    if images_dir.joinpath('images').is_dir():
        images_dir = images_dir.joinpath('images')
    assert images_dir.exists()

    if labels_dir.joinpath('labels').is_dir():
        labels_dir = labels_dir.joinpath('images')
    assert labels_dir.exists()

    if segmentation_mode not in ('binary', 'multilabel', 'multiclass'):
        error_messages.append(
            f'segmentationMode must be one of \'binary\', \'multilabel\', \'multiclass\'. '
            f'Got {segmentation_mode} instead.'
        )

    # Error catching on input params
    if not 0 < train_fraction < 1:
        error_messages.append(
            f'trainFraction must be a fraction between 0 and 1. '
            f'Got {train_fraction} instead.'
        )

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

    if metric_name not in utils.METRICS:
        error_messages.append(
            f'metricName must be one of {list(utils.METRICS.keys())}. '
            f'Got {metric_name} instead.'
        )

    if len(error_messages) > 0:
        error_messages = ['Oh no! Something went wrong'] + error_messages + ['See the README for details.']
        error_message = '\n'.join(error_messages)
        logger.error(error_message)
        raise ValueError(error_message)

    # log all input arguments
    logger.info(f'pretrainedModel = {pretrained_model}')
    logger.info(f'modelName = {checkpoint["model_name"]}')
    logger.info(f'encoderBase = {encoder_base}')
    logger.info(f'encoderVariant = {checkpoint["encoder_variant"]}')
    logger.info(f'encoderWeights = {checkpoint["encoder_weights"]}')
    logger.info(f'optimizerName = {checkpoint["optimizer_name"]}')

    logger.info(f'batchSize = {batch_size}')

    logger.info(f'imagesDir = {images_dir}')
    logger.info(f'imagesPattern = {images_pattern}')
    logger.info(f'labelsDir = {labels_dir}')
    logger.info(f'labelsPattern = {labels_pattern}')
    logger.info(f'trainFraction = {train_fraction}')
    logger.info(f'segmentationMode = {segmentation_mode}')

    logger.info(f'lossName = {loss_name}')
    logger.info(f'metricName = {metric_name}')
    logger.info(f'maxEpochs = {max_epochs}')
    logger.info(f'patience = {patience}')
    logger.info(f'minDelta = {min_delta}')
    logger.info(f'create_checkpointDirectory = {create_checkpointDirectory}')

    logger.info(f'outputDir = {output_dir}')

    # TODO: Add support for multiple GPUs
    if 'cuda' in device:
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}...')

    device = torch.device(device)
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

    dataloaders = training.initialize_dataloaders(
        images_dir=images_dir,
        images_pattern=images_pattern,
        labels_dir=labels_dir,
        labels_pattern=labels_pattern,
        train_fraction=train_fraction,
        batch_size=batch_size,
        trainAlbumentations = trainAlbumentations,
        validAlbumentations = validAlbumentations,
        segmentationMode = segmentation_mode,
        device=device
    )

    # TODO: segmentation_mode 'multiclass' is broken on some datasets. Investigate why.
    loss = utils.LOSSES[loss_name](segmentation_mode)
    loss.__name__ = loss_name
    epoch_iterators = training.initialize_epoch_iterators(
        model=model,
        loss=loss,
        metric=utils.METRICS[metric_name](),
        device='cuda',
        optimizer=optimizer,
    )

    if not os.path.exists(os.path.join(output_dir, "trainlogs.csv")):
        f = open(os.path.join(output_dir, "trainlogs.csv"), 'w+')
        f.close()
    if not os.path.exists(os.path.join(output_dir, "validlogs.csv")):
        f = open(os.path.join(output_dir, "validlogs.csv"), 'w+')
        f.close()

    checkFreq = args.checkFreq
    final_epoch = training.train_model(
        dataloaders=dataloaders,
        epoch_iterators=epoch_iterators,
        early_stopping=(max_epochs, patience, min_delta),
        starting_epoch=starting_epoch,
        output_dir=output_dir,
        model=model,
        checkpoint = checkpoint,
        optimizer=optimizer,
        checkpointFreq = checkFreq,
        create_checkpointDirectory = create_checkpointDirectory,
        device = device
    )

    logger.info('Saving model...')
    torch.save(model, output_dir.joinpath('model.pth'))

    logger.info('Saving checkpoint...')
    checkpoint.update({
        'final_epoch': final_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    })
    torch.save(checkpoint, output_dir.joinpath('checkpoint.pth'))
