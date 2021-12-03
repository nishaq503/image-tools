import logging
import json, os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Tuple
import csv

import segmentation_models_pytorch as smp
import torch
from filepattern import FilePattern
from segmentation_models_pytorch.base import SegmentationModel
from segmentation_models_pytorch.utils.base import Metric
from segmentation_models_pytorch.utils.train import TrainEpoch
from segmentation_models_pytorch.utils.train import ValidEpoch
from sklearn.model_selection import train_test_split
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss as TorchLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader as TorchDataLoader

import albumentations as albu

import utils

logging.basicConfig(
    format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
)
logger = logging.getLogger("training")
logger.setLevel(utils.POLUS_LOG)


def initialize_model(checkpoint: Dict[str, Any],
                     device: Any) -> Tuple[SegmentationModel, Optimizer, int]:
    """ Initializes a model from a Checkpoint. A checkpoint knows the:

        * 'model_name': The architecture of the model in use.
            See utils.params.MODELS
        * 'encoder_variant': The name of the specific encoder architecture.
            See utils.params.ENCODERS
        * 'encoder_weights': The name of the dataset used to pre-train the
            encoder. See utils.params.ENCODERS
        * 'optimizer_name': The name of the optimization algorithm for training
            the model. See utils.params.OPTIMIZERS
        * 'final_epoch': The number of epochs for which the model has been
            trained by this plugin. 0 indicates a new model.
        * 'model_state_dict': Model state from a previous run of this plugin.
        * 'optimizer_state_dict': Optimizer state from a previous run of this
            plugin.

    Args:
        checkpoint: A Checkpoint dictionary.

    Returns:
        A 3-tuple of the:
            * Instantiated SegmentationModel,
            * Instantiated Optimizer, and
            * Epoch from which to resume training, 0 indicates a new model.

        If resuming training from a previous run of this plugin, the states of
            the model and optimizer are loaded in.
    """
    logger.info('Initializing model...')

    # noinspection PyArgumentList
    model = utils.MODELS[checkpoint['model_name']](
        encoder_name=checkpoint['encoder_variant'],
        encoder_weights=checkpoint['encoder_weights'],
        in_channels=1,  # all images in WIPP are single-channel.
        activation='sigmoid', # TODO: Change for Cellpose FlowFields
    )
    model.to(device)
    model.cuda()
    # noinspection PyArgumentList
    optimizer = utils.OPTIMIZERS[checkpoint['optimizer_name']](params=model.parameters())

    starting_epoch = checkpoint['final_epoch']
    if starting_epoch > 0:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, starting_epoch

def configure_augmentations(albumentations: list):

    transform: list = []

    albu_values = {
        "HorizontalFlip": 0.5,
        "ShiftScaleRotate": 1,
        "GaussianNoise": 0.2,
        "Perspective": 0.5,
        "RandomBrightnessContrast": 1,
        "RandomGamma": 1,
        "Sharpen": 1,
        "Blur": 1,
        "MotionBlur": 1
    }

    albu_dictionary = {
        "HorizontalFlip": albu.HorizontalFlip(p=albu_values["HorizontalFlip"]),
        "ShiftScaleRotate": albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, \
                            shift_limit=0.1, p=albu_values["ShiftScaleRotate"], border_mode=0),
        "PadIfNeeded": albu.PadIfNeeded(min_height=256, min_width=256, always_apply=True, border_mode=0),
        "RandomCrop": albu.RandomCrop(height=256, width=256, always_apply=True),
        "GaussianNoise": albu.GaussNoise(p=albu_values["GaussianNoise"]),
        "Perspective": albu.Perspective(p=albu_values["Perspective"]),
        "RandomBrightnessContrast": albu.RandomBrightnessContrast(p=albu_values["RandomBrightnessContrast"]),
        "RandomGamma":albu.RandomGamma(p=albu_values["RandomGamma"]),
        "Sharpen":albu.Sharpen(p=albu_values["Sharpen"]),
        "Blur":albu.Blur(blur_limit=3, p=albu_values["Blur"]),
        "MotionBlur":albu.MotionBlur(blur_limit=3,p=albu_values["MotionBlur"])
    }
        

    for albu_transform in albumentations:
        if albu_transform in albu_dictionary.keys():
            transform.append(albu_dictionary[albu_transform])

    return transform


def initialize_dataloaders(
        *,
        images_dir: Path,
        images_pattern: str,
        labels_dir: Path,
        labels_pattern: str,
        train_fraction: float,
        batch_size: int,
        trainAlbumentations: list,
        validAlbumentations: list,
        segmentationMode: str,
        device: Any
) -> Tuple[TorchDataLoader, TorchDataLoader]:
    """ Initializes data-loaders for training and validation from the input
        image and label collections.

    Args:
        images_dir: Input Image collection.
        images_pattern: File-pattern for input images.
        labels_dir: Labels collection for the input images.
        labels_pattern: File-pattern for the labels.
        train_fraction: Fraction of input images to use for training. Must be a
            float between 0 and 1.
        batch_size: Number of tiles per batch to use for training.

    Returns:
        A 2-tuple of the data-loaders for training and validation.
    """
    logger.info('Initializing dataloaders...')

    label_paths = utils.get_labels_mapping(
        images_fp=FilePattern(images_dir, images_pattern),
        labels_fp=FilePattern(labels_dir, labels_pattern),
    )

    train_paths, valid_paths = train_test_split(
        list(label_paths.keys()),
        train_size=train_fraction,
        shuffle=True,
    )

    train_augmentations = configure_augmentations(albumentations=trainAlbumentations)
    valid_augmentations = configure_augmentations(albumentations=validAlbumentations)

    train_dataset = utils.Dataset(
        labels_map={k: label_paths[k] for k in train_paths},
        tile_map=utils.get_tiles_mapping(train_paths),
        segmentationMode = segmentationMode,
        augmentations=train_augmentations,
        device=device
    )
    valid_dataset = utils.Dataset(
        labels_map={k: label_paths[k] for k in valid_paths},
        tile_map=utils.get_tiles_mapping(valid_paths),
        augmentations=valid_augmentations,
        segmentationMode = segmentationMode,
        device=device
    )

    # train_dataset.to(device)
    # valid_dataset.to(device)
    train_loader = TorchDataLoader(dataset=train_dataset, batch_size=batch_size)
    valid_loader = TorchDataLoader(dataset=valid_dataset, batch_size=batch_size)

    return train_loader, valid_loader


def initialize_epoch_iterators(
        *,
        model: SegmentationModel,
        loss: TorchLoss,
        metric: Metric,
        device: torch.device,
        optimizer: Optimizer,
) -> Tuple[TrainEpoch, ValidEpoch]:
    """ Initializes the training and validation iterators that train the model
        for each epoch.

    Args:
        model: The model being trained.
        loss: An instantiated Loss function for the model.
        metric: An instantiated Metric with which to track model performance.
        device: A torch device, wither a GPU or a CPU.
        optimizer: An instantiated optimizer with which to update the model.

    Returns:
        A 2-tuple of the epoch-iterators for training and validation.
    """
    logger.info('Initializing Epoch Iterators...')

    epoch_kwargs = dict(model=model, loss=loss, \
        metrics=[metric, smp.utils.metrics.Fscore(), \
            smp.utils.metrics.Accuracy(), smp.utils.metrics.Recall(), smp.utils.metrics.Precision()], \
        device=device, verbose=False)
    trainer = smp.utils.train.TrainEpoch(optimizer=optimizer, **epoch_kwargs)
    validator = smp.utils.train.ValidEpoch(**epoch_kwargs)

    return trainer, validator


def train_model(
        *,
        dataloaders: Tuple[TorchDataLoader, TorchDataLoader],
        epoch_iterators: Tuple[TrainEpoch, ValidEpoch],
        early_stopping: Tuple[int, int, float],
        starting_epoch: int,
        output_dir: str,
        model: SegmentationModel, 
        checkpoint : SegmentationModel,
        optimizer : Any,
        checkpointFreq : int,
        create_checkpointDirectory : bool,
        device : Any
) -> int:
    """ Trains the model.

    Args:
        dataloaders: A 2-tuple of data-loaders for training and validation.
        epoch_iterators: A 2-tuple of iterators for training and validation.
        early_stopping: Criteria for cutting short model training. A 3-tuple of
            * the maximum number of epochs to train the model,
            * the maximum number of epochs to wait for the model to improve.
            * the minimum decrease in loss to consider an improvement.
        starting_epoch: The index of the epoch from which we resume training.

    Returns:
        The total number of epochs for which the model has been trained by this
            plugin.
    """
    train_loader, valid_loader = dataloaders
    trainer, validator = epoch_iterators
    max_epochs, patience, min_delta = early_stopping

    if create_checkpointDirectory:
        checkpoint_dirs = os.path.join(output_dir, "checkpoints")
        if not os.path.exists(checkpoint_dirs):
            os.mkdir(checkpoint_dirs)

    with open(os.path.join(output_dir, "trainlogs.csv"), 'a') as f_train:
        with open(os.path.join(output_dir, "validlogs.csv"), 'a') as f_valid:

            best_loss, epoch, epochs_without_improvement = 1e5, starting_epoch, 0
            for epoch, _ in enumerate(range(max_epochs), start=starting_epoch + 1):
                logger.info(''.join((5 * '-', f'   Epoch: {epoch}    ', 5 * '-')))

                # trainer.to(device)
                train_logs = trainer.run(train_loader)
                train_str = ', '.join(f'{k}: {v:.8f}' for k, v in train_logs.items())
                logger.info(f'Train logs: {train_str}')
                f_train.write(train_str + "\n")

                # validator.to(device)
                valid_logs = validator.run(valid_loader)
                valid_str = ', '.join(f'{k}: {v:.8f}' for k, v in valid_logs.items())
                logger.info(f'Valid logs: {valid_str}')
                f_valid.write(valid_str + "\n")

                # check for early stopping
                current_loss = valid_logs[trainer.loss.__name__]
                if best_loss > current_loss:
                    epochs_without_improvement, best_loss = 0, current_loss
                else:
                    epochs_without_improvement += 1

                logger.info(f"BEST LOSS: {best_loss}, CURRENT LOSS: {current_loss}, MIN_DELTA: {min_delta}, " + \
                        f"Epochs without Improvement: {epochs_without_improvement}")

                if (epoch%checkpointFreq) == 0:
                    if create_checkpointDirectory:
                        torch.save(checkpoint, os.path.join(checkpoint_dirs, f"Epoch_{epoch}.path"))
                    checkpoint.update({'final_epoch': epoch,
                                       'model_state_dict': model.state_dict(),
                                       'optimizer_state_dict': optimizer.state_dict()})
                    torch.save(checkpoint, output_dir.joinpath('checkpoint.pth'))
                if epochs_without_improvement >= patience:
                    logger.info(f'No improvement for {patience} epochs. Stopping training early...')
                    break
            else:
                logger.info(f'Finished training for user-specified {max_epochs} epochs...')

    return epoch
