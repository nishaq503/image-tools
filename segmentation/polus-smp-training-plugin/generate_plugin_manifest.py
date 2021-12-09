import json

from src import utils

INPUTS = {
    'pretrainedModel': {
        'description': (
            'Path to a model that was previously trained with this plugin. '
            'If starting fresh, you must instead provide: '
            '\'modelName\', '
            '\'encoderBaseVariantWeights\', and '
            '\'optimizerName\'. '
            'See the README for available options.'
        ),
        'type': 'genericData',
        'required': False,
    },
    'modelName': {
        'description': 'Model architecture to use. Required if starting fresh.',
        'type': 'enum',
        'required': False,
        'options': {'values': utils.MODEL_NAMES},
    },
    'encoderBase': {
        'description': 'The name of the base encoder to use.',
        'type': 'enum',
        'required': False,
        'options': {'values': utils.BASE_ENCODERS},
    },
    'encoderVariant': {
        'description': 'The name of the specific variant to use.',
        'type': 'enum',
        'required': False,
        'options': {'values': utils.ENCODER_VARIANTS},
    },
    'encoderWeights': {
        'description': 'The name of the pretrained weights to use.',
        'type': 'enum',
        'required': False,
        'options': {'values': list(sorted(utils.ENCODER_WEIGHTS))},
    },
    'optimizerName': {
        'description': (
            'Name of optimization algorithm to use for training the model. '
            'Required if starting fresh.'
        ),
        'type': 'enum',
        'required': False,
        'options': {'values': utils.OPTIMIZER_NAMES},
    },

    'batchSize': {
        'description': (
            'Size of each batch for training. '
            'If left unspecified, we use the maximum possible based on memory constraints.'
        ),
        'type': 'number',
        'required': False,
    },

    'imagesTrainDir': {
        'description': 'Collection containing images to use for training.',
        'type': 'collection',
        'required': True,
    },
    'labelsTrainDir': {
        'description': 'Collection containing labels, i.e. the ground-truth, for the training images.',
        'type': 'collection',
        'required': True,
    },
    'trainPattern': {
        'description': 'Filename pattern for training images and labels.',
        'type': 'string',
        'required': False,
    },

    'imagesValidDir': {
        'description': 'Collection containing images to use for validation.',
        'type': 'collection',
        'required': True,
    },
    'labelsValidDir': {
        'description': 'Collection containing labels, i.e. the ground-truth, for the validation images.',
        'type': 'collection',
        'required': True,
    },
    'validPattern': {
        'description': 'Filename pattern for validation images and labels.',
        'type': 'string',
        'required': False,
    },

    'device': {
        'description': 'Which device to use for the model',
        'type': 'string',
        'required': False,
    },
    'checkpointFrequency': {
        'description': 'How often to save model checkpoints',
        'type': 'number',
        'required': True,
    },

    'lossName': {
        'description': 'Name of loss function to use.',
        'type': 'enum',
        'required': False,
        'options': {'values': utils.LOSS_NAMES},
    },
    'maxEpochs': {
        'description': 'Maximum number of epochs for which to continue training the model.',
        'type': 'number',
        'required': True,
    },
    'patience': {
        'description': 'Maximum number of epochs to wait for model to improve.',
        'type': 'number',
        'required': True,
    },
    'minDelta': {
        'description': 'Minimum improvement in loss to reset patience.',
        'type': 'number',
        'required': False,
    },
}

OUTPUTS = [{
    'name': 'outputDir',
    'type': 'genericData',
    'description': 'Output model and checkpoint.'
}]

DEFAULTS = {
    'modelName': 'Unet',
    'encoderBase': 'ResNet',
    'encoderVariant': 'resnet34',
    'encoderWeights': 'imagenet',
    'optimizerName': 'Adam',
    'trainPattern': '.*',
    'validPattern': '.*',
    'device': 'cuda',
    'lossName': 'JaccardLoss',
    'metricName': 'IoU',
    'minDelta': 1e-4,
}


def bump_version(debug: bool) -> str:
    with open('VERSION', 'r') as infile:
        version = infile.read()

    if debug:
        if 'debug' in version:
            [version, debug] = version.split('debug')
            version = f'{version}debug{str(1 + int(debug))}'
        else:
            version = f'{version}debug1'
    else:
        numbering = version.split('.')
        minor = int(numbering[-1])
        minor += 1
        numbering[-1] = str(minor)
        version = '.'.join(numbering)

    with open('VERSION', 'w') as outfile:
        outfile.write(version)

    return version


def create_ui():
    ui = list()

    for key, values in INPUTS.items():
        field = {
            'key': f'inputs.{key}',
            'title': key,
            'description': values['description'],
        }

        if key in DEFAULTS:
            field['default'] = DEFAULTS[key]

        ui.append(field)

    return ui


def variants_conditionals():
    validator = list()
    for base, variant in utils.ENCODERS.items():
        validator.append({
            'condition': [{
                'input': 'encoderBase',
                'value': base,
                'eval': '==',
            }],
            'then': [{
                'action': 'show',
                'input': 'encoderVariant',
                'value': list(variant.keys()),
            }]
        })
    return validator


def weights_conditionals():
    validator = list()

    for base, variants in utils.ENCODERS.items():
        for variant, weights in variants.items():
            validator.append({
                'condition': [{
                    'input': 'encoderVariant',
                    'value': variant,
                    'eval': '==',
                }],
                'then': [{
                    'action': 'show',
                    'input': 'encoderWeights',
                    'value': [*weights, 'random'],
                }],
            })

    return validator


def generate_manifest(debug: bool):
    version = bump_version(debug)
    # noinspection PyTypeChecker
    manifest = {
        'name': 'SegmentationModelsTraining',
        'version': f'{version}',
        'title': 'SegmentationModelsTraining',
        'description': 'Segmentation models training plugin',
        'author': 'Gauhar Bains (gauhar.bains@labshare.org), Najib Ishaq (najib.ishaq@axleinfo.com), Madhuri Vihani (madhuri.vihani@nih.gov)',
        'institution': 'National Center for Advancing Translational Sciences, National Institutes of Health',
        'repository': 'https://github.com/PolusAI/polus-plugins/tree/dev/segmentation',
        'website': 'https://ncats.nih.gov/preclinical/core/informatics',
        'citation': '',
        'containerId': f'labshare/polus-smp-training-plugin::{version}',
        'inputs': [{'name': key, **value} for key, value in INPUTS.items()],
        'outputs': OUTPUTS,
        'ui': create_ui(),
        'validators': variants_conditionals() + weights_conditionals()
    }

    with open('plugin.json', 'w') as outfile:
        json.dump(manifest, outfile, indent=4)

    return


if __name__ == '__main__':
    generate_manifest(debug=True)
