#!/usr/bin/env python3

import re
from pathlib import Path
import importlib
import yaml
import jsonschema
import torch
from torch import nn


_ENCODER_KWARGS_SCHEMA = {
    'type': 'object',
    'required': [
        'dimension',
        'num_heads',
        'dim_feedforward',
        'num_layers',
        'activation_function',
        'dropout',
        'checkpointing',
    ],
    'properties': {
        'dimension': {
            'type': 'integer',
            'minimum': 1,
        },
        'num_heads': {
            'type': 'integer',
            'minimum': 1,
        },
        'dim_feedforward': {
            'type': 'integer',
            'minimum': 1,
        },
        'num_layers': {
            'type': 'integer',
            'minimum': 1,
        },
        'activation_function': {
            'type': 'string',
            'enum': [
                'relu',
                'gelu',
            ],
        },
        'dropout': {
            'type': 'number',
            'minimum': 0.0,
            'exclusiveMaximum': 1.0,
        },
        'checkpointing': {
            'type': 'boolean',
        },
    },
    'additionalProperties': False,
}


_DECODER_KWARGS_SCHEMA = {
    'type': 'object',
    'required': [
        'dimension',
        'dim_final_feedforward',
        'activation_function',
        'dropout',
    ],
    'properties': {
        'dimension': {
            'type': 'integer',
            'minimum': 1,
        },
        'dim_final_feedforward': {
            'type': 'integer',
            'minimum': 1,
        },
        'activation_function': {
            'type': 'string',
            'enum': [
                'relu',
                'gelu',
            ],
        },
        'dropout': {
            'type': 'number',
            'minimum': 0.0,
            'exclusiveMaximum': 1.0,
        },
    },
    'additionalProperties': False
}


_ENCODER_DECODER_SNAPSHOTS_CONFIG_SCHEMA = {
    'type': 'object',
    'required': [
        'encoder',
        'decoder',
        'model',
    ],
    'properties': {
        'encoder': {
            'type': 'object',
            'required': [
                'module',
                'class',
                'kwargs',
                'snapshot',
            ],
            'properties': {
                'module': {
                    'type': 'string',
                },
                'class': {
                    'type': 'string',
                },
                'kwargs': _ENCODER_KWARGS_SCHEMA,
                'snapshot': {
                    'type': 'string',
                },
            },
            'additionalProperties': False,
        },
        'decoder': {
            'type': 'object',
            'required': [
                'module',
                'class',
                'kwargs',
                'snapshot',
            ],
            'properties': {
                'module': {
                    'type': 'string',
                },
                'class': {
                    'type': 'string',
                },
                'kwargs': _DECODER_KWARGS_SCHEMA,
                'snapshot': {
                    'type': 'string',
                },
            },
            'additionalProperties': False,
        },
        'model': {
            'type': 'object',
            'required': [
                'module',
                'class',
            ],
            'properties': {
                'module': {
                    'type': 'string',
                },
                'class': {
                    'type': 'string',
                },
            },
            'additionalProperties': False,
        },
    },
    'additionalProperties': False,
}


_MODEL_SNAPSHOT_CONFIG_SCHEMA = {
    'type': 'object',
    'required': [
        'encoder',
        'decoder',
        'model',
    ],
    'properties': {
        'encoder': {
            'type': 'object',
            'required': [
                'module',
                'class',
                'kwargs',
            ],
            'properties': {
                'module': {
                    'type': 'string',
                },
                'class': {
                    'type': 'string',
                },
                'kwargs': _ENCODER_KWARGS_SCHEMA,
            },
            'additionalProperties': False,
        },
        'decoder': {
            'type': 'object',
            'required': [
                'module',
                'class',
                'kwargs',
            ],
            'properties': {
                'module': {
                    'type': 'string',
                },
                'class': {
                    'type': 'string',
                },
                'kwargs': _DECODER_KWARGS_SCHEMA,
            },
            'additionalProperties': False,
        },
        'model': {
            'type': 'object',
            'required': [
                'module',
                'class',
                'snapshot',
            ],
            'properties': {
                'module': {
                    'type': 'string',
                },
                'class': {
                    'type': 'string',
                },
                'snapshot': {
                    'type': 'string',
                },
            },
            'additionalProperties': False,
        },
    },
    'additionalProperties': False,
}


_MODEL_CONFIG_SCHEMA = {
    'oneOf': [
        _ENCODER_DECODER_SNAPSHOTS_CONFIG_SCHEMA,
        _MODEL_SNAPSHOT_CONFIG_SCHEMA
    ]
}


def load_model(config_path: Path) -> nn.Module:
    if not config_path.exists():
        raise RuntimeError(f'{config_path}: Does not exist.')
    if not config_path.is_file():
        raise RuntimeError(f'{config_path}: Not a file.')

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    jsonschema.validate(config, _MODEL_CONFIG_SCHEMA)

    this_dir = config_path.parent

    encoder_config = config['encoder']
    encoder_module = importlib.import_module(encoder_config['module'])
    encoder_class = getattr(encoder_module, encoder_config['class'])
    encoder_instance = encoder_class(**encoder_config['kwargs'])

    decoder_config = config['decoder']
    decoder_module = importlib.import_module(decoder_config['module'])
    decoder_class = getattr(decoder_module, decoder_config['class'])
    decoder_instance = decoder_class(**decoder_config['kwargs'])

    model_config = config['model']
    model_module = importlib.import_module(model_config['module'])
    model_class = getattr(model_module, model_config['class'])
    model = model_class(encoder_instance, decoder_instance)

    if 'snapshot' in encoder_config:
        assert('snapshot' in decoder_config)
        encoder_config['snapshot'] = re.sub(
            '^\\./', str(this_dir) + '/', encoder_config['snapshot'])
        decoder_config['snapshot'] = re.sub(
            '^\\./', str(this_dir) + '/', decoder_config['snapshot'])
        encoder_state_dict = torch.load(encoder_config['snapshot'])
        encoder_state_dict_fixed = {}
        for k, v in encoder_state_dict.items():
            k = re.sub('^module\\.', '', k)
            k = re.sub('^_.+?__', '', k)
            k = re.sub('\\._.+?__', '.', k)
            encoder_state_dict_fixed[k] = v
        encoder_instance.load_state_dict(encoder_state_dict_fixed)
        decoder_state_dict = torch.load(decoder_config['snapshot'])
        decoder_state_dict_fixed = {}
        for k, v in decoder_state_dict.items():
            k = re.sub('^module\\.', '', k)
            k = re.sub('^_.+?__', '', k)
            k = re.sub('\\._.+?__', '.', k)
            decoder_state_dict_fixed[k] = v
        decoder_instance.load_state_dict(decoder_state_dict_fixed)
    else:
        assert('snapshot' in model_config)
        model_config['snapshot'] = re.sub(
            '^\\./', str(this_dir) + '/', model_config['snapshot'])
        model_state_dict = torch.load(model_config['snapshot'])
        model_state_dict_fixed = {}
        for k, v in model_state_dict.items():
            k = re.sub('^module\\.', '', k)
            k = re.sub('^_.+?__', '', k)
            k = re.sub('\\._.+?__', '.', k)
            model_state_dict_fixed[k] = v
        model.load_state_dict(model_state_dict_fixed)

    return model
