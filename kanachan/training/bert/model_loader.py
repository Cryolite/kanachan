#!/usr/bin/env python3

import re
from pathlib import Path
import importlib
from typing import Dict
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
        'checkpointing'
    ],
    'properties': {
        'dimension': {
            'type': 'integer',
            'minimum': 1
        },
        'num_heads': {
            'type': 'integer',
            'minimum': 1
        },
        'dim_feedforward': {
            'type': 'integer',
            'minimum': 1
        },
        'num_layers': {
            'type': 'integer',
            'minimum': 1
        },
        'activation_function': {
            'type': 'string',
            'enum': [
                'relu',
                'gelu'
            ]
        },
        'dropout': {
            'type': 'number',
            'minimum': 0.0,
            'exclusiveMaximum': 1.0
        },
        'checkpointing': {
            'type': 'boolean'
        }
    },
    'additionalProperties': False
}


_DECODER_KWARGS_SCHEMA = {
    'type': 'object',
    'required': [
        'dimension',
        'dim_final_feedforward',
        'activation_function',
        'dropout'
    ],
    'properties': {
        'dimension': {
            'type': 'integer',
            'minimum': 1
        },
        'dim_final_feedforward': {
            'type': 'integer',
            'minimum': 1
        },
        'activation_function': {
            'type': 'string',
            'enum': [
                'relu',
                'gelu'
            ]
        },
        'dropout': {
            'type': 'number',
            'minimum': 0.0,
            'exclusiveMaximum': 1.0
        }
    },
    'additionalProperties': True
}


_ENCODER_DECODER_STATE_DICT_CONFIG_SCHEMA = {
    'type': 'object',
    'required': [
        'encoder',
        'decoder',
        'model'
    ],
    'properties': {
        'encoder': {
            'type': 'object',
            'required': [
                'module',
                'class',
                'kwargs',
                'state_dict'
            ],
            'properties': {
                'module': {
                    'type': 'string'
                },
                'class': {
                    'type': 'string'
                },
                'kwargs': _ENCODER_KWARGS_SCHEMA,
                'state_dict': True
            },
            'additionalProperties': False
        },
        'decoder': {
            'type': 'object',
            'required': [
                'module',
                'class',
                'kwargs',
                'state_dict'
            ],
            'properties': {
                'module': {
                    'type': 'string'
                },
                'class': {
                    'type': 'string'
                },
                'kwargs': _DECODER_KWARGS_SCHEMA,
                'state_dict': True
            },
            'additionalProperties': False
        },
        'model': {
            'type': 'object',
            'required': [
                'module',
                'class'
            ],
            'properties': {
                'module': {
                    'type': 'string'
                },
                'class': {
                    'type': 'string'
                },
                'kwargs': {
                    'type': 'object'
                }
            },
            'additionalProperties': False
        }
    },
    'additionalProperties': False
}


_ENCODER_DECODER_MODEL_STATE_DICT_CONFIG_SCHEMA = {
    'type': 'object',
    'required': [
        'encoder',
        'decoder',
        'model'
    ],
    'properties': {
        'encoder': {
            'type': 'object',
            'required': [
                'module',
                'class',
                'kwargs'
            ],
            'properties': {
                'module': {
                    'type': 'string'
                },
                'class': {
                    'type': 'string'
                },
                'kwargs': _ENCODER_KWARGS_SCHEMA
            },
            'additionalProperties': False
        },
        'decoder': {
            'type': 'object',
            'required': [
                'module',
                'class',
                'kwargs'
            ],
            'properties': {
                'module': {
                    'type': 'string'
                },
                'class': {
                    'type': 'string'
                },
                'kwargs': _DECODER_KWARGS_SCHEMA
            },
            'additionalProperties': False
        },
        'model': {
            'type': 'object',
            'required': [
                'module',
                'class',
                'state_dict'
            ],
            'properties': {
                'module': {
                    'type': 'string'
                },
                'class': {
                    'type': 'string'
                },
                'kwargs': {
                    'type': 'object'
                },
                'state_dict': True
            },
            'additionalProperties': False
        }
    },
    'additionalProperties': False,
}


_MODEL_CONFIG_STATE_DICT_SCHEMA = {
    'type': 'object',
    'required': [
        'model'
    ],
    'properties': {
        'model': {
            'type': 'object',
            'required': [
                'module',
                'class',
                'kwargs',
                'state_dict'
            ],
            'properties': {
                'module': {
                    'type': 'string'
                },
                'class': {
                    'type': 'string'
                },
                'kwargs': {
                    'type': 'object',
                    'required': [
                        'dimension',
                        'num_heads',
                        'num_layers',
                        'dim_feedforward',
                        'activation_function',
                        'dropout',
                        'checkpointing'
                    ],
                    'properties': {
                        'dimension': {
                            'type': 'integer',
                            'minimum': 1
                        },
                        'num_heads': {
                            'type': 'integer',
                            'minimum': 1
                        },
                        'num_layers': {
                            'type': 'integer',
                            'minimum': 1
                        },
                        'dim_feedforward': {
                            'type': 'integer',
                            'minimum': 1
                        },
                        'activation_function': {
                            'type': 'string',
                            'enum': [
                                'relu',
                                'gelu'
                            ]
                        },
                        'dropout': {
                            'type': 'number',
                            'minimum': 0.0,
                            'exclusiveMaximum': 1.0
                        },
                        'checkpointing': {
                            'type': 'boolean'
                        }
                    },
                    'additionalProperties': True
                },
                'state_dict': True
            },
            'additionalProperties': False
        }
    },
    'additionalProperties': False
}


_MODEL_CONFIG_SCHEMA = {
    'oneOf': [
        _ENCODER_DECODER_STATE_DICT_CONFIG_SCHEMA,
        _ENCODER_DECODER_MODEL_STATE_DICT_CONFIG_SCHEMA,
        _MODEL_CONFIG_STATE_DICT_SCHEMA
    ]
}


def load_model(model_path: Path) -> nn.Module:
    if not model_path.exists():
        raise RuntimeError(f'{model_path}: Does not exist.')
    if not model_path.is_file():
        raise RuntimeError(f'{model_path}: Not a file.')

    config = torch.load(model_path)
    jsonschema.validate(config, _MODEL_CONFIG_SCHEMA)

    def load_state_dict(model: nn.Module, state_dict: Dict[str, object]) -> None:
        state_dict_fixed = {}
        for key, value in state_dict.items():
            key: str
            key = re.sub('^module\\.', '', key)
            key = re.sub('\\.module\\.', '.', key)
            key = re.sub('^_.+?__', '', key)
            key = re.sub('\\._.+?__', '.', key)
            state_dict_fixed[key] = value
        model.load_state_dict(state_dict_fixed)

    if 'encoder' not in config:
        assert 'decoder' not in config
        model_config = config['model']
        model_module = importlib.import_module(model_config['module'])
        model_class = getattr(model_module, model_config['class'])
        model = model_class(**model_config['kwargs'])
        if not isinstance(model, nn.Module):
            raise RuntimeError(f'{model_module}.{model_class}: Not a module.')

        load_state_dict(model, model_config['state_dict'])
        return model

    encoder_config = config['encoder']
    encoder_module = importlib.import_module(encoder_config['module'])
    encoder_class = getattr(encoder_module, encoder_config['class'])
    encoder_instance = encoder_class(**encoder_config['kwargs'])
    if not isinstance(encoder_instance, nn.Module):
        raise RuntimeError(f'{encoder_module}.{encoder_class}: Not a module.')

    decoder_config = config['decoder']
    decoder_module = importlib.import_module(decoder_config['module'])
    decoder_class = getattr(decoder_module, decoder_config['class'])
    decoder_instance = decoder_class(**decoder_config['kwargs'])
    if not isinstance(decoder_instance, nn.Module):
        raise RuntimeError(f'{decoder_module}.{decoder_class}: Not a module.')

    model_config = config['model']
    model_module = importlib.import_module(model_config['module'])
    model_class = getattr(model_module, model_config['class'])
    kwargs: Dict[str, object] = {}
    if 'kwargs' in model_config:
        kwargs = model_config['kwargs']
    model = model_class(encoder_instance, decoder_instance, **kwargs)
    if not isinstance(model, nn.Module):
        raise RuntimeError(f'{model_module}.{model_class}: Not a module.')

    if 'state_dict' in encoder_config:
        assert 'state_dict' in decoder_config
        load_state_dict(encoder_instance, encoder_config['state_dict'])
        load_state_dict(decoder_instance, decoder_config['state_dict'])
    else:
        assert 'state_dict' in model_config
        load_state_dict(model, model_config['state_dict'])

    return model
