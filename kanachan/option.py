#!/usr/bin/env python3

import re
import datetime
import pathlib
from argparse import ArgumentParser
import logging
import torch
from torch import backends
from kanachan import common


def parse_argument(description: str) -> object:
    ap = ArgumentParser(description=description)
    ap_data = ap.add_argument_group(title='Data')
    ap_data.add_argument(
        '--training-data', type=pathlib.Path, required=True,
        help='Path to training data.', metavar='PATH')
    ap_data.add_argument(
        '--validation-data', type=pathlib.Path,
        help='Path to validation data.', metavar='PATH')
    ap_data.add_argument(
        '--num-workers', default=2, type=int,
        help='# of worker processes in data loading. Defaults to 2.',
        metavar='N')
    ap_device = ap.add_argument_group(title='Device')
    ap_device.add_argument('--device', metavar='DEV')
    ap_device.add_argument(
        '--dtype', default='float32', choices=('float16','float32'))
    ap_model = ap.add_argument_group(title='Model')
    ap_model.add_argument('--dimension', default=512, type=int, metavar='N')
    ap_model.add_argument('--num-heads', default=8, type=int, metavar='N')
    ap_model.add_argument('--num-layers', default=6, type=int, metavar='N')
    ap_model.add_argument(
        '--initial-encoder', type=pathlib.Path, metavar='PATH')
    ap_model.add_argument(
        '--initial-decoder', type=pathlib.Path, metavar='PATH')
    ap_training = ap.add_argument_group(title='Training')
    ap_training.add_argument('--batch-size', default=32, type=int, metavar='N')
    ap_training.add_argument('--max-epoch', default=100, type=int, metavar='N')
    ap_training.add_argument('--optimizer', default='sgd', choices=('adam', 'sgd'))
    ap_training.add_argument('--learning-rate', type=float, metavar='LR')
    ap_target = ap.add_argument_group(title='Target label/regression')
    ap_target.add_argument('--target-index', type=int, required=True, metavar='N')
    ap_target.add_argument('--target-num-classes', type=int, metavar='N')
    ap_output = ap.add_argument_group(title='Output')
    ap_output.add_argument('--output-prefix', type=pathlib.Path, required=True, metavar='PATH')
    ap_output.add_argument('--experiment-name', metavar='NAME')
    ap_output.add_argument('--resume', action='store_true')

    config = ap.parse_args()

    if not config.training_data.exists():
        raise RuntimeError(f'{config.training_data}: does not exist.')
    if config.validation_data is not None and not config.validation_data.exists():
        raise RuntimeError(f'{config.validation_data}: does not exist.')
    if config.num_workers < 0:
        raise RuntimeError(
            f'{config.num_workers}: An invalid number of workers.')

    if config.device is not None:
        if re.search('^(?:cpu)|(?:cuda(?::\\d+)?)', config.device) is None:
            raise RuntimeError(f'{config.device}: invalid device.')
        device = config.device
    elif backends.cuda.is_built():
        device = 'cuda'
    else:
        device = 'cpu'
    if config.dtype == 'float16':
        dtype = torch.float16
    elif config.dtype == 'float32':
        dtype = torch.float32
    else:
        assert(config.dtype == 'float64')
        dtype = torch.float64

    if config.dimension < 1:
        raise RuntimeError(f'{config.dimension}: invalid dimension.')
    if config.num_heads < 1:
        raise RuntimeError(f'{config.num_heads}: an invalid number of heads.')
    if config.num_layers < 1:
        raise RuntimeError(
            f'{config.num_layers}: An invalid number of layers.')
    if config.initial_encoder is not None and not config.initial_encoder.exists():
        raise RuntimeError(f'{config.initial_encoder}: does not exist.')
    if config.initial_encoder is not None and config.resume:
        raise RuntimeError(f'`--initial-encoder` conflicts with `--resume`.')
    if config.initial_decoder is not None and not config.initial_decoder.exists():
        raise RuntimeError(f'{config.initial_decoder}: does not exist.')
    if config.initial_decoder is not None and config.resume:
        raise RuntimeError(f'`--initial-decoder` conflicts with `--resume`.')

    if config.batch_size < 1:
        raise RuntimeError(
            f'{config.batch_size}: An invalid value for `--batch-size`.')
    if config.max_epoch < 1:
        raise RuntimeError(
            f'{config.max_epoch}: An invalid value for `--max-epoch`.')
    if config.optimizer == 'sgd':
        sparse = True
    else:
        assert(config.optimizer == 'adam')
        sparse = False
    if config.optimizer == 'adam':
        if config.learning_rate is None:
            learning_rate = 0.001
        else:
            learning_rate = config.learning_rate
    else:
        assert(config.optimizer == 'sgd')
        if config.learning_rate is None:
            learning_rate = 0.1
        else:
            learning_rate = config.learning_rate

    if config.target_index < 0:
        raise RuntimeError(f'{config.target_index}: An invalid target index.')
    if config.target_num_classes is not None and config.target_num_classes < 2:
        raise RuntimeError(
            f'{config.target_num_classes}: An invalid number of target classes.')

    if config.experiment_name is None:
        now = datetime.datetime.now()
        experiment_name = now.strftime('%Y-%m-%d-%H-%M-%S')
    else:
        experiment_name = config.experiment_name

    experiment_path = pathlib.Path(config.output_prefix / experiment_name)
    if experiment_path.exists() and not config.resume:
        raise RuntimeError(
            f'{experiment_path}: already exists. Did you mean `--resume`?')
    snapshots_path = experiment_path / 'snapshots'
    tensorboard_path = config.output_prefix / 'tensorboard' / experiment_name

    experiment_path.mkdir(parents=True, exist_ok=True)
    common.initialize_logging(experiment_path / 'training.log')

    logging.info(f'Training data: {config.training_data}')
    if config.validation_data is None:
        logging.info(f'Validation data: N/A')
    else:
        logging.info(f'Validation data: {config.validation_data}')
    logging.info(f'# of workers: {config.num_workers}')
    logging.info(f'Device: {device}')
    if backends.cudnn.is_available():
        logging.info(f'cuDNN: available')
        backends.cudnn.benchmark = True
    else:
        logging.info(f'cuDNN: N/A')
    logging.info(f'dtype: {dtype}')
    logging.info(f'Dimension: {config.dimension}')
    logging.info(f'# of heads: {config.num_heads}')
    logging.info(f'# of layers: {config.num_layers}')
    if config.initial_encoder is None:
        logging.info(f'Initial encoder: (initialized randomly)')
    else:
        logging.info(f'Initial encoder: {config.initial_encoder}')
    if config.initial_decoder is None:
        logging.info(f'Initial decoder: (initialized randomly)')
    else:
        logging.info(f'Initial decoder: {config.initial_decoder}')
    logging.info(f'Batch size: {config.batch_size}')
    logging.info(f'Max epoch: {config.max_epoch}')
    logging.info(f'Optimizer: {config.optimizer}')
    logging.info(f'Sparse: {sparse}')
    logging.info(f'Learning rate: {learning_rate}')
    logging.info(f'Target index: {config.target_index}')
    if config.target_num_classes is None:
        # Regression
        logging.info(f'Target: regression')
    else:
        logging.info(f'# of target classes: {config.target_num_classes}')
    if config.resume:
        logging.info(f'Resume from {experiment_path}')
    else:
        logging.info(f'Experiment output: {experiment_path}')

    return {
        'training_data': config.training_data,
        'validation_data': config.validation_data,
        'num_workers': config.num_workers,
        'device': device,
        'dtype': dtype,
        'dimension': config.dimension,
        'num_heads': config.num_heads,
        'num_layers': config.num_layers,
        'initial_encoder': config.initial_encoder,
        'initial_decoder': config.initial_decoder,
        'batch_size': config.batch_size,
        'max_epoch': config.max_epoch,
        'optimizer': config.optimizer,
        'sparse': sparse,
        'learning_rate': learning_rate,
        'target_index': config.target_index,
        'target_num_classes': config.target_num_classes,
        'experiment_name': experiment_name,
        'experiment_path': experiment_path,
        'snapshots_path': snapshots_path,
        'tensorboard_path': tensorboard_path,
        'resume': config.resume,
    }
