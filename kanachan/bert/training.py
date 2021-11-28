#!/usr/bin/env python3

import re
import datetime
import math
from pathlib import Path
import os
from argparse import ArgumentParser
import logging
import json
import sys
from typing import (Optional, Type,)
import torch
from torch import backends
from torch import nn
from torch.optim import (Optimizer, RAdam,)
from torch.utils.data import DataLoader
from torch.distributed import (init_process_group, all_reduce,)
from torch.utils.tensorboard.writer import SummaryWriter
from kanachan import common
from kanachan.iterator_adaptor_base import IteratorAdaptorBase
from kanachan.common import Dataset
from kanachan.bert.encoder import Encoder
from kanachan.bert.model_mode import ModelMode
from apex import amp
from apex.parallel import (DistributedDataParallel, convert_syncbn_model,)
from apex.optimizers import (FusedAdam, FusedSGD, FusedLAMB,)


def _validate(
        *, is_multiprocess: bool, world_size: Optional[int], rank: Optional[int],
        device: str, validation_data: Path, num_workers: int, dimension: int,
        iterator_adaptor_type: Type[IteratorAdaptorBase],
        validation_batch_size: int, model: nn.Module, **kwargs) -> float:
    start_time = datetime.datetime.now()

    # Prepare the validation data loader. Note that this data loader must
    # iterate the validation data set only once.
    iterator_adaptor = lambda path: iterator_adaptor_type(path, dimension)
    dataset = Dataset(validation_data, iterator_adaptor)
    data_loader = DataLoader(
        dataset, batch_size=validation_batch_size, num_workers=num_workers,
        pin_memory=True, drop_last=is_multiprocess)

    loss_function = nn.CrossEntropyLoss()
    validation_loss = 0.0

    for batch_count, annotation in enumerate(data_loader):
        batch_size = len(annotation[0])

        if device != 'cpu':
            if is_multiprocess:
                assert(world_size is not None)
                assert(rank is not None)
                if batch_size % world_size != 0:
                    raise RuntimeError(
                        'Batch size must be divisible by the world size.')
                first = (batch_size // world_size) * rank
                last = (batch_size // world_size) * (rank + 1)
                annotation = tuple(x[first:last].cuda() for x in annotation)
            else:
                annotation = tuple(x.cuda() for x in annotation)

        prediction = model(annotation[:-1])
        loss = loss_function(prediction, annotation[-1])
        if math.isnan(loss.item()):
            raise RuntimeError('Validation loss becomes NaN.')
        if is_multiprocess:
            all_reduce(loss)
            loss /= world_size
        validation_loss += loss.item()

    validation_loss /= batch_count

    elapsed_time = datetime.datetime.now() - start_time
    logging.info(f'Validation has finished (elapsed time = {elapsed_time}).')
    logging.info(f'Validation loss = {validation_loss}')

    return validation_loss


def _training_epoch(
        config: object, *, is_multiprocess: bool, world_size: Optional[int],
        rank: Optional[int], is_main_process: bool, training_data: Path,
        num_workers: int, device: str, dimension: int, encoder: Encoder,
        decoder: nn.Module, model: nn.Module, training_batch_size: int,
        gradient_accumulation_steps: int, max_gradient_norm: float,
        optimizer: Optimizer, iterator_adaptor_type: Type[IteratorAdaptorBase],
        loss_function, snapshots_path: Path, num_epoch_digits: int, epoch: int,
        snapshot_interval: int, num_samples: int, num_samples_to_skip: int,
        writer: SummaryWriter, **kwargs) -> int:
    start_time = datetime.datetime.now()

    # Prepare the training data loader. Note that this data loader must iterate
    # the training data set only once.
    iterator_adaptor = lambda path: iterator_adaptor_type(path, dimension)
    dataset = Dataset(training_data, iterator_adaptor)
    data_loader = DataLoader(
        dataset, batch_size=training_batch_size, num_workers=num_workers,
        pin_memory=True, drop_last=is_multiprocess)

    last_snapshot = None
    if snapshot_interval > 0:
        last_snapshot = num_samples

    skipped_samples = 0

    batch_count = 0
    for annotation in data_loader:
        batch_size = len(annotation[0])

        if skipped_samples < num_samples_to_skip:
            skipped_samples += batch_size
            continue

        if device != 'cpu':
            if is_multiprocess:
                assert(world_size is not None)
                assert(rank is not None)
                if batch_size % world_size != 0:
                    raise RuntimeError(
                        'Batch size must be divisible by the world size.')
                first = (batch_size // world_size) * rank
                last = (batch_size // world_size) * (rank + 1)
                annotation = tuple(x[first:last].cuda() for x in annotation)
            else:
                annotation = tuple(x.cuda() for x in annotation)

        prediction = model(annotation[:-1])
        loss = loss_function(prediction, annotation[-1])
        if math.isnan(loss.item()):
            raise RuntimeError('Training loss becomes NaN.')

        batch_loss = loss
        if is_multiprocess:
            all_reduce(batch_loss)
            batch_loss /= world_size
        batch_loss = batch_loss.item()

        loss = loss / gradient_accumulation_steps
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        num_samples += batch_size

        if (batch_count + 1) % gradient_accumulation_steps == 0:
            gradient = [torch.flatten(x.grad) for x in amp.master_params(optimizer) if x.grad is not None]
            gradient = torch.cat(gradient)
            gradient_norm = torch.linalg.vector_norm(gradient)
            gradient_norm = gradient_norm.item()
            nn.utils.clip_grad_norm_(
                amp.master_params(optimizer), max_gradient_norm,
                error_if_nonfinite=False)
            optimizer.step()
            optimizer.zero_grad()
            logging.info(
                f'sample = {num_samples}, training loss = {batch_loss}, gradient norm = {gradient_norm}')
            if is_main_process:
                writer.add_scalar('Training loss', batch_loss, num_samples)
                writer.add_scalar('Gradient norm', gradient_norm, num_samples)
        else:
            logging.info(
                f'sample = {num_samples}, training loss = {batch_loss}')
            if is_main_process:
                writer.add_scalar('Training loss', batch_loss, num_samples)

        if is_main_process and last_snapshot is not None and num_samples - last_snapshot >= snapshot_interval:
            snapshots_path.mkdir(parents=False, exist_ok=True)
            epoch_str = str(epoch).zfill(num_epoch_digits)
            infix = f'.{epoch_str}.{num_samples}'
            torch.save(
                encoder.state_dict(), snapshots_path / f'encoder{infix}.pth')
            torch.save(
                decoder.state_dict(), snapshots_path / f'decoder{infix}.pth')
            optimizer_state = {
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict(),
            }
            torch.save(
                optimizer_state, snapshots_path / f'optimizer{infix}.pth')
            with open(snapshots_path / f'progress{infix}.json', 'w') as f:
                progress_data = {'num_samples': num_samples}
                json.dump(progress_data, f, separators=(',', ':'))
            last_snapshot = num_samples

        batch_count += 1

    elapsed_time = datetime.datetime.now() - start_time
    logging.info(
        f'A training epoch has finished (elapsed time = {elapsed_time}).')

    if config['validation_data'] is not None:
        assert(config['validation_batch_size'] is not None)
        with ModelMode(config['model'], 'validation'):
            validation_loss = _validate(**config)
        if is_main_process:
            writer.add_scalar(
                'Validation epoch loss', validation_loss, epoch + 1)

    if is_main_process:
        snapshots_path.mkdir(parents=False, exist_ok=True)
        epoch_str = str(epoch + 1).zfill(num_epoch_digits)
        torch.save(
            encoder.state_dict(), snapshots_path / f'encoder.{epoch_str}.pth')
        torch.save(
            decoder.state_dict(), snapshots_path / f'decoder.{epoch_str}.pth')
        optimizer_state = {
            'optimizer': optimizer.state_dict(),
            'amp': amp.state_dict(),
        }
        torch.save(
            optimizer_state, snapshots_path / f'optimizer.{epoch_str}.pth')
        with open(snapshots_path / f'progress.{epoch_str}.json', 'w') as f:
            progress_data = {'num_samples': num_samples}
            json.dump(progress_data, f, separators=(',', ':'))

    return num_samples


def main(*, program_description: str, decoder_type: Type[nn.Module],
         model_type: Type[nn.Module], default_optimizer: str,
         iterator_adaptor_type: Type[IteratorAdaptorBase],
         loss_function) -> None:
    ap = ArgumentParser(description=program_description)
    ap_data = ap.add_argument_group(title='Data')
    ap_data.add_argument(
        '--training-data', type=Path, required=True,
        help='path to training data', metavar='PATH')
    ap_data.add_argument(
        '--validation-data', type=Path, help='path to validation data',
        metavar='PATH')
    ap_data.add_argument(
        '--num-workers', default=2, type=int,
        help='# of worker processes in data loading (defaults to `2`)',
        metavar='NWORKERS')
    ap_device = ap.add_argument_group(title='Device')
    ap_device.add_argument('--device', help='device', metavar='DEV')
    ap_device.add_argument(
        '--amp-optimization-level', default='O2',
        choices=('O0', 'O1', 'O2', 'O3',),
        help='Optimization level for automatic mixed precision (defaults to `O2`)')
    ap_model = ap.add_argument_group(title='Model')
    ap_model.add_argument('--model-preset', choices=('base', 'large',))
    ap_model.add_argument(
        '--dimension', type=int, help='embedding dimension', metavar='DIM')
    ap_model.add_argument(
        '--num-heads', type=int, help='number of heads', metavar='NHEADS')
    ap_model.add_argument(
        '--dim-feedforward', type=int,
        help='dimension of the feedforward network in each layer (defaults to `4 * DIM`)',
        metavar='DIM_FEEDFORWARD')
    ap_model.add_argument(
        '--num-layers', type=int, help='# of layers', metavar='NLAYERS')
    ap_model.add_argument(
        '--dim-final-feedforward', type=int,
        help='dimension of the final feedforward network (defaults to `DIM_FEEDFORWARD`)',
        metavar='DIM_FINAL_FEEDFORWARD')
    ap_model.add_argument(
        '--dropout', default=0.1, type=float, help='defaults to `0.1`',
        metavar='DROPOUT')
    ap_model.add_argument(
        '--activation-function', default='gelu', choices=('relu', 'gelu',),
        help='activation function for the feedforward networks (defaults to `gelu`)',
        metavar='ACTIVATION')
    ap_model.add_argument(
        '--initial-encoder', type=Path,
        help='path to the initial encoder; mutually exclusive to `--resume`',
        metavar='PATH')
    ap_model.add_argument(
        '--initial-decoder', type=Path,
        help='path to the initial decoder; mutually exclusive to `--resume`',
        metavar='PATH')
    ap_training = ap.add_argument_group(title='Training')
    ap_training.add_argument(
        '--training-batch-size', type=int, required=True,
        help='training batch size', metavar='N')
    ap_training.add_argument(
        '--validation-batch-size', type=int, help='validation batch size',
        metavar='N')
    ap_training.add_argument(
        '--freeze-encoder', action='store_true',
        help='freeze encoder parameters during training')
    ap_training.add_argument(
        '--optimizer', default=default_optimizer,
        choices=('sgd', 'adam', 'radam', 'lamb',),
        help=f'optimizer (defaults to `{default_optimizer}`)')
    ap_training.add_argument(
        '--momentum', default=0.9, type=float,
        help='momentum factor; only meaningful for SGD (defaults to `0.9`)',
        metavar='MOMENTUM')
    ap_training.add_argument(
        '--learning-rate', type=float,
        help='learning rate (defaults to `0.1` for `sgd`, `0.001` for `adam`, `radam`, and `lamb`)',
        metavar='LR')
    ap_training.add_argument(
        '--epsilon', type=float,
        help='epsilon parameter; only meaningful for Adam, RAdam, and LAMB\
 (defaults to `1.0e-8` for Adam and RAdam, `1.0e-6` for LAMB)',
        metavar='EPS')
    ap_training.add_argument(
        '--checkpointing', action='store_true', help='enable checkpointing')
    ap_training.add_argument(
        '--gradient-accumulation-steps', default=1, type=int,
        help='# of steps for gradient accumulation (defaults to `1`)',
        metavar='NSTEPS')
    ap_training.add_argument(
        '--max-gradient-norm', default=1.0, type=float,
        help='norm threshold for gradient clipping (defaults to `1.0`)',
        metavar='NORM')
    ap_training.add_argument(
        '--initial-optimizer', type=Path,
        help='path to the initial optimizer state; mutually exclusive to `--resume`',
        metavar='PATH')
    ap_training.add_argument(
        '--num-epochs', default=1, type=int,
        help='number of epochs to iterate (defaults to `1`)', metavar='N')
    ap_output = ap.add_argument_group(title='Output')
    ap_output.add_argument(
        '--output-prefix', type=Path, required=True, metavar='PATH')
    ap_output.add_argument('--experiment-name', metavar='NAME')
    ap_output.add_argument(
        '--num-epoch-digits', default=2, type=int,
        help='number of digits to index epochs (defaults to `2`)',
        metavar='NDIGITS')
    ap_output.add_argument(
        '--snapshot-interval', default=0, type=int,
        help='take a snapshot every specified number of samples (disabled by default)',
        metavar='NSAMPLES')
    ap_output.add_argument('--resume', action='store_true')

    config = ap.parse_args()

    if 'LOCAL_RANK' in os.environ:
        if os.environ['WORLD_SIZE'] != os.environ['LOCAL_WORLD_SIZE']:
            raise RuntimeError('Multi-node not supported')
        world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        rank = int(os.environ['LOCAL_RANK'])
        is_multiprocess = world_size >= 2
        is_main_process = rank == 0
        torch.cuda.set_device(rank)
    else:
        world_size = None
        rank = None
        is_main_process = True
        is_multiprocess = False

    if not config.training_data.exists():
        raise RuntimeError(f'{config.training_data}: does not exist')
    if config.validation_data is not None and not config.validation_data.exists():
        raise RuntimeError(f'{config.validation_data}: does not exist')
    if config.num_workers < 0:
        raise RuntimeError(
            f'{config.num_workers}: invalid number of workers')

    if config.device is not None:
        m = re.search('^(?:cpu|cuda(\\d*))$', config.device)
        if m is None:
            raise RuntimeError(f'{config.device}: invalid device')
        if is_multiprocess and m[1] != '':
            raise RuntimeError(
                'Must not specify any device number in multi-process mode')
        device = config.device
    elif backends.cuda.is_built():
        device = 'cuda'
    else:
        device = 'cpu'

    amp_optimization_level = config.amp_optimization_level

    if config.model_preset == 'base' and config.dimension is None:
        config.dimension = 768
    if config.model_preset == 'large' and config.dimension is None:
        config.dimension = 1024
    if config.dimension is None:
        raise RuntimeError('specify `--model-preset` or `--dimension`')
    if config.dimension < 1:
        raise RuntimeError(f'{config.dimension}: invalid embedding dimension')

    if config.model_preset == 'base' and config.num_heads is None:
        config.num_heads = 12
    if config.model_preset == 'large' and config.num_heads is None:
        config.num_heads = 16
    if config.num_heads is None:
        raise RuntimeError('specify `--model-preset` or `--num-heads`')
    if config.num_heads < 1:
        raise RuntimeError(f'{config.num_heads}: invalid number of heads')

    if config.dim_feedforward is None:
        config.dim_feedforward = 4 * config.dimension
    if config.dim_feedforward < 1:
        raise RuntimeError(
            f'{config.dim_feedforward}: invalid dimension of the feedfoward networks')

    if config.model_preset == 'base' and config.num_layers is None:
        config.num_layers = 12
    if config.model_preset == 'large' and config.num_layers is None:
        config.num_layers = 24
    if config.num_layers is None:
        raise RuntimeError('specify `--model-preset` or `--num-layers`')
    if config.num_layers < 1:
        raise RuntimeError(f'{config.num_layers}: invalid number of layers')

    if config.dim_final_feedforward is None:
        config.dim_final_feedforward = config.dim_feedforward
    if config.dim_final_feedforward < 1:
        raise RuntimeError(
            f'{config.dim_final_feedforward}: invalid dimension of the final feedforward network')

    if config.dropout < 0.0 or 1.0 <= config.dropout:
        raise RuntimeError(f'{config.dropout}: invalid value for dropout')

    if config.initial_encoder is not None and not config.initial_encoder.exists():
        raise RuntimeError(f'{config.initial_encoder}: does not exist')
    if config.initial_encoder is not None and config.resume:
        raise RuntimeError(f'`--initial-encoder` conflicts with `--resume`')
    initial_encoder = config.initial_encoder

    if config.initial_decoder is not None and not config.initial_decoder.exists():
        raise RuntimeError(f'{config.initial_decoder}: does not exist')
    if config.initial_decoder is not None and config.resume:
        raise RuntimeError(f'`--initial-decoder` conflicts with `--resume`')
    initial_decoder = config.initial_decoder

    if config.training_batch_size < 1:
        raise RuntimeError(
            f'{config.training_batch_size}: invalid training batch size')

    if config.validation_data is not None and config.validation_batch_size is None:
        raise RuntimeError('specify `--validation-batch-size`')
    if config.validation_data is None and config.validation_batch_size is not None:
        raise RuntimeError(
            '`--validation-batch-size` specified without `--validation-data`')
    if config.validation_batch_size is not None and config.validation_batch_size < 1:
        raise RuntimeError(
            f'{config.validation_batch_size}: invalid validation batch size')

    freeze_encoder = config.freeze_encoder

    if config.optimizer == 'sgd':
        if config.learning_rate is None:
            learning_rate = 0.1
        else:
            learning_rate = config.learning_rate
    elif config.optimizer in ('adam', 'radam', 'lamb',):
        if config.learning_rate is None:
            learning_rate = 0.001
        else:
            learning_rate = config.learning_rate
    else:
        raise NotImplemented(config.optimizer)
    if learning_rate <= 0.0:
        raise RuntimeError(f'{learning_rate}: invalid value for learning rate')

    if config.momentum < 0.0 or 1.0 <= config.momentum:
        raise RuntimeError(
            f'{config.momentum}: invalid value for momentum factor')
    momentum = config.momentum

    if config.epsilon is None:
        if config.optimizer in ('adam', 'radam',):
            epsilon = 1.0e-8
        elif config.optimizer == 'lamb':
            epsilon = 1.0e-6
    if epsilon is not None and epsilon <= 0.0:
        raise RuntimeError(f'{epsilon}: invalid value for epsilon parameter')

    if config.gradient_accumulation_steps < 1:
        raise RuntimeError(
            f'{config.gradient_accumulation_steps}: invalid steps for gradient accumulation')
    if config.max_gradient_norm <= 0.0:
        raise RuntimeError(
            f'{config.max_gradient_norm}: invalid norm for gradient clipping')

    if config.initial_optimizer is not None and not config.initial_optimizer.exists():
        raise RuntimeError(f'{config.initial_optimizer}: does not exist')
    if config.initial_optimizer is not None and config.resume:
        raise RuntimeError(f'`--initial-optimizer` conflicts with `--resume`')
    initial_optimizer = config.initial_optimizer

    if config.num_epochs <= -2:
        raise RuntimeError(f'{config.num_epochs}: invalid number of epochs')
    num_epochs = config.num_epochs

    if config.experiment_name is None:
        now = datetime.datetime.now()
        experiment_name = now.strftime('%Y-%m-%d-%H-%M-%S')
    else:
        experiment_name = config.experiment_name

    experiment_path = Path(config.output_prefix / experiment_name)
    if rank is None and (experiment_path / 'training.log').exists() and not config.resume:
        raise RuntimeError(
            f'{experiment_path}: already exists; did you mean `--resume`?')
    if rank == 0 and (experiment_path / 'training.0.log').exists() and not config.resume:
        raise RuntimeError(
            f'{experiment_path}: already exists; did you mean `--resume`?')
    snapshots_path = experiment_path / 'snapshots'
    tensorboard_path = experiment_path / 'tensorboard'

    if config.num_epoch_digits < 1:
        raise RuntimeError(
            f'{config.num_epoch_digits}: invalid number of epoch digits')

    if config.snapshot_interval < 0:
        raise RuntimeError(
            f'{config.snapshot_interval}: invalid snapshot interval')

    resume = config.resume

    epoch = 0
    snapshot_index = 0
    num_samples = 0
    num_samples_to_skip = 0
    if resume:
        assert(initial_encoder is None)
        assert(initial_decoder is None)
        assert(initial_optimizer is None)
        if not snapshots_path.exists():
            raise RuntimeError(f'{snapshots_path}: does not exist')
        for child in os.listdir(snapshots_path):
            m = re.search('^(?:encoder|decoder|optimizer)\\.(\\d+)(?:\\.(\\d+))?\\.pth$', child)
            if m is None:
                continue
            tmp = m[1].lstrip('0')
            if tmp == '':
                tmp = '0'
            tmp = int(tmp)
            if tmp > epoch:
                epoch = tmp
                snapshot_index = 0
            if tmp == epoch and m[2] is not None:
                tmp = int(m[2])
                if tmp > snapshot_index:
                    snapshot_index = tmp
        epoch_str = str(epoch).zfill(config.num_epoch_digits)
        snapshot_index_str = ''
        if snapshot_index != 0:
            snapshot_index_str = f'.{snapshot_index}'
        infix = f'.{epoch_str}{snapshot_index_str}'
        encoder_snapshot_path = snapshots_path / f'encoder{infix}.pth'
        if not encoder_snapshot_path.exists():
            raise RuntimeError(f'{encoder_snapshot_path}: does not exist')
        decoder_snapshot_path = snapshots_path / f'decoder{infix}.pth'
        if not decoder_snapshot_path.exists():
            raise RuntimeError(f'{decoder_snapshot_path}: does not exist')
        optimizer_snapshot_path = snapshots_path / f'optimizer{infix}.pth'
        if not optimizer_snapshot_path.exists():
            raise RuntimeError(f'{optimizer_snapshot_path}: does not exist')
        progress_file_path = snapshots_path / f'progress{infix}.json'
        if not progress_file_path.exists():
            raise RuntimeError(f'{progress_file_path}: does not exist')

        with open(progress_file_path) as f:
            progress_data = json.load(f)
        num_samples = progress_data['num_samples']

        if epoch == 0:
            num_samples_to_skip = num_samples
        else:
            with open(snapshots_path / f'progress.{epoch_str}.json') as f:
                progress_data = json.load(f)
            num_samples_to_skip = num_samples - progress_data['num_samples']

    experiment_path.mkdir(parents=True, exist_ok=True)
    common.initialize_logging(experiment_path, rank)

    if world_size is None:
        assert(rank is None)
        logging.info(f'World size: N/A (single process)')
        logging.info(f'Process rank: N/A (single process)')
    else:
        assert(rank is not None)
        logging.info(f'World size: {world_size}')
        logging.info(f'Process rank: {rank}')
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
    logging.info(f'AMP optimization level: {amp_optimization_level}')
    logging.info(f'Embedding dimension: {config.dimension}')
    logging.info(f'# of heads: {config.num_heads}')
    logging.info(
        f'Dimension of the feedforward network in each layer: {config.dim_feedforward}')
    logging.info(f'# of layers: {config.num_layers}')
    logging.info(
        f'Dimension of the final feedforward network: {config.dim_final_feedforward}')
    logging.info(f'Dropout: {config.dropout}')
    logging.info(f'Activation function: {config.activation_function}')
    if initial_encoder is None and not resume:
        logging.info(f'Initial encoder: (initialized randomly)')
    elif initial_encoder is not None:
        logging.info(f'Initial encoder: {initial_encoder}')
    if initial_decoder is None and not resume:
        logging.info(f'Initial decoder: (initialized randomly)')
    elif initial_decoder is not None:
        logging.info(f'Initial decoder: {initial_decoder}')
    logging.info(f'Training batch size: {config.training_batch_size}')
    if config.validation_batch_size is not None:
        logging.info(f'Validation batch size: {config.validation_batch_size}')
    logging.info(f'Freeze encoder: {freeze_encoder}')
    logging.info(f'Optimizer: {config.optimizer}')
    logging.info(f'Learning rate: {learning_rate}')
    if config.optimizer == 'sgd':
        logging.info(f'Momentum factor: {momentum}')
    if config.optimizer in ('adam', 'radam', 'lamb',):
        logging.info(f'Epsilon parameter: {epsilon}')
    logging.info(f'Checkpointing: {config.checkpointing}')
    logging.info(
        f'# of steps for gradient accumulation: {config.gradient_accumulation_steps}')
    logging.info(
        f'Virtual training batch size: {config.training_batch_size * config.gradient_accumulation_steps}')
    logging.info(
        f'norm threshold for gradient clipping: {config.max_gradient_norm}')
    if initial_optimizer is None and not resume:
        logging.info(f'Initial optimizer state: (initialized normally)')
    elif initial_optimizer is not None:
        logging.info(f'Initial optimizer state: {initial_optimizer}')
    if num_epochs == -1:
        logging.info('Number of epochs to iterate: INFINITY')
    else:
        logging.info(f'Number of epochs to iterate: {num_epochs}')
    if resume:
        logging.info(f'Resume from {experiment_path}')
        logging.info(f'Encoder snapshot: {encoder_snapshot_path}')
        logging.info(f'Decoder snapshot: {decoder_snapshot_path}')
        logging.info(f'Optimizer snapshot: {optimizer_snapshot_path}')
        logging.info(f'# of training samples so far: {num_samples}')
        logging.info(f'# of samples to skip: {num_samples_to_skip}')
    else:
        logging.info(f'Experiment output: {experiment_path}')
    logging.info(f'# of digits to index epochs: {config.num_epoch_digits}')
    if config.snapshot_interval == 0:
        logging.info(f'Snapshot interval: N/A')
    else:
        logging.info(f'Snapshot interval: {config.snapshot_interval}')

    config = {
        'is_multiprocess': is_multiprocess,
        'world_size': world_size,
        'rank': rank,
        'is_main_process': is_main_process,
        'training_data': config.training_data,
        'validation_data': config.validation_data,
        'num_workers': config.num_workers,
        'device': device,
        'dimension': config.dimension,
        'num_heads': config.num_heads,
        'dim_feedforward': config.dim_feedforward,
        'num_layers': config.num_layers,
        'dim_final_feedforward': config.dim_final_feedforward,
        'dropout': config.dropout,
        'activation_function': config.activation_function,
        'training_batch_size': config.training_batch_size,
        'validation_batch_size': config.validation_batch_size,
        'optimizer': config.optimizer,
        'checkpointing': config.checkpointing,
        'gradient_accumulation_steps': config.gradient_accumulation_steps,
        'max_gradient_norm': config.max_gradient_norm,
        'experiment_name': experiment_name,
        'experiment_path': experiment_path,
        'snapshots_path': snapshots_path,
        'tensorboard_path': tensorboard_path,
        'num_epoch_digits': config.num_epoch_digits,
        'epoch': epoch,
        'snapshot_interval': config.snapshot_interval,
        'num_samples': num_samples,
        'num_samples_to_skip': num_samples_to_skip,
    }

    encoder = Encoder(**config)
    decoder = decoder_type(**config)
    model = model_type(encoder, decoder, freeze_encoder=freeze_encoder)
    model.to(device=config['device'], dtype=torch.float32)

    if config['optimizer'] == 'sgd':
        optimizer = FusedSGD(
            model.parameters(), lr=learning_rate, momentum=momentum)
    elif config['optimizer'] == 'adam':
        optimizer = FusedAdam(model.parameters(), lr=learning_rate, eps=epsilon)
    elif config['optimizer'] == 'radam':
        optimizer = RAdam(model.parameters(), lr=learning_rate, eps=epsilon)
    elif config['optimizer'] == 'lamb':
        optimizer = FusedLAMB(model.parameters(), lr=learning_rate, eps=epsilon)
    else:
        raise NotImplemented(config['optimizer'])

    if config['is_main_process']:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=amp_optimization_level)
    else:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=amp_optimization_level, verbosity=0)

    if initial_encoder is not None:
        assert(not resume)
        assert(initial_encoder.exists())
        encoder.load_state_dict(torch.load(initial_encoder))
    if initial_decoder is not None:
        assert(not resume)
        assert(initial_decoder.exists())
        decoder.load_state_dict(torch.load(initial_decoder))
    if resume:
        assert(initial_encoder is None)
        assert(initial_decoder is None)
        assert(encoder_snapshot_path.exists())
        assert(decoder_snapshot_path.exists())
        encoder.load_state_dict(torch.load(encoder_snapshot_path))
        decoder.load_state_dict(torch.load(decoder_snapshot_path))

    if initial_optimizer is not None:
        assert(not resume)
        assert(initial_optimizer.exists())
        optimizer_state = torch.load(initial_optimizer)
        optimizer.load_state_dict(optimizer_state['optimizer'])
        amp.load_state_dict(optimizer_state['amp'])
    if resume:
        assert(initial_optimizer is None)
        assert(optimizer_snapshot_path.exists())
        optimizer_state = torch.load(optimizer_snapshot_path)
        optimizer.load_state_dict(optimizer_state['optimizer'])
        amp.load_state_dict(optimizer_state['amp'])

    if config['is_multiprocess']:
        init_process_group(backend='nccl')
        model = DistributedDataParallel(model)
        model = convert_syncbn_model(model)

    config['encoder'] = encoder
    config['decoder'] = decoder
    config['model'] = model
    config['iterator_adaptor_type'] = iterator_adaptor_type
    config['loss_function'] = loss_function
    config['optimizer'] = optimizer

    with SummaryWriter(log_dir=config['tensorboard_path']) as writer:
        if config['validation_data'] is not None and snapshot_index == 0:
            assert(config['validation_batch_size'] is not None)
            with ModelMode(config['model'], 'validation'):
                validation_loss = _validate(**config)
            if config['is_main_process']:
                writer.add_scalar(
                    'Validation epoch loss', validation_loss, epoch)

        initial_epoch = config['epoch']
        while num_epochs == -1 or config['epoch'] < initial_epoch + num_epochs:
            config['num_samples'] = _training_epoch(
                config, **config, writer=writer)
            config['epoch'] += 1
