#!/usr/bin/env python3

import re
import datetime
import math
from pathlib import Path
import os
from argparse import ArgumentParser
import logging
import sys
from typing import (Optional, Type,)
import torch
from torch import backends
from torch import nn
from torch.optim import (Optimizer, RAdam,)
from torch.utils.data import DataLoader
from torch.distributed import (init_process_group, all_reduce, ReduceOp)
from torch.utils.tensorboard.writer import SummaryWriter
from apex import amp
from apex.parallel import (DistributedDataParallel, convert_syncbn_model,)
from apex.optimizers import (FusedAdam, FusedSGD, FusedLAMB,)
from kanachan.training.constants import (
    NUM_TYPES_OF_SPARSE_FEATURES, NUM_TYPES_OF_ACTIONS,
    MAX_NUM_ACTION_CANDIDATES,)
from kanachan.training.common import (initialize_logging, Dataset,)
from kanachan.training.bert.encoder import Encoder
from kanachan.training.bert.model_loader import load_model
from kanachan.training.iql.policy_model import (PolicyDecoder, PolicyModel,)
from kanachan.training.iql.iterator_adaptor import IteratorAdaptor


def _take_snapshot(
        *, snapshots_path: Path, policy_model: nn.Module, optimizer: Optimizer,
        num_samples: int, infix: str) -> None:
    if isinstance(policy_model, DistributedDataParallel):
        policy_model = policy_model.module

    snapshots_path.mkdir(parents=False, exist_ok=True)
    torch.save(
        policy_model.encoder.state_dict(),
        snapshots_path / f'policy-encoder{infix}.pth')
    torch.save(
        policy_model.decoder.state_dict(),
        snapshots_path / f'policy-decoder{infix}.pth')
    torch.save(
        optimizer.state_dict(),
        snapshots_path / f'policy-optimizer{infix}.pth')
    torch.save(
        amp.state_dict(),
        snapshots_path / f'policy-amp{infix}.pth')
    with open(snapshots_path / f'policy-progress{infix}.yaml', 'w') as f:
        print(f'''---
num_samples: {num_samples}''', file=f)


def _training_epoch(
        *, is_multiprocess: bool, world_size: Optional[int],
        rank: Optional[int], is_main_process: bool, training_data: Path,
        num_workers: int, device: str, value_model: nn.Module,
        q1_model: nn.Module, q2_model: nn.Module, policy_model: nn.Module,
        inverse_temperature: float, advantage_threshold: float, batch_size: int,
        gradient_accumulation_steps: int, max_gradient_norm: float,
        optimizer: Optimizer, snapshots_path: Path, num_epoch_digits: int,
        snapshot_interval: int, epoch: int, epoch_sample_offset,
        num_samples: int, num_samples_to_skip: int, writer: SummaryWriter,
        **kwargs) -> int:
    start_time = datetime.datetime.now()

    # Prepare the training data loader. Note that this data loader must iterate
    # the training data set only once.
    iterator_adaptor = lambda path: IteratorAdaptor(path)
    dataset = Dataset(training_data, iterator_adaptor)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
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

        with torch.no_grad():
            q1 = q1_model(annotation[:4])
            q1 = q1[torch.arange(annotation[4].size(0)), annotation[4]]
            q2 = q2_model(annotation[:4])
            q2 = q2[torch.arange(annotation[4].size(0)), annotation[4]]
            q = torch.minimum(q1, q2)
            q *= inverse_temperature
            q = torch.exp(q)
            value = value_model(annotation[:4])
            value *= inverse_temperature
            value = torch.exp(value)
            advantage = q / value
            max_advantage = torch.max(advantage)
            if is_multiprocess:
                all_reduce(max_advantage, op=ReduceOp.MAX)
            max_advantage = max_advantage.item()
            advantage = torch.clamp(advantage, max=advantage_threshold)
        policy = policy_model(annotation[:4])
        policy = nn.LogSoftmax(dim=1)(policy)
        policy = policy[torch.arange(annotation[4].size(0)), annotation[4]]
        loss = -advantage.detach() * policy
        loss = torch.mean(loss)
        if math.isnan(loss.item()):
            raise RuntimeError('Loss becomes NaN.')

        batch_loss = loss
        if is_multiprocess:
            all_reduce(batch_loss)
            batch_loss /= world_size
        batch_loss = batch_loss.item()

        loss /= gradient_accumulation_steps
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        num_samples += batch_size
        batch_count += 1

        if batch_count % gradient_accumulation_steps == 0:
            gradient = [
                torch.flatten(x.grad) for x in amp.master_params(optimizer) if x.grad is not None]
            gradient = torch.cat(gradient)
            if is_multiprocess:
                all_reduce(gradient)
            gradient_norm = torch.linalg.vector_norm(gradient)
            gradient_norm = gradient_norm.item()
            nn.utils.clip_grad_norm_(
                amp.master_params(optimizer), max_gradient_norm,
                error_if_nonfinite=False)
            optimizer.step()
            optimizer.zero_grad()

            logging.info(
                f'sample = {num_samples},'
f' max advantage = {max_advantage},'
f' loss = {batch_loss},'
f' gradient norm = {gradient_norm}')
            if is_main_process:
                writer.add_scalar('Max Advantage', max_advantage, num_samples)
                writer.add_scalar('Loss', batch_loss, num_samples)
                writer.add_scalar('Gradient Norm', gradient_norm, num_samples)
        else:
            logging.info(
                f'sample = {num_samples},'
f' max advantage = {max_advantage},'
f' loss = {batch_loss}')
            if is_main_process:
                writer.add_scalar('Max Advantage', max_advantage, num_samples)
                writer.add_scalar('Loss', batch_loss, num_samples)

        if is_main_process and last_snapshot is not None and num_samples - last_snapshot >= snapshot_interval:
            epoch_str = str(epoch).zfill(num_epoch_digits)
            infix = f'.{epoch_str}.{num_samples - epoch_sample_offset}'
            _take_snapshot(
                snapshots_path=snapshots_path, policy_model=policy_model,
                optimizer=optimizer, num_samples=num_samples, infix=infix)
            last_snapshot = num_samples

    elapsed_time = datetime.datetime.now() - start_time
    logging.info(
        f'The {epoch}-th training epoch has finished (elapsed time = {elapsed_time}).')

    if is_main_process:
        infix = '.' + str(epoch + 1).zfill(num_epoch_digits)
        _take_snapshot(
            snapshots_path=snapshots_path, policy_model=policy_model,
            optimizer=optimizer, num_samples=num_samples, infix=infix)

    return num_samples


def main() -> None:
    ap = ArgumentParser(
        description='Extract policy by advantage weighted regression (AWR)')
    ap_data = ap.add_argument_group(title='Data')
    ap_data.add_argument(
        '--training-data', type=Path, required=True,
        help='path to training data', metavar='PATH')
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
        '--num-layers', type=int, help='number of layers', metavar='NLAYERS')
    ap_model.add_argument(
        '--dim-final-feedforward', type=int,
        help='dimension of the final feedforward network (defaults to `DIM_FEEDFORWARD`)',
        metavar='DIM_FINAL_FEEDFORWARD')
    ap_model.add_argument(
        '--activation-function', default='gelu', choices=('relu', 'gelu',),
        help='activation function for the feedforward networks of (defaults to `gelu`)',
        metavar='ACTIVATION')
    ap_model.add_argument(
        '--dropout', default=0.1, type=float, help='defaults to `0.1`',
        metavar='DROPOUT')
    ap_model.add_argument(
        '--initial-encoder', type=Path,
        help='path to the initial encoder; mutually exclusive to `--resume`',
        metavar='PATH')
    ap_model.add_argument(
        '--value-model', type=Path, required=True,
        help='path to the value model; mutually exclusive to `--resume`',
        metavar='PATH')
    ap_model.add_argument(
        '--q1-model', type=Path, required=True,
        help='path to the Q1 model; mutually exclusive to `--resume`',
        metavar='PATH')
    ap_model.add_argument(
        '--q2-model', type=Path, required=True,
        help='path to the Q2 model; mutually exclusive to `--resume`',
        metavar='PATH')

    ap_training = ap.add_argument_group(title='Training')
    ap_training.add_argument(
        '--inverse-temperature', type=float, required=True, metavar='BETA')
    ap_training.add_argument(
        '--advantage-threshold', default=math.inf, type=float,
        metavar='ADV_THRESHOLD')
    ap_training.add_argument(
        '--batch-size', type=int, required=True,
        help='batch size', metavar='N')
    ap_training.add_argument(
        '--optimizer', default='lamb',
        choices=('sgd', 'adam', 'radam', 'lamb',),
        help='optimizer (defaults to `lamb`)')
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
        '--max-gradient-norm', default=10.0, type=float,
        help='norm threshold for gradient clipping (defaults to `10.0`)',
        metavar='NORM')
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
            raise RuntimeError('Multi-node not supported.')
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
        raise RuntimeError(f'{config.training_data}: Does not exist.')
    if not config.training_data.is_file():
        raise RuntimeError(f'{config.training_data}: Not a file.')
    if config.num_workers < 0:
        raise RuntimeError(
            f'{config.num_workers}: An invalid number of workers.')

    if config.device is not None:
        m = re.search('^(?:cpu|cuda(\\d*))$', config.device)
        if m is None:
            raise RuntimeError(f'{config.device}: An invalid device.')
        if is_multiprocess and m[1] != '':
            raise RuntimeError(
                'Must not specify any device number in multi-process mode.')
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
        raise RuntimeError('Specify `--model-preset` or `--dimension`.')
    if config.dimension < 1:
        raise RuntimeError(f'{config.dimension}: An invalid embedding dimension.')

    if config.model_preset == 'base' and config.num_heads is None:
        config.num_heads = 12
    if config.model_preset == 'large' and config.num_heads is None:
        config.num_heads = 16
    if config.num_heads is None:
        raise RuntimeError('Specify `--model-preset` or `--num-heads`.')
    if config.num_heads < 1:
        raise RuntimeError(f'{config.num_heads}: An invalid number of heads.')

    if config.dim_feedforward is None:
        config.dim_feedforward = 4 * config.dimension
    if config.dim_feedforward < 1:
        raise RuntimeError(
            f'{config.dim_feedforward}: An invalid dimension of the feedfoward networks')

    if config.model_preset == 'base' and config.num_layers is None:
        config.num_layers = 12
    if config.model_preset == 'large' and config.num_layers is None:
        config.num_layers = 24
    if config.num_layers is None:
        raise RuntimeError('Specify `--model-preset` or `--num-layers`.')
    if config.num_layers < 1:
        raise RuntimeError(f'{config.num_layers}: An invalid number of layers.')

    if config.dim_final_feedforward is None:
        config.dim_final_feedforward = config.dim_feedforward
    if config.dim_final_feedforward < 1:
        raise RuntimeError(
            f'{config.dim_final_feedforward}: An invalid dimension of the final feedforward network.')

    if config.dropout < 0.0 or 1.0 <= config.dropout:
        raise RuntimeError(f'{config.dropout}: An invalid value for `--dropout`.')

    if config.initial_encoder is not None and not config.initial_encoder.exists():
        raise RuntimeError(f'{config.initial_encoder}: Does not exist.')
    if config.initial_encoder is not None and not config.initial_encoder.is_file():
        raise RuntimeError(f'{config.initial_encoder}: Not a file.')
    if config.initial_encoder is not None and config.resume:
        raise RuntimeError(f'`--initial-encoder` conflicts with `--resume`.')
    initial_encoder = config.initial_encoder

    if config.value_model is not None and not config.value_model.exists():
        raise RuntimeError(f'{config.value_model}: Does not exist.')
    if config.value_model is not None and not config.value_model.is_file():
        raise RuntimeError(f'{config.value_model}: Not a file.')
    value_model = config.value_model

    if config.q1_model is not None and not config.q1_model.exists():
        raise RuntimeError(f'{config.q1_model}: Does not exist.')
    if config.q1_model is not None and not config.q1_model.is_file():
        raise RuntimeError(f'{config.q1_model}: Not a file.')
    q1_model = config.q1_model

    if config.q2_model is not None and not config.q2_model.exists():
        raise RuntimeError(f'{config.q2_model}: Does not exist.')
    if config.q2_model is not None and not config.q2_model.is_file():
        raise RuntimeError(f'{config.q2_model}: Not a file.')
    q2_model = config.q2_model

    if config.inverse_temperature < 0.0:
        raise RuntimeError(
            f'{config.inverse_temperature}: An invalid value for `--inverse-temperature`.')

    if config.advantage_threshold <= 0.0:
        raise RuntimeError(
            f'{config.advantage_threshold}: An invalid value for `--advantage-threshold`.')

    if config.batch_size < 1:
        raise RuntimeError(
            f'{config.batch_size}: An invalid value for `--batch-size`.')

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
        raise RuntimeError(f'{learning_rate}: An invalid value for `--learning-rate`.')

    if config.momentum < 0.0 or 1.0 <= config.momentum:
        raise RuntimeError(
            f'{config.momentum}: An invalid value for `--momentum`.')
    momentum = config.momentum

    if config.epsilon is None:
        if config.optimizer in ('adam', 'radam',):
            epsilon = 1.0e-8
        elif config.optimizer == 'lamb':
            epsilon = 1.0e-6
    if epsilon is not None and epsilon <= 0.0:
        raise RuntimeError(f'{epsilon}: An invalid value for `--epsilon`.')

    if config.gradient_accumulation_steps < 1:
        raise RuntimeError(
            f'{config.gradient_accumulation_steps}: An invalid value for `--gradient-accumulation`.')
    if config.max_gradient_norm <= 0.0:
        raise RuntimeError(
            f'{config.max_gradient_norm}: An invalid norm for `--max-gradient-norm`.')

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
            f'{experiment_path}: Already exists; did you mean `--resume`?')
    if rank == 0 and (experiment_path / 'training.0.log').exists() and not config.resume:
        raise RuntimeError(
            f'{experiment_path}: Already exists; did you mean `--resume`?')
    snapshots_path = experiment_path / 'snapshots'
    tensorboard_path = experiment_path / 'tensorboard'

    if config.num_epoch_digits < 1:
        raise RuntimeError(
            f'{config.num_epoch_digits}: invalid number of epoch digits')

    if config.snapshot_interval < 0:
        raise RuntimeError(
            f'{config.snapshot_interval}: An invalid value for `--snapshot-interval`.')

    resume = config.resume

    epoch = 0
    num_samples = 0
    num_samples_to_skip = 0
    if resume:
        assert(initial_encoder is None)
        if not snapshots_path.exists():
            raise RuntimeError(f'{snapshots_path}: Does not exist.')
        snapshot_index = 0
        for child in os.listdir(snapshots_path):
            m = re.search('^policy-(?:encoder|decoder|optimizer|amp)\\.(\\d+)(?:\\.(\\d+))?\\.pth$', child)
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
        encoder_snapshot_path = snapshots_path / f'policy-encoder{infix}.pth'
        if not encoder_snapshot_path.exists():
            raise RuntimeError(f'{encoder_snapshot_path}: Does not exist.')
        if not encoder_snapshot_path.is_file():
            raise RuntimeError(f'{encoder_snapshot_path}: Not a file.')
        decoder_snapshot_path = snapshots_path / f'policy-decoder{infix}.pth'
        if not decoder_snapshot_path.exists():
            raise RuntimeError(f'{decoder_snapshot_path}: Does not exist.')
        if not decoder_snapshot_path.is_file():
            raise RuntimeError(f'{decoder_snapshot_path}: Not a file.')
        optimizer_snapshot_path = snapshots_path / f'policy-optimizer{infix}.pth'
        if not optimizer_snapshot_path.exists():
            raise RuntimeError(f'{optimizer_snapshot_path}: Does not exist.')
        if not optimizer_snapshot_path.is_file():
            raise RuntimeError(f'{optimizer_snapshot_path}: Not a file.')
        amp_snapshot_path = snapshots_path / f'policy-amp{infix}.pth'
        if not amp_snapshot_path.exists():
            raise RuntimeError(f'{amp_snapshot_path}: Does not exist.')
        if not amp_snapshot_path.is_file():
            raise RuntimeError(f'{amp_snapshot_path}: Not a file.')
        progress_file_path = snapshots_path / f'policy-progress{infix}.yaml'
        if not progress_file_path.exists():
            raise RuntimeError(f'{progress_file_path}: Does not exist.')
        if not progress_file_path.is_file():
            raise RuntimeError(f'{progress_file_path}: Not a file.')

        with open(progress_file_path) as f:
            progress_data = yaml.load(f, Loader=yaml.Loader)
        num_samples = progress_data['num_samples']

        if epoch == 0:
            num_samples_to_skip = num_samples
        else:
            with open(snapshots_path / f'progress.{epoch_str}.yaml') as f:
                progress_data = yaml.load(f, Loader=yaml.Loader)
            num_samples_to_skip = num_samples - progress_data['num_samples']

    experiment_path.mkdir(parents=True, exist_ok=True)
    initialize_logging(experiment_path, rank)

    if world_size is None:
        assert(rank is None)
        logging.info(f'World size: N/A (single process)')
        logging.info(f'Process rank: N/A (single process)')
    else:
        assert(rank is not None)
        logging.info(f'World size: {world_size}')
        logging.info(f'Process rank: {rank}')
    logging.info(f'Training data: {config.training_data}')
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
    logging.info(f'Activation function: {config.activation_function}')
    logging.info(f'Dropout: {config.dropout}')
    if initial_encoder is None and not resume:
        logging.info(f'Initial encoder: (initialized randomly)')
    elif initial_encoder is not None:
        logging.info(f'Initial encoder: {initial_encoder}')
    logging.info(f'Value model: {value_model}')
    logging.info(f'Q1 model: {q1_model}')
    logging.info(f'Q2 model: {q2_model}')
    logging.info(f'Inverse temperature: {config.inverse_temperature}')
    logging.info(f'Advantage threshold: {config.advantage_threshold}')
    logging.info(f'Batch size: {config.batch_size}')
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
        f'Virtual batch size: {config.batch_size * config.gradient_accumulation_steps}')
    logging.info(
        f'Norm threshold for gradient clipping: {config.max_gradient_norm}')
    if num_epochs == -1:
        logging.info('Number of epochs to iterate: INFINITY')
    else:
        logging.info(f'Number of epochs to iterate: {num_epochs}')
    if resume:
        logging.info(f'Resume from {experiment_path}')
        logging.info(f'Policy encoder snapshot: {encoder_snapshot_path}')
        logging.info(f'Policy decoder snapshot: {decoder_snapshot_path}')
        logging.info(f'Policy optimizer snapshot: {optimizer_snapshot_path}')
        logging.info(f'Policy AMP snapshot: {amp_snapshot_path}')
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
        'num_workers': config.num_workers,
        'device': device,
        'dimension': config.dimension,
        'num_heads': config.num_heads,
        'dim_feedforward': config.dim_feedforward,
        'num_layers': config.num_layers,
        'dim_final_feedforward': config.dim_final_feedforward,
        'activation_function': config.activation_function,
        'dropout': config.dropout,
        'inverse_temperature': config.inverse_temperature,
        'advantage_threshold': config.advantage_threshold,
        'batch_size': config.batch_size,
        'optimizer': config.optimizer,
        'checkpointing': config.checkpointing,
        'gradient_accumulation_steps': config.gradient_accumulation_steps,
        'max_gradient_norm': config.max_gradient_norm,
        'experiment_name': experiment_name,
        'experiment_path': experiment_path,
        'snapshots_path': snapshots_path,
        'tensorboard_path': tensorboard_path,
        'num_epoch_digits': config.num_epoch_digits,
        'snapshot_interval': config.snapshot_interval,
        'epoch': epoch,
        'num_samples': num_samples,
        'num_samples_to_skip': num_samples_to_skip,
    }

    value_model = load_model(value_model)
    value_model.requires_grad_(False)
    value_model.to(device=config['device'], dtype=torch.float32)

    q1_model = load_model(q1_model)
    q1_model.requires_grad_(False)
    q1_model.to(device=config['device'], dtype=torch.float32)

    q2_model = load_model(q2_model)
    q2_model.requires_grad_(False)
    q2_model.to(device=config['device'], dtype=torch.float32)

    policy_encoder = Encoder(**config)
    policy_decoder = PolicyDecoder(**config)
    policy_model = PolicyModel(policy_encoder, policy_decoder)
    policy_model.to(device=config['device'], dtype=torch.float32)

    if config['optimizer'] == 'sgd':
        construct = lambda model: FusedSGD(
            model.parameters(), lr=learning_rate, momentum=momentum)
    elif config['optimizer'] == 'adam':
        construct = lambda model: FusedAdam(
            model.parameters(), lr=learning_rate, eps=epsilon)
    elif config['optimizer'] == 'radam':
        construct = lambda model: RAdam(
            model.parameters(), lr=learning_rate, eps=epsilon)
    elif config['optimizer'] == 'lamb':
        construct = lambda model: FusedLAMB(
            model.parameters(), lr=learning_rate, eps=epsilon)
    else:
        raise NotImplemented(config['optimizer'])
    optimizer = construct(policy_model)

    if config['is_main_process']:
        policy_model, optimizer = amp.initialize(
            policy_model, optimizer, opt_level=amp_optimization_level)
    else:
        policy_model, optimizer = amp.initialize(
            policy_model, optimizer, opt_level=amp_optimization_level,
            verbosity=0)

    if initial_encoder is not None:
        assert(not resume)
        assert(initial_encoder.exists())
        assert(initial_encoder.is_file())
        policy_encoder.load_state_dict(torch.load(initial_encoder))
    if resume:
        assert(initial_encoder is None)
        assert(encoder_snapshot_path.exists())
        assert(encoder_snapshot_path.is_file())
        assert(decoder_snapshot_path.exists())
        assert(decoder_snapshot_path.is_file())
        assert(optimizer_snapshot_path.exists())
        assert(optimizer_snapshot_path.is_file())
        assert(amp_snapshot_path.exists())
        assert(amp_snapshot_path.is_file())
        policy_encoder.load_state_dict(torch.load(encoder_snapshot_path))
        policy_decoder.load_state_dict(torch.load(decoder_snapshot_path))
        optimizer.load_state_dict(torch.load(optimizer_snapshot_path))
        amp.load_state_dict(torch.load(amp_snapshot_path))

    if config['is_multiprocess']:
        init_process_group(backend='nccl')
        policy_model = DistributedDataParallel(policy_model)
        policy_model = convert_syncbn_model(policy_model)

    config['value_model'] = value_model
    config['q1_model'] = q1_model
    config['q2_model'] = q2_model
    config['policy_model'] = policy_model
    config['optimizer'] = optimizer

    with SummaryWriter(log_dir=config['tensorboard_path']) as writer:
        while num_epochs == -1 or config['epoch'] < num_epochs:
            assert(config['num_samples'] >= config['num_samples_to_skip'])
            config['epoch_sample_offset'] = config['num_samples'] - config['num_samples_to_skip']
            config['num_samples'] = _training_epoch(**config, writer=writer)
            config['epoch'] += 1


if __name__ == '__main__':
    main()
    sys.exit(0)
