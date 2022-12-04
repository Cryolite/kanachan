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
from torch.distributed import (init_process_group, all_reduce,)
from torch.utils.tensorboard.writer import SummaryWriter
from apex import amp
from apex.parallel import (DistributedDataParallel, convert_syncbn_model,)
from apex.optimizers import (FusedAdam, FusedSGD, FusedLAMB,)
from kanachan.training.constants import (
    NUM_TYPES_OF_SPARSE_FEATURES, NUM_TYPES_OF_ACTIONS,
    MAX_NUM_ACTION_CANDIDATES,)
from kanachan.training.common import (initialize_logging, Dataset,)
from kanachan.training.bert.encoder import Encoder
from kanachan.training.iql.value_model import (ValueDecoder, ValueModel,)
from kanachan.training.iql.q_model import (QDecoder, QModel,)
from kanachan.training.iql.iterator_adaptor import IteratorAdaptor


def _training(
        *, is_multiprocess: bool, world_size: Optional[int],
        rank: Optional[int], is_main_process: bool, training_data: Path,
        num_workers: int, device: str, value_model: nn.Module,
        q_source1_model: nn.Module, q_source2_model: nn.Module,
        q_target1_model: nn.Module, q_target2_model: nn.Module,
        discount_factor: float, expectile: float, target_update_interval: int,
        target_update_rate: float, batch_size: int,
        gradient_accumulation_steps: int, v_max_gradient_norm: float,
        q_max_gradient_norm: float, value_optimizer: Optimizer,
        q1_optimizer: Optimizer, q2_optimizer: Optimizer, snapshots_path: Path,
        snapshot_interval: int, num_samples: int, writer: SummaryWriter,
        **kwargs) -> None:
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
        local_batch_size = batch_size

        if skipped_samples < num_samples:
            skipped_samples += batch_size
            continue

        if device != 'cpu':
            if is_multiprocess:
                assert(world_size is not None)
                assert(rank is not None)
                if batch_size % world_size != 0:
                    raise RuntimeError(
                        'Batch size must be divisible by the world size.')
                local_batch_size //= world_size
                first = local_batch_size * rank
                last = local_batch_size * (rank + 1)
                annotation = tuple(x[first:last].cuda() for x in annotation)
            else:
                annotation = tuple(x.cuda() for x in annotation)

        with torch.no_grad():
            q1 = q_target1_model(annotation[:4])
            q1 = q1[torch.arange(local_batch_size), annotation[4]]
            q2 = q_target2_model(annotation[:4])
            q2 = q2[torch.arange(local_batch_size), annotation[4]]
            q = torch.minimum(q1, q2)
        value = value_model(annotation[:4])
        value_loss = q.detach() - value
        value_loss = torch.where(
            value_loss < 0.0, (1.0 - expectile) * (value_loss ** 2.0),
            expectile * (value_loss ** 2.0))
        value_loss = torch.mean(value_loss)
        if math.isnan(value_loss.item()):
            raise RuntimeError('Value loss becomes NaN.')

        value_batch_loss = value_loss
        if is_multiprocess:
            all_reduce(value_batch_loss)
            value_batch_loss /= world_size
        value_batch_loss = value_batch_loss.item()

        value_loss /= gradient_accumulation_steps
        with amp.scale_loss(value_loss, value_optimizer) as scaled_value_loss:
            scaled_value_loss.backward()

        mask = [0.0 if annotation[5][i][0].item() == NUM_TYPES_OF_SPARSE_FEATURES else 1.0 for i in range(local_batch_size)]
        mask = torch.tensor(mask, device=annotation[5].device, dtype=torch.float32)
        reward = annotation[9]
        with torch.no_grad():
            value = value_model(annotation[5:9])
            value *= discount_factor
            value *= mask
        q1 = q_source1_model(annotation[:4])
        q1 = q1[torch.arange(local_batch_size), annotation[4]]
        q1_loss = reward + value - q1
        q1_loss = q1_loss ** 2.0
        q2 = q_source2_model(annotation[:4])
        q2 = q2[torch.arange(local_batch_size), annotation[4]]
        q2_loss = reward + value - q2
        q2_loss = q2_loss ** 2.0

        q1_loss = torch.mean(q1_loss)
        if math.isnan(q1_loss.item()):
            raise RuntimeError('Q1 loss becomes NaN.')
        q2_loss = torch.mean(q2_loss)
        if math.isnan(q2_loss.item()):
            raise RuntimeError('Q2 loss becomes NaN.')

        q1_batch_loss = q1_loss
        if is_multiprocess:
            all_reduce(q1_batch_loss)
            q1_batch_loss /= world_size
        q1_batch_loss = q1_batch_loss.item()
        q2_batch_loss = q2_loss
        if is_multiprocess:
            all_reduce(q2_batch_loss)
            q2_batch_loss /= world_size
        q2_batch_loss = q2_batch_loss.item()

        q1_loss /= gradient_accumulation_steps
        with amp.scale_loss(q1_loss, q1_optimizer) as scaled_q1_loss:
            scaled_q1_loss.backward()
        q2_loss /= gradient_accumulation_steps
        with amp.scale_loss(q2_loss, q2_optimizer) as scaled_q2_loss:
            scaled_q2_loss.backward()

        num_samples += batch_size
        batch_count += 1

        if batch_count % gradient_accumulation_steps == 0:
            value_gradient = [
                torch.flatten(x.grad) for x in amp.master_params(value_optimizer) if x.grad is not None]
            value_gradient = torch.cat(value_gradient)
            if is_multiprocess:
                all_reduce(value_gradient)
            value_gradient_norm = torch.linalg.vector_norm(value_gradient)
            value_gradient_norm = value_gradient_norm.item()
            nn.utils.clip_grad_norm_(
                amp.master_params(value_optimizer), v_max_gradient_norm,
                error_if_nonfinite=False)
            value_optimizer.step()
            value_optimizer.zero_grad()

            q1_gradient = [
                torch.flatten(x.grad) for x in amp.master_params(q1_optimizer) if x.grad is not None]
            q1_gradient = torch.cat(q1_gradient)
            if is_multiprocess:
                all_reduce(q1_gradient)
            q1_gradient_norm = torch.linalg.vector_norm(q1_gradient)
            q1_gradient_norm = q1_gradient_norm.item()
            nn.utils.clip_grad_norm_(
                amp.master_params(q1_optimizer), q_max_gradient_norm,
                error_if_nonfinite=False)
            q1_optimizer.step()
            q1_optimizer.zero_grad()

            q2_gradient = [
                torch.flatten(x.grad) for x in amp.master_params(q2_optimizer) if x.grad is not None]
            q2_gradient = torch.cat(q2_gradient)
            if is_multiprocess:
                all_reduce(q2_gradient)
            q2_gradient_norm = torch.linalg.vector_norm(q2_gradient)
            q2_gradient_norm = q2_gradient_norm.item()
            nn.utils.clip_grad_norm_(
                amp.master_params(q2_optimizer), q_max_gradient_norm,
                error_if_nonfinite=False)
            q2_optimizer.step()
            q2_optimizer.zero_grad()

            if batch_count % (gradient_accumulation_steps * target_update_interval) == 0:
                param_source1_iter = amp.master_params(q1_optimizer)
                param_target1_iter = q_target1_model.parameters()
                for param_source1, param_target1 in zip(param_source1_iter, param_target1_iter):
                    param_target1 *= (1.0 - target_update_rate)
                    param_target1 += target_update_rate * param_source1

                param_source2_iter = amp.master_params(q2_optimizer)
                param_target2_iter = q_target2_model.parameters()
                for param_source2, param_target2 in zip(param_source2_iter, param_target2_iter):
                    param_target2 *= (1.0 - target_update_rate)
                    param_target2 += target_update_rate * param_source2

            logging.info(
                f'sample = {num_samples}, value loss = {value_batch_loss},'
f' Q1 loss = {q1_batch_loss},'
f' Q2 loss = {q2_batch_loss},'
f' value gradient norm = {value_gradient_norm},'
f' Q1 gradient norm = {q1_gradient_norm},'
f' Q2 gradient norm = {q2_gradient_norm}')
            if is_main_process:
                writer.add_scalar('Value Loss', value_batch_loss, num_samples)
                writer.add_scalar(
                    'Value Gradient Norm', value_gradient_norm, num_samples)
                writer.add_scalars(
                    'Q Loss', { 'Q1': q1_batch_loss, 'Q2': q2_batch_loss },
                    num_samples)
                writer.add_scalars(
                    'Q Gradient Norm',
                    { 'Q1': q1_gradient_norm, 'Q2': q2_gradient_norm },
                    num_samples)
        else:
            logging.info(
                f'sample = {num_samples}, value loss = {value_batch_loss},'
f' Q1 loss = {q1_batch_loss}, Q2 loss = {q2_batch_loss}')
            if is_main_process:
                writer.add_scalar('Value Loss', value_batch_loss, num_samples)
                writer.add_scalars(
                    'Q Loss', { 'Q1': q1_batch_loss, 'Q2': q2_batch_loss },
                    num_samples)

        if is_main_process and last_snapshot is not None and num_samples - last_snapshot >= snapshot_interval:
            snapshots_path.mkdir(parents=False, exist_ok=True)
            torch.save(
                value_model.state_dict(),
                snapshots_path / f'value.{num_samples}.pth')
            torch.save(
                q_source1_model.state_dict(),
                snapshots_path / f'q-source1.{num_samples}.pth')
            torch.save(
                q_source2_model.state_dict(),
                snapshots_path / f'q-source2.{num_samples}.pth')
            torch.save(
                q_target1_model.state_dict(),
                snapshots_path / f'q-target1.{num_samples}.pth')
            torch.save(
                q_target2_model.state_dict(),
                snapshots_path / f'q-target2.{num_samples}.pth')
            torch.save(
                value_optimizer.state_dict(),
                snapshots_path / f'value-optimizer.{num_samples}.pth')
            torch.save(
                q1_optimizer.state_dict(),
                snapshots_path / f'q1-optimizer.{num_samples}.pth')
            torch.save(
                q2_optimizer.state_dict(),
                snapshots_path / f'q2-optimizer.{num_samples}.pth')
            torch.save(
                amp.state_dict(), snapshots_path / f'amp.{num_samples}.pth')
            last_snapshot = num_samples

    elapsed_time = datetime.datetime.now() - start_time
    logging.info(
        f'A training has finished (elapsed time = {elapsed_time}).')

    if is_main_process:
        snapshots_path.mkdir(parents=False, exist_ok=True)
        torch.save(
            value_model.state_dict(),
            snapshots_path / f'value.pth')
        torch.save(
            q_source1_model.state_dict(),
            snapshots_path / f'q-source1.pth')
        torch.save(
            q_source2_model.state_dict(),
            snapshots_path / f'q-source2.pth')
        torch.save(
            q_target1_model.state_dict(),
            snapshots_path / f'q-target1.pth')
        torch.save(
            q_target2_model.state_dict(),
            snapshots_path / f'q-target2.pth')
        torch.save(
            value_optimizer.state_dict(),
            snapshots_path / f'value-optimizer.pth')
        torch.save(
            q1_optimizer.state_dict(),
            snapshots_path / f'q1-optimizer.pth')
        torch.save(
            q2_optimizer.state_dict(),
            snapshots_path / f'q2-optimizer.pth')
        torch.save(
            amp.state_dict(), snapshots_path / f'amp.pth')


def main() -> None:
    ap = ArgumentParser(description='Train by implicit Q-learning (IQL)')
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
        '--initial-model-prefix', type=Path,
        help='prefix to initial model; mutually exclusive to `--resume`',
        metavar='PREFIX')
    ap_model.add_argument(
        '--initial-model-index', type=int,
        help='index of snapshots for initial model; mutually exclusive to `--resume`',
        metavar='N')

    ap_training = ap.add_argument_group(title='Training')
    ap_training.add_argument(
        '--discount-factor', type=float, required=True, metavar='GAMMA')
    ap_training.add_argument(
        '--expectile', type=float, required=True, metavar='TAU')
    ap_training.add_argument(
        '--target-update-interval', type=int, required=True, metavar='N')
    ap_training.add_argument(
        '--target-update-rate', type=float, required=True, metavar='ALPHA')
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
        '--v-max-gradient-norm', default=1.0, type=float,
        help='norm threshold for gradient clipping on value (defaults to `1.0`)',
        metavar='NORM')
    ap_training.add_argument(
        '--q-max-gradient-norm', default=10.0, type=float,
        help='norm threshold for gradient clipping on Q (defaults to `10.0`)',
        metavar='NORM')

    ap_output = ap.add_argument_group(title='Output')
    ap_output.add_argument(
        '--output-prefix', type=Path, required=True, metavar='PATH')
    ap_output.add_argument('--experiment-name', metavar='NAME')
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

    if config.initial_model_prefix is not None and not config.initial_model_prefix.exists():
        raise RuntimeError(f'{config.initial_model_prefix}: Does not exist.')
    if config.initial_model_prefix is not None and not config.initial_model_prefix.is_dir():
        raise RuntimeError(f'{config.initial_model_prefix}: Not a directory')
    if config.initial_model_prefix is not None and config.resume:
        raise RuntimeError('`--initial-model-prefix` conflicts with `--resume`.')
    initial_model_prefix = config.initial_model_prefix

    if config.initial_model_index is not None and config.initial_model_index < 0:
        raise RuntimeError(
            f'{config.initial_model_index}: An invalid initial model index.')
    if config.initial_model_index is not None and initial_model_prefix is None:
        raise RuntimeError('`--initial-model-index` requires `--initial-model-prefix`.')
    if config.initial_model_index is not None and config.resume:
        raise RuntimeError(f'`--initial-model-index` conflicts with `--resume`.')
    initial_model_index = config.initial_model_index

    if initial_model_prefix is not None:
        assert(not config.resume)
        if initial_model_index is None:
            for child in os.listdir(initial_model_prefix):
                m = re.search('^(?:value|q-source[12]|q-target[12]|value-optimizer|q[12]-optimizer)\\.(\\d+)\\.pth$', child)
                if m is None:
                    continue
                if initial_model_index is None:
                    initial_model_index = 0
                if int(m[1]) > initial_model_index:
                    initial_model_index = int(m[1])
        if initial_model_index is None:
            raise RuntimeError(f'{initial_model_prefix}: No model found.')
        value_snapshot_path = initial_model_prefix / f'value.{initial_model_index}.pth'
        if not value_snapshot_path.exists():
            raise RuntimeError(f'{value_snapshot_path}: Does not exist.')
        if not value_snapshot_path.is_file():
            raise RuntimeError(f'{value_snapshot_path}: Not a file.')
        q_source1_snapshot_path = initial_model_prefix / f'q-source1.{initial_model_index}.pth'
        if not q_source1_snapshot_path.exists():
            raise RuntimeError(f'{q_source1_snapshot_path}: Does not exist.')
        if not q_source1_snapshot_path.is_file():
            raise RuntimeError(f'{q_source1_snapshot_path}: Not a file.')
        q_source2_snapshot_path = initial_model_prefix / f'q-source2.{initial_model_index}.pth'
        if not q_source2_snapshot_path.exists():
            raise RuntimeError(f'{q_source2_snapshot_path}: Does not exist.')
        if not q_source2_snapshot_path.is_file():
            raise RuntimeError(f'{q_source2_snapshot_path}: Not a file.')
        q_target1_snapshot_path = initial_model_prefix / f'q-target1.{initial_model_index}.pth'
        if not q_target1_snapshot_path.exists():
            raise RuntimeError(f'{q_target1_snapshot_path}: Does not exist.')
        if not q_target1_snapshot_path.is_file():
            raise RuntimeError(f'{q_target1_snapshot_path}: Not a file.')
        q_target2_snapshot_path = initial_model_prefix / f'q-target2.{initial_model_index}.pth'
        if not q_target2_snapshot_path.exists():
            raise RuntimeError(f'{q_target2_snapshot_path}: Does not exist.')
        if not q_target2_snapshot_path.is_file():
            raise RuntimeError(f'{q_target2_snapshot_path}: Not a file.')
        value_optimizer_snapshot_path = initial_model_prefix / f'value-optimizer.{initial_model_index}.pth'
        if not value_optimizer_snapshot_path.is_file():
            value_optimizer_snapshot_path = None
        q1_optimizer_snapshot_path = initial_model_prefix / f'q1-optimizer.{initial_model_index}.pth'
        if not q1_optimizer_snapshot_path.is_file():
            q1_optimizer_snapshot_path = None
        q2_optimizer_snapshot_path = initial_model_prefix / f'q2-optimizer.{initial_model_index}.pth'
        if not q2_optimizer_snapshot_path.is_file():
            q2_optimizer_snapshot_path = None
        amp_snapshot_path = initial_model_prefix / f'amp.{initial_model_index}.pth'
        if not amp_snapshot_path.is_file():
            amp_snapshot_path = None

    if config.discount_factor <= 0.0 or 1.0 < config.discount_factor:
        raise RuntimeError(
            f'{config.discount_factor}: An invalid value for `--discount-factor`.')

    if config.expectile <= 0.0 or 1.0 <= config.expectile:
        raise RuntimeError(
            f'{config.expectile}: An invalid value for `--expectile`.')

    if config.target_update_interval <= 0:
        raise RuntimeError(
            f'{config.target_update_interval}: An invalid value for `--target-update-interval`.')

    if config.target_update_rate <= 0.0 or 1.0 <= config.target_update_rate:
        raise RuntimeError(
            f'{config.target_update_rate}: An invalid value for `--target-update-rate`.')

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
    else:
        epsilon = config.epsilon
    if epsilon is not None and epsilon <= 0.0:
        raise RuntimeError(f'{epsilon}: An invalid value for `--epsilon`.')

    if config.gradient_accumulation_steps < 1:
        raise RuntimeError(
            f'{config.gradient_accumulation_steps}: An invalid value for `--gradient-accumulation`.')
    if config.v_max_gradient_norm <= 0.0:
        raise RuntimeError(
            f'{config.v_max_gradient_norm}: An invalid value for `--v-max-gradient-norm`.')
    if config.q_max_gradient_norm <= 0.0:
        raise RuntimeError(
            f'{config.q_max_gradient_norm}: An invalid value for `--q-max-gradient-norm`.')

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

    if config.snapshot_interval < 0:
        raise RuntimeError(
            f'{config.snapshot_interval}: An invalid value for `--snapshot-interval`.')

    resume = config.resume

    num_samples = 0
    if resume:
        assert(initial_model_prefix is None)
        assert(initial_model_index is None)
        if not snapshots_path.exists():
            raise RuntimeError(f'{snapshots_path}: Does not exist.')
        for child in os.listdir(snapshots_path):
            m = re.search('^(?:value|q-source[12]|q-target[12]|value-optimizer|q[12]-optimizer)\\.(\\d+)\\.pth$', child)
            if m is None:
                continue
            if int(m[1]) > num_samples:
                num_samples = int(m[1])
        value_snapshot_path = snapshots_path / f'value.{num_samples}.pth'
        if not value_snapshot_path.exists():
            raise RuntimeError(f'{value_snapshot_path}: Does not exist.')
        if not value_snapshot_path.is_file():
            raise RuntimeError(f'{value_snapshot_path}: Not a file.')
        q_source1_snapshot_path = snapshots_path / f'q-source1.{num_samples}.pth'
        if not q_source1_snapshot_path.exists():
            raise RuntimeError(f'{q_source1_snapshot_path}: Does not exist.')
        if not q_source1_snapshot_path.is_file():
            raise RuntimeError(f'{q_source1_snapshot_path}: Not a file.')
        q_source2_snapshot_path = snapshots_path / f'q-source2.{num_samples}.pth'
        if not q_source2_snapshot_path.exists():
            raise RuntimeError(f'{q_source2_snapshot_path}: Does not exist.')
        if not q_source2_snapshot_path.is_file():
            raise RuntimeError(f'{q_source2_snapshot_path}: Not a file.')
        q_target1_snapshot_path = snapshots_path / f'q-target1.{num_samples}.pth'
        if not q_target1_snapshot_path.exists():
            raise RuntimeError(f'{q_target1_snapshot_path}: Does not exist.')
        if not q_target1_snapshot_path.is_file():
            raise RuntimeError(f'{q_target1_snapshot_path}: Not a file.')
        q_target2_snapshot_path = snapshots_path / f'q-target2.{num_samples}.pth'
        if not q_target2_snapshot_path.exists():
            raise RuntimeError(f'{q_target2_snapshot_path}: Does not exist.')
        if not q_target2_snapshot_path.is_file():
            raise RuntimeError(f'{q_target2_snapshot_path}: Not a file.')
        value_optimizer_snapshot_path = snapshots_path / f'value-optimizer.{num_samples}.pth'
        if not value_optimizer_snapshot_path.exists():
            raise RuntimeError(f'{value_optimizer_snapshot_path}: Does not exist.')
        if not value_optimizer_snapshot_path.is_file():
            raise RuntimeError(f'{value_optimizer_snapshot_path}: Not a file.')
        q1_optimizer_snapshot_path = snapshots_path / f'q1-optimizer.{num_samples}.pth'
        if not q1_optimizer_snapshot_path.exists():
            raise RuntimeError(f'{q1_optimizer_snapshot_path}: Does not exist.')
        if not q1_optimizer_snapshot_path.is_file():
            raise RuntimeError(f'{q1_optimizer_snapshot_path}: Not a file.')
        q2_optimizer_snapshot_path = snapshots_path / f'q2-optimizer.{num_samples}.pth'
        if not q2_optimizer_snapshot_path.exists():
            raise RuntimeError(f'{q2_optimizer_snapshot_path}: Does not exist.')
        if not q2_optimizer_snapshot_path.is_file():
            raise RuntimeError(f'{q2_optimizer_snapshot_path}: Not a file.')
        amp_snapshot_path = snapshots_path / f'amp.{num_samples}.pth'
        if not amp_snapshot_path.exists():
            raise RuntimeError(f'{amp_snapshot_path}: Does not exist.')
        if not amp_snapshot_path.is_file():
            raise RuntimeError(f'{amp_snapshot_path}: Not a file.')

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
    if initial_model_prefix is None and not resume:
        logging.info('Initial model: (initialized randomly)')
    elif initial_model_prefix is not None:
        logging.info(f'Initial model prefix: {initial_model_prefix}')
        if initial_model_index is not None:
            logging.info(f'Initlal model index: {initial_model_index}')
        else:
            logging.info('Initial model index: (latest one)')
    logging.info(f'Discount factor: {config.discount_factor}')
    logging.info(f'Expectile: {config.expectile}')
    logging.info(f'Target update interval: {config.target_update_interval}')
    logging.info(f'Target update rate: {config.target_update_rate}')
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
        f'Norm threshold for gradient clipping on value: {config.v_max_gradient_norm}')
    logging.info(
        f'Norm threshold for gradient clipping on Q: {config.q_max_gradient_norm}')
    if initial_model_prefix is not None:
        assert(not resume)
        logging.info(f'Initial value network snapshot: {value_snapshot_path}')
        logging.info(f'Initial q source network 1 snapshot: {q_source1_snapshot_path}')
        logging.info(f'Initial q source network 2 snapshot: {q_source2_snapshot_path}')
        logging.info(f'Initial q target network 1 snapshot: {q_target1_snapshot_path}')
        logging.info(f'Initial q target network 2 snapshot: {q_target2_snapshot_path}')
        if value_optimizer_snapshot_path is not None:
            logging.info(f'Initial value optimizer snapshot: {value_optimizer_snapshot_path}')
        if q1_optimizer_snapshot_path is not None:
            logging.info(f'Initial Q1 optimizer snapshot: {q1_optimizer_snapshot_path}')
        if q2_optimizer_snapshot_path is not None:
            logging.info(f'Initial Q2 optimizer snapshot: {q2_optimizer_snapshot_path}')
        if amp_snapshot_path is not None:
            logging.info(f'Initial AMP snapshot: {amp_snapshot_path}')
    if resume:
        assert(initial_model_prefix is None)
        logging.info(f'Resume from {experiment_path}')
        logging.info(f'Value network snapshot: {value_snapshot_path}')
        logging.info(f'Q source network 1 snapshot: {q_source1_snapshot_path}')
        logging.info(f'Q source network 2 snapshot: {q_source2_snapshot_path}')
        logging.info(f'Q target network 1 snapshot: {q_target1_snapshot_path}')
        logging.info(f'Q target network 2 snapshot: {q_target2_snapshot_path}')
        logging.info(f'Value network optimizer snapshot: {value_optimizer_snapshot_path}')
        logging.info(f'Q source network 1 optimizer snapshot: {q1_optimizer_snapshot_path}')
        logging.info(f'Q source network 2 optimizer snapshot: {q2_optimizer_snapshot_path}')
        logging.info(f'AMP snapshot: {amp_snapshot_path}')
        logging.info(f'# of training samples so far: {num_samples}')
    else:
        logging.info(f'Experiment output: {experiment_path}')
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
        'dropout': config.dropout,
        'activation_function': config.activation_function,
        'discount_factor': config.discount_factor,
        'expectile': config.expectile,
        'target_update_interval': config.target_update_interval,
        'target_update_rate': config.target_update_rate,
        'batch_size': config.batch_size,
        'optimizer': config.optimizer,
        'checkpointing': config.checkpointing,
        'gradient_accumulation_steps': config.gradient_accumulation_steps,
        'v_max_gradient_norm': config.v_max_gradient_norm,
        'q_max_gradient_norm': config.q_max_gradient_norm,
        'experiment_name': experiment_name,
        'experiment_path': experiment_path,
        'snapshots_path': snapshots_path,
        'tensorboard_path': tensorboard_path,
        'snapshot_interval': config.snapshot_interval,
        'num_samples': num_samples,
    }

    value_encoder = Encoder(**config)
    value_decoder = ValueDecoder(**config)
    value_model = ValueModel(value_encoder, value_decoder)
    value_model.to(device=config['device'], dtype=torch.float32)

    q_source1_encoder = Encoder(**config)
    q_source1_decoder = QDecoder(**config)
    q_source1_model = QModel(q_source1_encoder, q_source1_decoder)
    q_source1_model.to(device=config['device'], dtype=torch.float32)

    q_source2_encoder = Encoder(**config)
    q_source2_decoder = QDecoder(**config)
    q_source2_model = QModel(q_source2_encoder, q_source2_decoder)
    q_source2_model.to(device=config['device'], dtype=torch.float32)

    q_target1_encoder = Encoder(**config)
    q_target1_decoder = QDecoder(**config)
    q_target1_model = QModel(q_target1_encoder, q_target1_decoder)
    q_target1_model.requires_grad_(False)
    q_target1_model.to(device=config['device'], dtype=torch.float32)

    q_target2_encoder = Encoder(**config)
    q_target2_decoder = QDecoder(**config)
    q_target2_model = QModel(q_target2_encoder, q_target2_decoder)
    q_target2_model.requires_grad_(False)
    q_target2_model.to(device=config['device'], dtype=torch.float32)

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
    value_optimizer = construct(value_model)
    q1_optimizer = construct(q_source1_model)
    q2_optimizer = construct(q_source2_model)

    if config['is_main_process']:
        value_model, value_optimizer = amp.initialize(
            value_model, value_optimizer, opt_level=amp_optimization_level)
        q_source1_model, q1_optimizer = amp.initialize(
            q_source1_model, q1_optimizer, opt_level=amp_optimization_level)
        q_source2_model, q2_optimizer = amp.initialize(
            q_source2_model, q2_optimizer, opt_level=amp_optimization_level)
    else:
        value_model, value_optimizer = amp.initialize(
            value_model, value_optimizer, opt_level=amp_optimization_level,
            verbosity=0)
        q_source1_model, q1_optimizer = amp.initialize(
            q_source1_model, q1_optimizer, opt_level=amp_optimization_level,
            verbosity=0)
        q_source2_model, q2_optimizer = amp.initialize(
            q_source2_model, q2_optimizer, opt_level=amp_optimization_level,
            verbosity=0)

    if initial_model_prefix is not None:
        assert(not resume)
        assert(initial_model_prefix.exists())
        assert(initial_model_prefix.is_dir())
        value_model_state_dict = torch.load(value_snapshot_path)
        value_model_new_state_dict = {}
        for key, value in value_model_state_dict.items():
            new_key = re.sub('^module\.', '', key)
            value_model_new_state_dict[new_key] = value
        value_model.load_state_dict(value_model_new_state_dict)
        q_source1_state_dict = torch.load(q_source1_snapshot_path)
        q_source1_new_state_dict = {}
        for key, value in q_source1_state_dict.items():
            new_key = re.sub('^module\.', '', key)
            q_source1_new_state_dict[new_key] = value
        q_source1_model.load_state_dict(q_source1_new_state_dict)
        q_source2_state_dict = torch.load(q_source2_snapshot_path)
        q_source2_new_state_dict = {}
        for key, value in q_source2_state_dict.items():
            new_key = re.sub('^module\.', '', key)
            q_source2_new_state_dict[new_key] = value
        q_source2_model.load_state_dict(q_source2_new_state_dict)
        q_target1_state_dict = torch.load(q_target1_snapshot_path)
        q_target1_new_state_dict = {}
        for key, value in q_target1_state_dict.items():
            new_key = re.sub('^module\.', '', key)
            q_target1_new_state_dict[new_key] = value
        q_target1_model.load_state_dict(q_target1_new_state_dict)
        q_target2_state_dict = torch.load(q_target2_snapshot_path)
        q_target2_new_state_dict = {}
        for key, value in q_target2_state_dict.items():
            new_key = re.sub('^module\.', '', key)
            q_target2_new_state_dict[new_key] = value
        q_target2_model.load_state_dict(q_target2_new_state_dict)
        if value_optimizer_snapshot_path is not None:
            value_optimizer_state_dict = torch.load(
                value_optimizer_snapshot_path)
            value_optimizer_new_state_dict = {}
            for key, value in value_optimizer_state_dict.items():
                new_key = re.sub('^module\.', '', key)
                value_optimizer_new_state_dict[new_key] = value
            value_optimizer.load_state_dict(value_optimizer_new_state_dict)
        if q1_optimizer_snapshot_path is not None:
            q1_optimizer_state_dict = torch.load(q1_optimizer_snapshot_path)
            q1_optimizer_new_state_dict = {}
            for key, value in q1_optimizer_state_dict.items():
                new_key = re.sub('^module\.', '', key)
                q1_optimizer_new_state_dict[new_key] = value
            q1_optimizer.load_state_dict(q1_optimizer_new_state_dict)
        if q2_optimizer_snapshot_path is not None:
            q2_optimizer_state_dict = torch.load(q2_optimizer_snapshot_path)
            q2_optimizer_new_state_dict = {}
            for key, value in q2_optimizer_state_dict.items():
                new_key = re.sub('^module\.', '', key)
                q2_optimizer_new_state_dict[new_key] = value
            q2_optimizer.load_state_dict(q2_optimizer_new_state_dict)
        if amp_snapshot_path is not None:
            amp.load_state_dict(torch.load(amp_snapshot_path))

    if resume:
        assert(initial_model_prefix is None)
        assert(initial_model_index is None)
        assert(value_snapshot_path.exists())
        assert(value_snapshot_path.is_file())
        assert(q_source1_snapshot_path.exists())
        assert(q_source1_snapshot_path.is_file())
        assert(q_source2_snapshot_path.exists())
        assert(q_source2_snapshot_path.is_file())
        assert(q_target1_snapshot_path.exists())
        assert(q_target1_snapshot_path.is_file())
        assert(q_target2_snapshot_path.exists())
        assert(q_target2_snapshot_path.is_file())
        value_model_state_dict = torch.load(value_snapshot_path)
        value_model_new_state_dict = {}
        for key, value in value_model_state_dict.items():
            new_key = re.sub('^module\.', '', key)
            value_model_new_state_dict[new_key] = value
        value_model.load_state_dict(value_model_new_state_dict)
        q_source1_state_dict = torch.load(q_source1_snapshot_path)
        q_source1_new_state_dict = {}
        for key, value in q_source1_state_dict.items():
            new_key = re.sub('^module\.', '', key)
            q_source1_new_state_dict[new_key] = value
        q_source1_model.load_state_dict(q_source1_new_state_dict)
        q_source2_state_dict = torch.load(q_source2_snapshot_path)
        q_source2_new_state_dict = {}
        for key, value in q_source2_state_dict.items():
            new_key = re.sub('^module\.', '', key)
            q_source2_new_state_dict[new_key] = value
        q_source2_model.load_state_dict(q_source2_new_state_dict)
        q_target1_state_dict = torch.load(q_target1_snapshot_path)
        q_target1_new_state_dict = {}
        for key, value in q_target1_state_dict.items():
            new_key = re.sub('^module\.', '', key)
            q_target1_new_state_dict[new_key] = value
        q_target1_model.load_state_dict(q_target1_new_state_dict)
        q_target2_state_dict = torch.load(q_target2_snapshot_path)
        q_target2_new_state_dict = {}
        for key, value in q_target2_state_dict.items():
            new_key = re.sub('^module\.', '', key)
            q_target2_new_state_dict[new_key] = value
        q_target2_model.load_state_dict(q_target2_new_state_dict)

    if resume:
        assert(value_optimizer_snapshot_path.exists())
        assert(value_optimizer_snapshot_path.is_file())
        assert(q1_optimizer_snapshot_path.exists())
        assert(q1_optimizer_snapshot_path.is_file())
        assert(q2_optimizer_snapshot_path.exists())
        assert(q2_optimizer_snapshot_path.is_file())
        assert(amp_snapshot_path.exists())
        assert(amp_snapshot_path.is_file())
        value_optimizer.load_state_dict(
            torch.load(value_optimizer_snapshot_path))
        q1_optimizer.load_state_dict(torch.load(q1_optimizer_snapshot_path))
        q2_optimizer.load_state_dict(torch.load(q2_optimizer_snapshot_path))
        amp.load_state_dict(torch.load(amp_snapshot_path))

    if config['is_multiprocess']:
        init_process_group(backend='nccl')
        value_model = DistributedDataParallel(value_model)
        value_model = convert_syncbn_model(value_model)
        q_source1_model = DistributedDataParallel(q_source1_model)
        q_source1_model = convert_syncbn_model(q_source1_model)
        q_source2_model = DistributedDataParallel(q_source2_model)
        q_source2_model = convert_syncbn_model(q_source2_model)
        q_target1_model = DistributedDataParallel(q_target1_model)
        q_target1_model = convert_syncbn_model(q_target1_model)
        q_target2_model = DistributedDataParallel(q_target2_model)
        q_target2_model = convert_syncbn_model(q_target2_model)

    config['value_model'] = value_model
    config['q_source1_model'] = q_source1_model
    config['q_source2_model'] = q_source2_model
    config['q_target1_model'] = q_target1_model
    config['q_target2_model'] = q_target2_model
    config['value_optimizer'] = value_optimizer
    config['q1_optimizer'] = q1_optimizer
    config['q2_optimizer'] = q2_optimizer

    with SummaryWriter(log_dir=config['tensorboard_path']) as writer:
        _training(**config, writer=writer)


if __name__ == '__main__':
    main()
    sys.exit(0)
