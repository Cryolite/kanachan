#!/usr/bin/env python3

import re
import datetime
import math
from pathlib import Path
import os
from argparse import ArgumentParser
import logging
import sys
from typing import Tuple, Optional, Callable
import torch
from torch import backends
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer, RAdam
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, all_reduce, all_gather, barrier
from torch.utils.tensorboard.writer import SummaryWriter
from apex import amp
from apex.parallel import convert_syncbn_model
from apex.optimizers import FusedAdam, FusedSGD, FusedLAMB
from kanachan.training.constants import NUM_TYPES_OF_SPARSE_FEATURES, MAX_NUM_ACTION_CANDIDATES
from kanachan.training.common import initialize_logging, Dataset
from kanachan.training.iql.iterator_adaptor import IteratorAdaptor
from kanachan.training.bert.encoder import Encoder
from kanachan.training.iql.qv_model import QVDecoder, QVModel
from kanachan.training.iql.q_model import QModel
from kanachan.training.mtadam import MTAdam


SnapshotWriter = Callable[
    [QModel, QModel, QModel, QModel, Optimizer, Optimizer, Optional[int]],
    None
]


def _training(
        *, is_multiprocess: bool, world_size: Optional[int], rank: Optional[int],
        is_main_process: bool, training_data: Path, num_workers: int, device: str,
        qv1_source_model: QVModel, qv2_source_model: QVModel, qv1_target_model: QVModel,
        qv2_target_model: QVModel, reward_plugin: Path, discount_factor: float, expectile: float,
        target_update_interval: int, target_update_rate: float, batch_size: int,
        v_loss_scaling: float, gradient_accumulation_steps: int, max_gradient_norm: float,
        qv1_optimizer: Optimizer, qv2_optimizer: Optimizer, snapshot_interval: int,
        num_samples: int, writer: SummaryWriter, snapshot_writer: SnapshotWriter, **kwargs) -> None:
    start_time = datetime.datetime.now()

    # Load the reward plugin.
    with open(reward_plugin, encoding='UTF-8') as file_pointer:
        exec(file_pointer.read(), globals())

    # Prepare the training data loader. Note that this data loader must iterate
    # the training data set only once.
    iterator_adaptor = lambda path: IteratorAdaptor(path, get_reward)
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
        if device != 'cpu':
            annotation = [t.cuda() for t in annotation]

        local_batch_size = len(annotation[0])

        world_batch_size = local_batch_size
        if is_multiprocess:
            assert world_size is not None
            assert rank is not None
            world_batch_size *= world_size

        if skipped_samples < num_samples:
            if is_multiprocess:
                barrier()
            skipped_samples += world_batch_size
            continue

        # Compute the Q target value.
        with torch.no_grad():
            def get_q_target(qv_target_model: QVModel) -> Tuple[torch.Tensor, float]:
                q_target, _ = qv_target_model(annotation[:4])
                assert q_target.dim() == 2
                assert q_target.size(0) == local_batch_size
                assert q_target.size(1) == MAX_NUM_ACTION_CANDIDATES
                q_target = q_target[torch.arange(local_batch_size), annotation[4]]
                assert q_target.dim() == 1
                assert q_target.size(0) == local_batch_size

                if not is_multiprocess:
                    q_target_gathered = q_target
                else:
                    q_target_gathered = [torch.zeros_like(q_target) for i in range(world_size)]
                    all_gather(q_target_gathered, q_target)
                    q_target_gathered = torch.cat(q_target_gathered)
                assert q_target_gathered.dim() == 1
                assert q_target_gathered.size(0) == world_batch_size
                q_batch_mean = torch.mean(q_target_gathered)

                return (q_target, q_batch_mean.item())

            q1_target, q1_batch_mean = get_q_target(qv1_target_model)
            q2_target, q2_batch_mean = get_q_target(qv2_target_model)

            q_target = torch.minimum(q1_target, q2_target)
            q_target = q_target.detach()

        reward = annotation[9]

        def backward(qv_source_model: QVModel, qv_optimizer: Optimizer) -> Tuple[float, float]:
            q, v = qv_source_model(annotation[:4])
            assert q.dim() == 2
            assert q.size(0) == local_batch_size
            assert q.size(1) == MAX_NUM_ACTION_CANDIDATES
            assert v.dim() == 1
            assert v.size(0) == local_batch_size
            q = q[torch.arange(local_batch_size), annotation[4]]

            if not is_multiprocess:
                v_gathered = v
            else:
                v_gathered = [torch.zeros_like(v) for i in range(world_size)]
                all_gather(v_gathered, v)
                v_gathered = torch.cat(v_gathered)
            assert v_gathered.dim() == 1
            assert v_gathered.size(0) == world_batch_size
            v_batch_mean = torch.mean(v_gathered).item()

            _, vv = qv_source_model(annotation[5:9])
            is_terminal_state = (annotation[5][:, 0] == NUM_TYPES_OF_SPARSE_FEATURES)
            vv = torch.where(is_terminal_state, torch.zeros_like(vv), vv)
            assert vv.dim() == 1
            assert vv.size(0) == local_batch_size

            q_loss = torch.square(reward + discount_factor * vv - q)
            q_loss = torch.mean(q_loss)
            v_loss = q_target - v
            v_loss = torch.where(
                v_loss < 0.0, (1.0 - expectile) * torch.square(v_loss),
                expectile * torch.square(v_loss))
            v_loss = torch.mean(v_loss)
            qv_loss = q_loss + v_loss_scaling * v_loss

            qv_batch_loss = qv_loss.detach().clone()
            if is_multiprocess:
                all_reduce(qv_batch_loss)
                qv_batch_loss /= world_size
            qv_batch_loss = qv_batch_loss.item()

            if math.isnan(qv_loss.item()):
                raise RuntimeError('QV loss becomes NaN.')
            qv_loss /= gradient_accumulation_steps
            if not isinstance(qv_optimizer, MTAdam):
                with amp.scale_loss(qv_loss, qv_optimizer) as scaled_qv_loss:
                    scaled_qv_loss.backward()

            return (q_loss, v_loss, v_batch_mean, qv_batch_loss)

        # Backprop for the QV1 source model.
        q1_loss, v1_loss, v1_batch_mean, qv1_batch_loss = backward(qv1_source_model, qv1_optimizer)

        # Backprop for the QV2 source model.
        q2_loss, v2_loss, v2_batch_mean, qv2_batch_loss = backward(qv2_source_model, qv2_optimizer)

        num_samples += world_batch_size
        batch_count += 1

        if batch_count % gradient_accumulation_steps == 0:
            if not isinstance(qv1_optimizer, MTAdam):
                qv1_gradient = [
                    torch.flatten(x.grad) for x in amp.master_params(qv1_optimizer)
                    if x.grad is not None
                ]
                qv1_gradient = torch.cat(qv1_gradient)
                qv1_gradient_norm = torch.linalg.vector_norm(qv1_gradient).item()
                if qv1_gradient_norm > max_gradient_norm:
                    nn.utils.clip_grad_norm_(
                        amp.master_params(qv1_optimizer), max_gradient_norm,
                        error_if_nonfinite=False)
                qv1_optimizer.step()
            else:
                qv1_optimizer.step((q1_loss, v1_loss), (1.0, 1.0), None)
                qv1_gradient = [
                    torch.flatten(p.grad) for p in qv1_source_model.parameters() if p.grad is not None
                ]
                qv1_gradient = torch.cat(qv1_gradient)
                qv1_gradient_norm = torch.linalg.vector_norm(qv1_gradient).item()
            qv1_optimizer.zero_grad()

            if not isinstance(qv2_optimizer, MTAdam):
                qv2_gradient = [
                    torch.flatten(x.grad) for x in amp.master_params(qv2_optimizer)
                    if x.grad is not None
                ]
                qv2_gradient = torch.cat(qv2_gradient)
                qv2_gradient_norm = torch.linalg.vector_norm(qv2_gradient).item()
                if qv2_gradient_norm > max_gradient_norm:
                    nn.utils.clip_grad_norm_(
                        amp.master_params(qv2_optimizer), max_gradient_norm,
                        error_if_nonfinite=False)
                qv2_optimizer.step()
            else:
                qv2_optimizer.step((q2_loss, v2_loss), (1.0, 1.0), None)
                qv2_gradient = [
                    torch.flatten(p.grad) for p in qv2_source_model.parameters() if p.grad is not None
                ]
                qv2_gradient = torch.cat(qv2_gradient)
                qv2_gradient_norm = torch.linalg.vector_norm(qv2_gradient).item()
            qv2_optimizer.zero_grad()

            if batch_count % (gradient_accumulation_steps * target_update_interval) == 0:
                if not isinstance(qv1_optimizer, MTAdam):
                    param1_source_iter = amp.master_params(qv1_optimizer)
                else:
                    param1_source_iter = qv1_source_model.parameters()
                param1_target_iter = qv1_target_model.parameters()
                for param1_source, param1_target in zip(param1_source_iter, param1_target_iter):
                    param1_target *= (1.0 - target_update_rate)
                    param1_target += target_update_rate * param1_source

                if not isinstance(qv2_optimizer, MTAdam):
                    param2_source_iter = amp.master_params(qv2_optimizer)
                else:
                    param2_source_iter = qv2_source_model.parameters()
                param2_target_iter = qv2_target_model.parameters()
                for param2_source, param2_target in zip(param2_source_iter, param2_target_iter):
                    param2_target *= (1.0 - target_update_rate)
                    param2_target += target_update_rate * param2_source

            logging.info(
                'sample = %s, QV1 loss = %s, QV2 loss = %s,'
                ' QV1 gradient norm = %s, QV2 gradient norm = %s',
                num_samples, qv1_batch_loss, qv2_batch_loss, qv1_gradient_norm, qv2_gradient_norm)
            if is_main_process:
                writer.add_scalars(
                    'Q', { 'Q1': q1_batch_mean, 'Q2': q2_batch_mean },
                    num_samples)
                writer.add_scalars(
                    'V', { 'V1': v1_batch_mean, 'V2': v2_batch_mean },
                    num_samples)
                writer.add_scalars(
                    'QV Loss', { 'QV1': qv1_batch_loss, 'QV2': qv2_batch_loss },
                    num_samples)
                writer.add_scalars(
                    'QV Gradient Norm',
                    { 'QV1': qv1_gradient_norm, 'QV2': qv2_gradient_norm },
                    num_samples)
        else:
            logging.info(
                'sample = %s, QV1 loss = %s, QV2 loss = %s',
                num_samples, qv1_batch_loss, qv2_batch_loss)
            if is_main_process:
                writer.add_scalars(
                    'Q', { 'Q1': q1_batch_mean, 'Q2': q2_batch_mean },
                    num_samples)
                writer.add_scalars(
                    'V', { 'V1': v1_batch_mean, 'V2': v2_batch_mean },
                    num_samples)
                writer.add_scalars(
                    'QV Loss', { 'QV1': qv1_batch_loss, 'QV2': qv2_batch_loss },
                    num_samples)

        if is_main_process and last_snapshot is not None and num_samples - last_snapshot >= snapshot_interval:
            snapshot_writer(
                qv1_source_model, qv2_source_model,
                qv1_target_model, qv2_target_model,
                qv1_optimizer, qv2_optimizer, num_samples)
            last_snapshot = num_samples

    elapsed_time = datetime.datetime.now() - start_time
    logging.info('A training has finished (elapsed time = %s).', elapsed_time)

    if is_main_process:
        snapshot_writer(
            qv1_source_model, qv2_source_model,
            qv1_target_model, qv2_target_model,
            qv1_optimizer, qv2_optimizer)


def _main() -> None:
    ap = ArgumentParser(description='Train by implicit Q-learning (IQL)')
    ap_data = ap.add_argument_group(title='Data')
    ap_data.add_argument(
        '--training-data', type=Path, required=True,
        help='path to training data', metavar='PATH')
    ap_data.add_argument(
        '--num-workers', type=int,
        help='# of worker processes in data loading (default to `2` for multiprocessing, `0` otherwise)',
        metavar='NWORKERS')

    ap_device = ap.add_argument_group(title='Device')
    ap_device.add_argument('--device', help='device', metavar='DEV')
    ap_device.add_argument(
        '--amp-optimization-level', default='O2', choices=('O0', 'O1', 'O2', 'O3',),
        help='optimization level for automatic mixed precision (defaults to `O2`)')

    ap_model = ap.add_argument_group(title='Model')
    ap_model.add_argument('--model-preset', choices=('base', 'large',))
    ap_model.add_argument('--dimension', type=int, help='embedding dimension', metavar='DIM')
    ap_model.add_argument('--num-heads', type=int, help='number of heads', metavar='NHEADS')
    ap_model.add_argument(
        '--dim-feedforward', type=int,
        help='dimension of the feedforward network in each layer (defaults to `4 * DIM`)',
        metavar='DIM_FEEDFORWARD')
    ap_model.add_argument('--num-layers', type=int, help='number of layers', metavar='NLAYERS')
    ap_model.add_argument(
        '--dim-final-feedforward', type=int,
        help='dimension of the final feedforward network (defaults to `DIM_FEEDFORWARD`)',
        metavar='DIM_FINAL_FEEDFORWARD')
    ap_model.add_argument(
        '--activation-function', default='gelu', choices=('relu', 'gelu',),
        help='activation function for the feedforward networks (defaults to `gelu`)',
        metavar='ACTIVATION')
    ap_model.add_argument(
        '--dropout', default=0.1, type=float, help='dropout rate (defaults to `0.1`)',
        metavar='DROPOUT')
    ap_model.add_argument(
        '--initial-encoder', type=Path,
        help='path to initial encoder; mutually exclusive to `--resume`',
        metavar='PATH')
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
        '--reward-plugin', type=Path, required=True,
        help='path to reward plugin', metavar='PATH')
    ap_training.add_argument(
        '--checkpointing', action='store_true', help='enable checkpointing')
    ap_training.add_argument(
        '--discount-factor', type=float, required=True, metavar='GAMMA')
    ap_training.add_argument(
        '--expectile', type=float, required=True, metavar='TAU')
    ap_training.add_argument(
        '--v-loss-scaling', default=1.0, type=float, metavar='SCALING')
    ap_training.add_argument(
        '--batch-size', type=int, required=True,
        help='batch size', metavar='N')
    ap_training.add_argument(
        '--gradient-accumulation-steps', default=1, type=int,
        help='# of steps for gradient accumulation (defaults to `1`)',
        metavar='NSTEPS')
    ap_training.add_argument(
        '--max-gradient-norm', default=math.inf, type=float,
        help='norm threshold for gradient clipping (defaults to `+INF`)',
        metavar='NORM')
    ap_training.add_argument(
        '--optimizer', default='lamb',
        choices=('sgd', 'adam', 'radam', 'mtadam', 'lamb',),
        help='optimizer (defaults to `lamb`)')
    ap_training.add_argument(
        '--momentum', default=0.9, type=float,
        help='momentum factor; only meaningful for SGD (defaults to `0.9`)',
        metavar='MOMENTUM')
    ap_training.add_argument(
        '--learning-rate', type=float,
        help='learning rate (defaults to `0.1` for `sgd`, `0.001` for `adam`, `radam`, `mtadam`, and `lamb`)',
        metavar='LR')
    ap_training.add_argument(
        '--epsilon', type=float,
        help='epsilon parameter; only meaningful for Adam, RAdam, and LAMB\
 (defaults to `1.0e-8` for Adam and RAdam, `1.0e-6` for LAMB)',
        metavar='EPS')
    ap_training.add_argument(
        '--target-update-interval', type=int, required=True, metavar='N')
    ap_training.add_argument(
        '--target-update-rate', type=float, required=True, metavar='ALPHA')

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

    if is_multiprocess:
        if config.num_workers is None:
            config.num_workers = 2
        if config.num_workers < 0:
            raise RuntimeError(f'{config.num_workers}: An invalid number of workers.')
        if config.num_workers == 0:
            raise RuntimeError(
                f'{config.num_workers}: An invalid number of workers for multiprocessing.')
    else:
        if config.num_workers is None:
            config.num_workers = 0
        if config.num_workers < 0:
            raise RuntimeError(f'{config.num_workers}: An invalid number of workers.')

    if config.device is not None:
        match = re.search('^(?:cpu|cuda(\\d*))$', config.device)
        if match is None:
            raise RuntimeError(f'{config.device}: An invalid device.')
        if is_multiprocess and match[1] != '':
            raise RuntimeError('Must not specify any device number in multi-process mode.')
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
            f'{config.dim_final_feedforward}: '
            'An invalid dimension of the final feedforward network.')

    if config.dropout < 0.0 or 1.0 <= config.dropout:
        raise RuntimeError(f'{config.dropout}: An invalid value for `--dropout`.')

    if config.initial_encoder is not None and not config.initial_encoder.exists():
        raise RuntimeError(f'{config.initial_encoder}: Does not exist.')
    if config.initial_encoder is not None and not config.initial_encoder.is_file():
        raise RuntimeError(f'{config.initial_encoder}: Not a file')
    if config.initial_encoder is not None and config.resume:
        raise RuntimeError('`--initial-encoder` conflicts with `--resume`.')
    initial_encoder = config.initial_encoder

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
        raise RuntimeError('`--initial-model-index` conflicts with `--resume`.')
    initial_model_index = config.initial_model_index

    num_samples = 0
    if initial_model_prefix is not None:
        if initial_encoder is not None:
            raise RuntimeError('`--initial-model-prefix` conflicts with `--initial-encoder`.')
        assert not config.resume
        if initial_model_index is None:
            for child in os.listdir(initial_model_prefix):
                match = re.search(
                    '^(?:qv[12]-source|qv[12]-target|qv[12]-optimizer)\\.(\\d+)\\.pth$', child)
                if match is None:
                    continue
                if initial_model_index is None:
                    initial_model_index = 0
                if int(match[1]) > initial_model_index:
                    initial_model_index = int(match[1])
        if initial_model_index is None:
            raise RuntimeError(f'{initial_model_prefix}: No model found.')
        num_samples = initial_model_index
        qv1_source_snapshot_path = initial_model_prefix / f'qv1-source.{initial_model_index}.pth'
        if not qv1_source_snapshot_path.exists():
            raise RuntimeError(f'{qv1_source_snapshot_path}: Does not exist.')
        if not qv1_source_snapshot_path.is_file():
            raise RuntimeError(f'{qv1_source_snapshot_path}: Not a file.')
        qv2_source_snapshot_path = initial_model_prefix / f'qv2-source.{initial_model_index}.pth'
        if not qv2_source_snapshot_path.exists():
            raise RuntimeError(f'{qv2_source_snapshot_path}: Does not exist.')
        if not qv2_source_snapshot_path.is_file():
            raise RuntimeError(f'{qv2_source_snapshot_path}: Not a file.')
        qv1_target_snapshot_path = initial_model_prefix / f'qv1-target.{initial_model_index}.pth'
        if not qv1_target_snapshot_path.exists():
            raise RuntimeError(f'{qv1_target_snapshot_path}: Does not exist.')
        if not qv1_target_snapshot_path.is_file():
            raise RuntimeError(f'{qv1_target_snapshot_path}: Not a file.')
        qv2_target_snapshot_path = initial_model_prefix / f'qv2-target.{initial_model_index}.pth'
        if not qv2_target_snapshot_path.exists():
            raise RuntimeError(f'{qv2_target_snapshot_path}: Does not exist.')
        if not qv2_target_snapshot_path.is_file():
            raise RuntimeError(f'{qv2_target_snapshot_path}: Not a file.')
        qv1_optimizer_snapshot_path \
            = initial_model_prefix / f'qv1-optimizer.{initial_model_index}.pth'
        if not qv1_optimizer_snapshot_path.is_file():
            qv1_optimizer_snapshot_path = None
        qv2_optimizer_snapshot_path \
            = initial_model_prefix / f'qv2-optimizer.{initial_model_index}.pth'
        if not qv2_optimizer_snapshot_path.is_file():
            qv2_optimizer_snapshot_path = None
        amp_snapshot_path = initial_model_prefix / f'amp.{initial_model_index}.pth'
        if not amp_snapshot_path.is_file():
            amp_snapshot_path = None

    if config.reward_plugin is not None and not config.reward_plugin.exists():
        raise RuntimeError(f'{config.reward_plugin}: Does not exist.')
    if config.reward_plugin is not None and not config.reward_plugin.is_file():
        raise RuntimeError(f'{config.reward_plugin}: Not a file.')

    if config.discount_factor <= 0.0 or 1.0 < config.discount_factor:
        raise RuntimeError(
            f'{config.discount_factor}: An invalid value for `--discount-factor`.')

    if config.expectile <= 0.0 or 1.0 <= config.expectile:
        raise RuntimeError(
            f'{config.expectile}: An invalid value for `--expectile`.')

    if config.v_loss_scaling < 0.0:
        raise RuntimeError(f'{config.v_loss_scaling}: An invalid value for `--v-loss-scaling`.')

    if config.batch_size < 1:
        raise RuntimeError(
            f'{config.batch_size}: An invalid value for `--batch-size`.')

    if config.gradient_accumulation_steps < 1:
        raise RuntimeError(
            f'{config.gradient_accumulation_steps}: '
            'An invalid value for `--gradient-accumulation`.')

    if config.max_gradient_norm <= 0.0:
        raise RuntimeError(
            f'{config.max_gradient_norm}: An invalid value for `--max-gradient-norm`.')

    if config.optimizer == 'sgd':
        if config.learning_rate is None:
            learning_rate = 0.1
        else:
            learning_rate = config.learning_rate
    elif config.optimizer in ('adam', 'radam', 'mtadam', 'lamb',):
        if config.learning_rate is None:
            learning_rate = 0.001
        else:
            learning_rate = config.learning_rate
    else:
        raise NotImplementedError(config.optimizer)
    if learning_rate <= 0.0:
        raise RuntimeError(f'{learning_rate}: An invalid value for `--learning-rate`.')

    if config.momentum < 0.0 or 1.0 <= config.momentum:
        raise RuntimeError(
            f'{config.momentum}: An invalid value for `--momentum`.')
    momentum = config.momentum

    if config.epsilon is None:
        if config.optimizer in ('adam', 'radam', 'mtadam',):
            epsilon = 1.0e-8
        elif config.optimizer == 'lamb':
            epsilon = 1.0e-6
        else:
            epsilon = None
    else:
        epsilon = config.epsilon
    if epsilon is not None and epsilon <= 0.0:
        raise RuntimeError(f'{epsilon}: An invalid value for `--epsilon`.')

    if config.target_update_interval <= 0:
        raise RuntimeError(
            f'{config.target_update_interval}: An invalid value for `--target-update-interval`.')

    if config.target_update_rate <= 0.0 or 1.0 < config.target_update_rate:
        raise RuntimeError(
            f'{config.target_update_rate}: An invalid value for `--target-update-rate`.')

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

    if resume:
        assert initial_encoder is None
        assert initial_model_prefix is None
        assert initial_model_index is None
        if not snapshots_path.exists():
            raise RuntimeError(f'{snapshots_path}: Does not exist.')
        for child in os.listdir(snapshots_path):
            match = re.search(
                '^(?:qv[12]-source|qv[12]-target|qv[12]-optimizer|amp)\\.(\\d+)\\.pth$', child)
            if match is None:
                continue
            if int(match[1]) > num_samples:
                num_samples = int(match[1])
        qv1_source_snapshot_path = snapshots_path / f'qv1-source.{num_samples}.pth'
        if not qv1_source_snapshot_path.exists():
            raise RuntimeError(f'{qv1_source_snapshot_path}: Does not exist.')
        if not qv1_source_snapshot_path.is_file():
            raise RuntimeError(f'{qv1_source_snapshot_path}: Not a file.')
        qv2_source_snapshot_path = snapshots_path / f'qv2-source.{num_samples}.pth'
        if not qv2_source_snapshot_path.exists():
            raise RuntimeError(f'{qv2_source_snapshot_path}: Does not exist.')
        if not qv2_source_snapshot_path.is_file():
            raise RuntimeError(f'{qv2_source_snapshot_path}: Not a file.')
        qv1_target_snapshot_path = snapshots_path / f'qv1-target.{num_samples}.pth'
        if not qv1_target_snapshot_path.exists():
            raise RuntimeError(f'{qv1_target_snapshot_path}: Does not exist.')
        if not qv1_target_snapshot_path.is_file():
            raise RuntimeError(f'{qv1_target_snapshot_path}: Not a file.')
        qv2_target_snapshot_path = snapshots_path / f'qv2-target.{num_samples}.pth'
        if not qv2_target_snapshot_path.exists():
            raise RuntimeError(f'{qv2_target_snapshot_path}: Does not exist.')
        if not qv2_target_snapshot_path.is_file():
            raise RuntimeError(f'{qv2_target_snapshot_path}: Not a file.')
        qv1_optimizer_snapshot_path = snapshots_path / f'qv1-optimizer.{num_samples}.pth'
        if not qv1_optimizer_snapshot_path.exists():
            raise RuntimeError(f'{qv1_optimizer_snapshot_path}: Does not exist.')
        if not qv1_optimizer_snapshot_path.is_file():
            raise RuntimeError(f'{qv1_optimizer_snapshot_path}: Not a file.')
        qv2_optimizer_snapshot_path = snapshots_path / f'qv2-optimizer.{num_samples}.pth'
        if not qv2_optimizer_snapshot_path.exists():
            raise RuntimeError(f'{qv2_optimizer_snapshot_path}: Does not exist.')
        if not qv2_optimizer_snapshot_path.is_file():
            raise RuntimeError(f'{qv2_optimizer_snapshot_path}: Not a file.')
        amp_snapshot_path = snapshots_path / f'amp.{num_samples}.pth'
        if not amp_snapshot_path.exists():
            raise RuntimeError(f'{amp_snapshot_path}: Does not exist.')
        if not amp_snapshot_path.is_file():
            raise RuntimeError(f'{amp_snapshot_path}: Not a file.')

    experiment_path.mkdir(parents=True, exist_ok=True)
    initialize_logging(experiment_path, rank)

    if world_size is None:
        assert rank is None
        logging.info('World size: N/A (single process)')
        logging.info('Process rank: N/A (single process)')
    else:
        assert rank is not None
        logging.info('World size: %s', world_size)
        logging.info('Process rank: %s', rank)
    logging.info('Training data: %s', config.training_data)
    if num_samples > 0:
        logging.info('# of training samples consumed so far: %s', num_samples)
    logging.info('# of workers: %s', config.num_workers)
    logging.info('Device: %s', device)
    if backends.cudnn.is_available():
        logging.info('cuDNN: available')
        backends.cudnn.benchmark = True
    else:
        logging.info('cuDNN: N/A')
    logging.info('AMP optimization level: %s', amp_optimization_level)
    logging.info('Embedding dimension: %s', config.dimension)
    logging.info('# of heads: %s', config.num_heads)
    logging.info(
        'Dimension of the feedforward network in each layer: %s', config.dim_feedforward)
    logging.info('# of layers: %s', config.num_layers)
    logging.info(
        'Dimension of the final feedforward network: %s', config.dim_final_feedforward)
    logging.info('Activation function: %s', config.activation_function)
    logging.info('Dropout: %s', config.dropout)
    if initial_encoder is not None:
        logging.info('Initial encoder: %s', initial_encoder)
    elif initial_model_prefix is not None:
        logging.info('Initial model prefix: %s', initial_model_prefix)
        if initial_model_index is not None:
            logging.info('Initlal model index: %s', initial_model_index)
        else:
            logging.info('Initial model index: (latest one)')
    elif not resume:
        logging.info('Initial model: (initialized randomly)')
    logging.info('Reward plugin: %s', config.reward_plugin)
    logging.info('Checkpointing: %s', config.checkpointing)
    logging.info('Discount factor: %s', config.discount_factor)
    logging.info('Expectile: %s', config.expectile)
    logging.info('V loss scaling: %s', config.v_loss_scaling)
    if world_size is None:
        logging.info('Batch size: %s', config.batch_size)
    else:
        logging.info('Local batch size: %s', config.batch_size)
        logging.info('World batch size: %s', config.batch_size * world_size)
    logging.info('# of steps for gradient accumulation: %s', config.gradient_accumulation_steps)
    if world_size is None:
        logging.info(
            'Virtual batch size: %s', config.batch_size * config.gradient_accumulation_steps)
    else:
        logging.info(
            'Virtual batch size: %s',
            config.batch_size * world_size * config.gradient_accumulation_steps)
    logging.info('Norm threshold for gradient clipping: %s', config.max_gradient_norm)
    logging.info('Optimizer: %s', config.optimizer)
    logging.info('Learning rate: %s', learning_rate)
    if config.optimizer == 'sgd':
        logging.info('Momentum factor: %s', momentum)
    if config.optimizer in ('adam', 'radam', 'mtadam', 'lamb',):
        logging.info('Epsilon parameter: %s', epsilon)
    logging.info('Target update interval: %s', config.target_update_interval)
    logging.info('Target update rate: %s', config.target_update_rate)
    if initial_model_prefix is not None:
        assert initial_encoder is None
        assert not resume
        logging.info('Initial QV1 source network snapshot: %s', qv1_source_snapshot_path)
        logging.info('Initial QV2 source network snapshot: %s', qv2_source_snapshot_path)
        logging.info('Initial QV1 target network snapshot: %s', qv1_target_snapshot_path)
        logging.info('Initial QV2 target network snapshot: %s', qv2_target_snapshot_path)
        if qv1_optimizer_snapshot_path is not None:
            logging.info('Initial QV1 optimizer snapshot: %s', qv1_optimizer_snapshot_path)
        if qv2_optimizer_snapshot_path is not None:
            logging.info('Initial QV2 optimizer snapshot: %s', qv2_optimizer_snapshot_path)
        if amp_snapshot_path is not None:
            logging.info('Initial AMP snapshot: %s', amp_snapshot_path)
    if resume:
        assert initial_encoder is None
        assert initial_model_prefix is None
        logging.info('Resume from %s', experiment_path)
        logging.info('QV1 source network snapshot: %s', qv1_source_snapshot_path)
        logging.info('QV2 source network snapshot: %s', qv2_source_snapshot_path)
        logging.info('QV1 target network snapshot: %s', qv1_target_snapshot_path)
        logging.info('QV2 target network snapshot: %s', qv2_target_snapshot_path)
        logging.info('QV1 source network optimizer snapshot: %s', qv1_optimizer_snapshot_path)
        logging.info('QV2 source network optimizer snapshot: %s', qv2_optimizer_snapshot_path)
        logging.info('AMP snapshot: %s', amp_snapshot_path)
    else:
        logging.info('Experiment output: %s', experiment_path)
    if config.snapshot_interval == 0:
        logging.info('Snapshot interval: N/A')
    else:
        logging.info('Snapshot interval: %s', config.snapshot_interval)

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
        'reward_plugin': config.reward_plugin,
        'checkpointing': config.checkpointing,
        'discount_factor': config.discount_factor,
        'expectile': config.expectile,
        'v_loss_scaling': config.v_loss_scaling,
        'batch_size': config.batch_size,
        'gradient_accumulation_steps': config.gradient_accumulation_steps,
        'max_gradient_norm': config.max_gradient_norm,
        'optimizer': config.optimizer,
        'target_update_interval': config.target_update_interval,
        'target_update_rate': config.target_update_rate,
        'experiment_name': experiment_name,
        'experiment_path': experiment_path,
        'tensorboard_path': tensorboard_path,
        'snapshot_interval': config.snapshot_interval,
        'num_samples': num_samples,
    }

    qv1_source_encoder = Encoder(**config)
    qv1_source_decoder = QVDecoder(**config)
    qv1_source_model = QVModel(qv1_source_encoder, qv1_source_decoder)
    qv1_source_model.to(device=config['device'], dtype=torch.float32)

    qv2_source_encoder = Encoder(**config)
    qv2_source_decoder = QVDecoder(**config)
    qv2_source_model = QVModel(qv2_source_encoder, qv2_source_decoder)
    qv2_source_model.to(device=config['device'], dtype=torch.float32)

    qv1_target_encoder = Encoder(**config)
    qv1_target_decoder = QVDecoder(**config)
    qv1_target_model = QVModel(qv1_target_encoder, qv1_target_decoder)
    qv1_target_model.requires_grad_(False)
    qv1_target_model.to(device=config['device'], dtype=torch.float32)

    qv2_target_encoder = Encoder(**config)
    qv2_target_decoder = QVDecoder(**config)
    qv2_target_model = QVModel(qv2_target_encoder, qv2_target_decoder)
    qv2_target_model.requires_grad_(False)
    qv2_target_model.to(device=config['device'], dtype=torch.float32)

    if config['optimizer'] == 'sgd':
        construct = lambda model: FusedSGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif config['optimizer'] == 'adam':
        construct = lambda model: FusedAdam(model.parameters(), lr=learning_rate, eps=epsilon)
    elif config['optimizer'] == 'radam':
        construct = lambda model: RAdam(model.parameters(), lr=learning_rate, eps=epsilon)
    elif config['optimizer'] == 'mtadam':
        construct = lambda model: MTAdam(model.parameters(), lr=learning_rate, eps=epsilon)
    elif config['optimizer'] == 'lamb':
        construct = lambda model: FusedLAMB(model.parameters(), lr=learning_rate, eps=epsilon)
    else:
        raise NotImplementedError(config['optimizer'])
    qv1_optimizer = construct(qv1_source_model)
    qv2_optimizer = construct(qv2_source_model)

    if config['optimizer'] != 'mtadam':
        if config['is_main_process']:
            qv1_source_model, qv1_optimizer = amp.initialize(
                qv1_source_model, qv1_optimizer, opt_level=amp_optimization_level)
            qv2_source_model, qv2_optimizer = amp.initialize(
                qv2_source_model, qv2_optimizer, opt_level=amp_optimization_level)
        else:
            qv1_source_model, qv1_optimizer = amp.initialize(
                qv1_source_model, qv1_optimizer, opt_level=amp_optimization_level, verbosity=0)
            qv2_source_model, qv2_optimizer = amp.initialize(
                qv2_source_model, qv2_optimizer, opt_level=amp_optimization_level, verbosity=0)

    if initial_encoder is not None:
        assert initial_model_prefix is None
        assert not resume
        assert initial_encoder.exists()
        assert initial_encoder.is_file()
        initial_encoder_state_dict = torch.load(initial_encoder, map_location='cpu')
        initial_encoder_new_state_dict = {}
        for key, value in initial_encoder_state_dict.items():
            new_key = re.sub('^module\\.', '', key)
            initial_encoder_new_state_dict[new_key] = value
        qv1_source_encoder.load_state_dict(initial_encoder_new_state_dict)
        qv1_source_encoder.cuda()
        qv2_source_encoder.load_state_dict(initial_encoder_new_state_dict)
        qv2_source_encoder.cuda()
        qv1_target_encoder.load_state_dict(initial_encoder_new_state_dict)
        qv1_target_encoder.cuda()
        qv2_target_encoder.load_state_dict(initial_encoder_new_state_dict)
        qv2_target_encoder.cuda()

    if initial_model_prefix is not None:
        assert initial_encoder is None
        assert not resume
        assert initial_model_prefix.exists()
        assert initial_model_prefix.is_dir()
        qv1_source_state_dict = torch.load(qv1_source_snapshot_path, map_location='cpu')
        qv1_source_new_state_dict = {}
        for key, value in qv1_source_state_dict.items():
            new_key = re.sub('^module\\.', '', key)
            qv1_source_new_state_dict[new_key] = value
        qv1_source_model.load_state_dict(qv1_source_new_state_dict)
        qv1_source_model.cuda()
        qv2_source_state_dict = torch.load(qv2_source_snapshot_path, map_location='cpu')
        qv2_source_new_state_dict = {}
        for key, value in qv2_source_state_dict.items():
            new_key = re.sub('^module\\.', '', key)
            qv2_source_new_state_dict[new_key] = value
        qv2_source_model.load_state_dict(qv2_source_new_state_dict)
        qv2_source_model.cuda()
        qv1_target_state_dict = torch.load(qv1_target_snapshot_path, map_location='cpu')
        qv1_target_new_state_dict = {}
        for key, value in qv1_target_state_dict.items():
            new_key = re.sub('^module\\.', '', key)
            qv1_target_new_state_dict[new_key] = value
        qv1_target_model.load_state_dict(qv1_target_new_state_dict)
        qv1_target_model.cuda()
        qv2_target_state_dict = torch.load(qv2_target_snapshot_path, map_location='cpu')
        qv2_target_new_state_dict = {}
        for key, value in qv2_target_state_dict.items():
            new_key = re.sub('^module\\.', '', key)
            qv2_target_new_state_dict[new_key] = value
        qv2_target_model.load_state_dict(qv2_target_new_state_dict)
        qv2_target_model.cuda()
        if qv1_optimizer_snapshot_path is not None:
            qv1_optimizer.load_state_dict(
                torch.load(qv1_optimizer_snapshot_path, map_location='cpu'))
        if qv2_optimizer_snapshot_path is not None:
            qv2_optimizer.load_state_dict(
                torch.load(qv2_optimizer_snapshot_path, map_location='cpu'))
        if amp_snapshot_path is not None:
            amp.load_state_dict(torch.load(amp_snapshot_path, map_location='cpu'))

    if resume:
        assert initial_encoder is None
        assert initial_model_prefix is None
        assert initial_model_index is None
        assert qv1_source_snapshot_path.exists()
        assert qv1_source_snapshot_path.is_file()
        assert qv2_source_snapshot_path.exists()
        assert qv2_source_snapshot_path.is_file()
        assert qv1_target_snapshot_path.exists()
        assert qv1_target_snapshot_path.is_file()
        assert qv2_target_snapshot_path.exists()
        assert qv2_target_snapshot_path.is_file()
        assert qv1_optimizer_snapshot_path.exists()
        assert qv1_optimizer_snapshot_path.is_file()
        assert qv2_optimizer_snapshot_path.exists()
        assert qv2_optimizer_snapshot_path.is_file()
        assert amp_snapshot_path.exists()
        assert amp_snapshot_path.is_file()
        qv1_source_state_dict = torch.load(qv1_source_snapshot_path, map_location='cpu')
        qv1_source_new_state_dict = {}
        for key, value in qv1_source_state_dict.items():
            new_key = re.sub('^module\\.', '', key)
            qv1_source_new_state_dict[new_key] = value
        qv1_source_model.load_state_dict(qv1_source_new_state_dict)
        qv1_source_model.cuda()
        qv2_source_state_dict = torch.load(qv2_source_snapshot_path, map_location='cpu')
        qv2_source_new_state_dict = {}
        for key, value in qv2_source_state_dict.items():
            new_key = re.sub('^module\\.', '', key)
            qv2_source_new_state_dict[new_key] = value
        qv2_source_model.load_state_dict(qv2_source_new_state_dict)
        qv2_source_model.cuda()
        qv1_target_state_dict = torch.load(qv1_target_snapshot_path, map_location='cpu')
        qv1_target_new_state_dict = {}
        for key, value in qv1_target_state_dict.items():
            new_key = re.sub('^module\\.', '', key)
            qv1_target_new_state_dict[new_key] = value
        qv1_target_model.load_state_dict(qv1_target_new_state_dict)
        qv1_target_model.cuda()
        qv2_target_state_dict = torch.load(qv2_target_snapshot_path, map_location='cpu')
        qv2_target_new_state_dict = {}
        for key, value in qv2_target_state_dict.items():
            new_key = re.sub('^module\\.', '', key)
            qv2_target_new_state_dict[new_key] = value
        qv2_target_model.load_state_dict(qv2_target_new_state_dict)
        qv2_target_model.cuda()
        qv1_optimizer.load_state_dict(torch.load(qv1_optimizer_snapshot_path, map_location='cpu'))
        qv2_optimizer.load_state_dict(torch.load(qv2_optimizer_snapshot_path, map_location='cpu'))
        amp.load_state_dict(torch.load(amp_snapshot_path, map_location='cpu'))

    if config['is_multiprocess']:
        init_process_group(backend='nccl')
        qv1_source_model = DistributedDataParallel(qv1_source_model)
        qv1_source_model = convert_syncbn_model(qv1_source_model)
        qv2_source_model = DistributedDataParallel(qv2_source_model)
        qv2_source_model = convert_syncbn_model(qv2_source_model)

    config['qv1_source_model'] = qv1_source_model
    config['qv2_source_model'] = qv2_source_model
    config['qv1_target_model'] = qv1_target_model
    config['qv2_target_model'] = qv2_target_model
    config['qv1_optimizer'] = qv1_optimizer
    config['qv2_optimizer'] = qv2_optimizer

    def snapshot_writer(
            qv1_source_model: QVModel, qv2_source_model: QVModel,
            qv1_target_model: QVModel, qv2_target_model: QVModel,
            qv1_optimizer: Optimizer, qv2_optimizer: Optimizer,
            num_samples: Optional[int]=None) -> None:
        snapshots_path.mkdir(parents=False, exist_ok=True)

        infix = '' if num_samples is None else f'.{num_samples}'

        torch.save(
            qv1_source_model.state_dict(),
            snapshots_path / f'qv1-source{infix}.pth')
        torch.save(
            qv2_source_model.state_dict(),
            snapshots_path / f'qv2-source{infix}.pth')
        torch.save(
            qv1_target_model.state_dict(),
            snapshots_path / f'qv1-target{infix}.pth')
        torch.save(
            qv2_target_model.state_dict(),
            snapshots_path / f'qv2-target{infix}.pth')
        torch.save(
            qv1_optimizer.state_dict(),
            snapshots_path / f'qv1-optimizer{infix}.pth')
        torch.save(
            qv2_optimizer.state_dict(),
            snapshots_path / f'qv2-optimizer{infix}.pth')
        if config['optimizer'] != 'mtadam':
            torch.save(
                amp.state_dict(), snapshots_path / f'amp{infix}.pth')

        q_model = QModel(qv1_model=qv1_target_model, qv2_model=qv2_target_model)
        q_model_state_dict = q_model.state_dict()
        compiled_q_model = {
            'model': {
                'module': 'kanachan.training.iql.q_model',
                'class': 'QModel',
                'kwargs': {
                    'dimension': config['dimension'],
                    'num_heads': config['num_heads'],
                    'num_layers': config['num_layers'],
                    'dim_feedforward': config['dim_feedforward'],
                    'dim_final_feedforward': config['dim_final_feedforward'],
                    'activation_function': config['activation_function'],
                    'dropout': 0.0,
                    'checkpointing': False
                },
                'state_dict': q_model_state_dict
            }
        }
        torch.save(compiled_q_model, snapshots_path / f'q-model{infix}.kanachan')

    with SummaryWriter(log_dir=config['tensorboard_path']) as writer:
        _training(**config, writer=writer, snapshot_writer=snapshot_writer)


if __name__ == '__main__':
    _main()
    sys.exit(0)
