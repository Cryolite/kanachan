#!/usr/bin/env python3

import re
import datetime
import math
from pathlib import Path
import os
import logging
import sys
from typing import Optional, Tuple, Callable
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from torch import backends
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import (Optimizer, SGD, Adam, RAdam)
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.distributed import (init_process_group, barrier, all_reduce)
from torch.utils.tensorboard.writer import SummaryWriter
from apex.optimizers import (FusedAdam, FusedSGD, FusedLAMB)
import kanachan.training.iql.config # pylint: disable=unused-import
from kanachan.training.constants import MAX_NUM_ACTION_CANDIDATES
from kanachan.training.common import Dataset
from kanachan.training.bert.encoder import Encoder
from kanachan.training.iql.value_model import (ValueDecoder, ValueModel)
from kanachan.training.iql.q_model import (QDecoder, QModel)
from kanachan.training.iql.qq_model import QQModel
from kanachan.training.iql.iterator_adaptor import IteratorAdaptor
from kanachan.model_loader import dump_model, dump_object


SnapshotWriter = Callable[
    [nn.Module, nn.Module, nn.Module, QModel, QModel, Optimizer, Optimizer, Optimizer, Optional[int]],
    None
]


def _training(
        *, is_multiprocess: bool, world_size: Optional[int], rank: Optional[int],
        is_main_process: bool, training_data: Path, num_workers: int, device: torch.device,
        dtype: torch.dtype, amp_dtype: torch.dtype, value_model: nn.Module,
        q1_source_model: nn.Module, q2_source_model: nn.Module, q1_target_model: nn.Module,
        q2_target_model: nn.Module, reward_plugin: Path, discount_factor: float, expectile: float,
        batch_size: int, gradient_accumulation_steps: int, v_max_gradient_norm: float,
        q_max_gradient_norm: float, value_optimizer: Optimizer, q1_optimizer: Optimizer,
        q2_optimizer: Optimizer, v_lr_scheduler, q1_lr_scheduler, q2_lr_scheduler,
        target_update_interval: int, target_update_rate: float, snapshot_interval: int,
        num_samples: int, summary_writer: SummaryWriter, snapshot_writer: SnapshotWriter) -> None:
    start_time = datetime.datetime.now()

    # Load the reward plugin.
    with open(reward_plugin, encoding='UTF-8') as file_pointer:
        exec(file_pointer.read(), globals()) # pylint: disable=exec-used

    # Prepare the training data loader. Note that this data loader must iterate
    # the training data set only once.
    def iterator_adaptor(path: Path) -> IteratorAdaptor:
        return IteratorAdaptor(path, get_reward) # type: ignore pylint: disable=undefined-variable
    dataset = Dataset(training_data, iterator_adaptor)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=(num_workers >= 1),
        drop_last=is_multiprocess)

    last_snapshot = None
    if snapshot_interval > 0:
        last_snapshot = num_samples

    num_consumed_samples = 0
    batch_count = 0

    grad_scaler = None
    if device != 'cpu':
        grad_scaler = GradScaler()

    batch_count = 0
    for annotation in data_loader:
        if num_consumed_samples < num_samples:
            num_consumed_samples += batch_size
            continue

        if is_multiprocess:
            assert world_size is not None
            assert rank is not None
            assert batch_size % world_size == 0
            barrier()
            first = (batch_size // world_size) * rank
            last = (batch_size // world_size) * (rank + 1)
            annotation = tuple(x[first:last] for x in annotation)

        if device != 'cpu':
            annotation = tuple(x.cuda() for x in annotation)

        local_batch_size = annotation[0].size(0)
        world_batch_size = batch_size

        # Get the Q target value to compute the loss of the V model.
        with torch.no_grad():
            def _compute_q_target(q_target_model: nn.Module) -> Tuple[torch.Tensor, float]:
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device != 'cpu' and dtype != amp_dtype)):
                    q: torch.Tensor = q_target_model(*(annotation[:4])) # pylint: disable=cell-var-from-loop
                assert q.dim() == 2
                assert q.size(0) == local_batch_size # pylint: disable=cell-var-from-loop
                assert q.size(1) == MAX_NUM_ACTION_CANDIDATES
                q = q[torch.arange(local_batch_size), annotation[4]] # pylint: disable=cell-var-from-loop
                assert q.dim() == 1
                assert q.size(0) == local_batch_size # pylint: disable=cell-var-from-loop

                q_batch_mean = q.detach().clone().mean()
                if is_multiprocess:
                    assert world_size is not None
                    all_reduce(q_batch_mean)
                    q_batch_mean /= world_size

                return q, q_batch_mean.item()

            q1, q1_mean = _compute_q_target(q1_target_model)
            q2, q2_mean = _compute_q_target(q2_target_model)

            q = torch.minimum(q1, q2)
            q = q.detach()
            assert(q.dim() == 1)
            assert(q.size(0) == local_batch_size)

        # Backprop for the V model.
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device != 'cpu' and dtype != amp_dtype)):
            value: torch.Tensor = value_model(*(annotation[:4]))
        assert(value.dim() == 1)
        assert(value.size(0) == local_batch_size)

        value_mean = value.detach().clone().mean()
        if is_multiprocess:
            assert world_size is not None
            all_reduce(value_mean)
            value_mean /= world_size

        value_loss = q - value
        value_loss = torch.where(
            value_loss < 0.0, (1.0 - expectile) * (value_loss ** 2.0),
            expectile * (value_loss ** 2.0))
        value_loss = torch.mean(value_loss)
        if math.isnan(value_loss.item()):
            raise RuntimeError('Value loss becomes NaN.')

        _value_loss = value_loss.detach().clone()
        if is_multiprocess:
            assert world_size is not None
            all_reduce(_value_loss)
            _value_loss /= world_size
        value_loss_to_display = _value_loss.item()

        value_loss /= gradient_accumulation_steps
        if grad_scaler is None:
            value_loss.backward()
        else:
            grad_scaler.scale(value_loss).backward()

        # Get the reward to compute the loss of the Q source models.
        reward = annotation[9]

        # Get V to compute the loss of the Q source models.
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device != 'cpu' and dtype != amp_dtype)):
                value = value_model(*(annotation[5:9]))
            assert(value.dim() == 1)
            assert(value.size(0) == local_batch_size)
            value *= discount_factor
            value = value.detach()

        def _backprop(q_source_model: nn.Module) -> float:
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device != 'cpu' and dtype != amp_dtype)):
                q: torch.Tensor = q_source_model(*(annotation[:4])) # pylint: disable=cell-var-from-loop
            assert q.dim() == 2
            assert q.size(0) == local_batch_size # pylint: disable=cell-var-from-loop
            assert q.size(1) == MAX_NUM_ACTION_CANDIDATES
            q = q[torch.arange(local_batch_size), annotation[4]] # pylint: disable=cell-var-from-loop

            q_loss = reward + value - q # pylint: disable=cell-var-from-loop
            q_loss = q_loss ** 2.0
            q_loss = torch.mean(q_loss)
            if math.isnan(q_loss.item()):
                raise RuntimeError('Q loss becomes NaN.')

            _q_loss = q_loss.detach().clone()
            if is_multiprocess:
                all_reduce(_q_loss)
                _q_loss /= world_size
            q_loss_to_display = _q_loss.item()

            q_loss /= gradient_accumulation_steps
            if grad_scaler is None:
                q_loss.backward()
            else:
                grad_scaler.scale(q_loss).backward()

            return q_loss_to_display

        # Backprop for the Q1 source model.
        q1_loss_to_display = _backprop(q1_source_model)

        # Backprop for the Q2 source model.
        q2_loss_to_display = _backprop(q2_source_model)

        num_samples += world_batch_size
        num_consumed_samples += world_batch_size
        batch_count += 1

        if batch_count % gradient_accumulation_steps == 0:
            def _step(
                    model: nn.Module, max_gradient_norm: float, optimizer: Optimizer,
                    scheduler) -> float:
                if grad_scaler is not None:
                    grad_scaler.unscale_(optimizer)
                gradient = nn.utils.parameters_to_vector(model.parameters())
                _gradient_norm: torch.Tensor = torch.linalg.vector_norm(gradient)
                gradient_norm: float = _gradient_norm.item()
                nn.utils.clip_grad_norm_(
                    model.parameters(), max_gradient_norm, error_if_nonfinite=False)
                if grad_scaler is None:
                    optimizer.step()
                else:
                    grad_scaler.step(optimizer)
                optimizer.zero_grad()
                scheduler.step()
                return gradient_norm

            v_gradient_norm = _step(
                value_model, v_max_gradient_norm, value_optimizer, v_lr_scheduler)
            q1_gradient_norm = _step(
                q1_source_model, q_max_gradient_norm, q1_optimizer, q1_lr_scheduler)
            q2_gradient_norm = _step(
                q2_source_model, q_max_gradient_norm, q2_optimizer, q2_lr_scheduler)
            if grad_scaler is not None:
                grad_scaler.update()

            if batch_count % (gradient_accumulation_steps * target_update_interval) == 0:
                with torch.no_grad():
                    q1_source_params = nn.utils.parameters_to_vector(q1_source_model.parameters())
                    q1_target_params = nn.utils.parameters_to_vector(q1_target_model.parameters())
                    q1_target_params *= (1.0 - target_update_rate)
                    q1_target_params += target_update_rate * q1_source_params
                    nn.utils.vector_to_parameters(q1_target_params, q1_target_model.parameters())

                    q2_source_params = nn.utils.parameters_to_vector(q2_source_model.parameters())
                    q2_target_params = nn.utils.parameters_to_vector(q2_target_model.parameters())
                    q2_target_params *= (1.0 - target_update_rate)
                    q2_target_params += target_update_rate * q2_source_params
                    nn.utils.vector_to_parameters(q2_target_params, q2_target_model.parameters())

            if is_main_process:
                logging.info(
                    'sample = %d, value loss = %E, Q1 loss = %E, Q2 loss = %E,'
                    ' value gradient norm = %E, Q1 gradient norm = %E, Q2 gradient norm = %E',
                    num_samples, value_loss_to_display, q1_loss_to_display, q2_loss_to_display,
                    v_gradient_norm, q1_gradient_norm, q2_gradient_norm)
                summary_writer.add_scalars('Q', { 'Q1': q1_mean, 'Q2': q2_mean }, num_samples)
                summary_writer.add_scalars(
                    'Q Gradient Norm', { 'Q1': q1_gradient_norm, 'Q2': q2_gradient_norm },
                    num_samples)
                summary_writer.add_scalars(
                    'Q Loss', { 'Q1': q1_loss_to_display, 'Q2': q2_loss_to_display }, num_samples)
                summary_writer.add_scalar('Value', value_mean.item(), num_samples)
                summary_writer.add_scalar('Value Gradient Norm', v_gradient_norm, num_samples)
                summary_writer.add_scalar('Value Loss', value_loss_to_display, num_samples)
        else:
            if is_main_process:
                logging.info(
                    'sample = %d, value loss = %E, Q1 loss = %E, Q2 loss = %E',
                    num_samples, value_loss_to_display, q1_loss_to_display, q2_loss_to_display)
                summary_writer.add_scalars('Q', { 'Q1': q1_mean, 'Q2': q2_mean }, num_samples)
                summary_writer.add_scalars(
                    'Q Loss', { 'Q1': q1_loss_to_display, 'Q2': q2_loss_to_display }, num_samples)
                summary_writer.add_scalar('Value', value_mean.item(), num_samples)
                summary_writer.add_scalar('Value Loss', value_loss_to_display, num_samples)

        if is_main_process and last_snapshot is not None and num_samples - last_snapshot >= snapshot_interval:
            snapshot_writer(
                value_model, q1_source_model, q2_source_model, q1_target_model, q2_target_model,
                value_optimizer, q1_optimizer, q2_optimizer, num_samples)
            last_snapshot = num_samples

    if is_multiprocess:
        barrier()

    elapsed_time = datetime.datetime.now() - start_time

    if is_main_process:
        logging.info('A training has finished (elapsed time = %s).', elapsed_time)
        snapshot_writer(
            value_model, q1_source_model, q2_source_model, q1_target_model, q2_target_model,
            value_optimizer, q1_optimizer, q2_optimizer)


@hydra.main(version_base=None, config_name='config')
def _main(config: DictConfig) -> None:
    if 'LOCAL_RANK' in os.environ:
        if os.environ['WORLD_SIZE'] != os.environ['LOCAL_WORLD_SIZE']:
            raise RuntimeError('Multi-node not supported.')
        world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        rank = int(os.environ['LOCAL_RANK'])
        is_multiprocess = True
        is_main_process = (rank == 0)
    else:
        world_size = None
        rank = None
        is_multiprocess = False
        is_main_process = True

    if not config.training_data.exists():
        raise RuntimeError(f'{config.training_data}: Does not exist.')
    if not config.training_data.is_file():
        raise RuntimeError(f'{config.training_data}: Not a file.')

    if config.device.type == 'cpu':
        if config.num_workers is None:
            config.num_workers = 0
        if config.num_workers < 0:
            raise RuntimeError(f'{config.num_workers}: An invalid number of workers.')
        if config.num_workers > 0:
            raise RuntimeError(f'{config.num_workers}: An invalid number of workers for CPU.')
    else:
        if config.num_workers is None:
            config.num_workers = 2
        if config.num_workers < 0:
            raise RuntimeError(f'{config.num_workers}: An invalid number of workers.')
        if config.num_workers == 0:
            raise RuntimeError(f'{config.num_workers}: An invalid number of workers for GPU.')

    if config.device.type is not None:
        match = re.search('^(?:cpu|cuda(?::\\d+)?)$', config.device.type)
        if match is None:
            raise RuntimeError(f'{config.device.type}: An invalid device.')
    elif backends.cuda.is_built():
        config.device.type = 'cuda'
    else:
        config.device.type = 'cpu'
    if config.device.type == 'cuda':
        torch.cuda.set_device(rank)

    if config.device.dtype not in ('float64', 'double', 'float32', 'float', 'float16', 'half', 'bfloat16'):
        raise RuntimeError(f'{config.device.dtype}: An invalid dtype.')
    dtype = {
        'float64': torch.float64, 'double': torch.float64,
        'float32': torch.float32, 'float': torch.float32,
        'float16': torch.float16, 'half': torch.float16,
        'bfloat16': torch.bfloat16
    }[config.device.dtype]

    if config.device.type == 'cpu':
        if config.device.amp_dtype is not None:
            raise RuntimeError('AMP is not supported on CPU.')
        config.device.amp_dtype = 'bfloat16'
    if config.device.amp_dtype not in ('float64', 'double', 'float32', 'float', 'float16', 'half', 'bfloat16'):
        raise RuntimeError(f'{config.device.amp_dtype}: An invalid AMP dtype.')
    amp_dtype = {
        'float64': torch.float64, 'double': torch.float64,
        'float32': torch.float32, 'float': torch.float32,
        'float16': torch.float16, 'half': torch.float16,
        'bfloat16': torch.bfloat16
    }[config.device.amp_dtype]

    if backends.cudnn.is_available():
        backends.cudnn.benchmark = True

    if config.encoder.position_encoder not in ('positional_encoding', 'position_embedding'):
        raise RuntimeError(f'{config.encoder.position_encoder}: An invalid position encoder.')

    if config.encoder.dimension < 1:
        raise RuntimeError(f'{config.encoder.dimension}: An invalid dimension for the encoder.')

    if config.encoder.num_heads < 1:
        raise RuntimeError(
            f'{config.encoder.num_heads}: An invalid number of heads for the encoder.')

    if config.encoder.dim_feedforward is None:
        config.encoder.dim_feedforward = 4 * config.encoder.dimension
    if config.encoder.dim_feedforward < 1:
        raise RuntimeError(
            f'{config.encoder.dim_feedforward}:'
            ' An invalid dimension of the feedfoward networks for the encoder.')

    if config.encoder.activation_function not in ('relu', 'gelu'):
        raise RuntimeError(
            f'{config.encoder.activation_function}: '
            'An invalid activation function for the encoder.')

    if config.encoder.dropout < 0.0 or 1.0 <= config.encoder.dropout:
        raise RuntimeError(f'{config.encoder.dropout}: An invalid dropout value for the encoder.')

    if config.encoder.num_layers < 1:
        raise RuntimeError(f'{config.encoder.num_layers}: An invalid number of encoder layers.')

    if config.encoder.load_from is not None:
        if not config.encoder.load_from.exists():
            raise RuntimeError(f'{config.encoder.load_from}: Does not exist.')
        if not config.encoder.load_from.is_file():
            raise RuntimeError(f'{config.encoder.load_from}: Not a file.')

    if config.decoder.dim_feedforward is None:
        config.decoder.dim_feedforward = config.encoder.dim_feedforward
    if config.decoder.dim_feedforward < 1:
        raise RuntimeError(
            f'{config.decoder.dim_feedforward}: '
            'An invalid dimension of the feedforward networks for the decoder.')

    if config.decoder.activation_function not in ('relu', 'gelu'):
        raise RuntimeError(
            f'{config.decoder.activation_function}: '
            'An invalid activation function for the decoder.')

    if config.decoder.dropout < 0.0 or 1.0 <= config.decoder.dropout:
        raise RuntimeError(f'{config.decoder.dropout}: An invalid dropout value for the decoder.')

    if config.decoder.num_layers < 1:
        raise RuntimeError(f'{config.decoder.num_layers}: An invalid number of decoder layers.')

    if config.decoder.load_from is not None:
        if not config.decoder.load_from.exists():
            raise RuntimeError(f'{config.decoder.load_from}: Does not exist.')
        if not config.decoder.load_from.is_file():
            raise RuntimeError(f'{config.decoder.load_from}: Not a file.')

    if config.initial_q_encoder is not None:
        if config.encoder.load_from is not None:
            raise RuntimeError('`initial_q_encoder` conflicts with `encoder.load_from`.')
        if config.decoder.load_from is not None:
            raise RuntimeError('`initial_q_encoder` conflicts with `decoder.load_from`.')
        if not config.initial_q_encoder.exists():
            raise RuntimeError(f'{config.initial_q_encoder}: Does not exist.')
        if not config.initial_q_encoder.is_file():
            raise RuntimeError(f'{config.initial_q_encoder}: Not a file.')

    if config.initial_q_decoder is not None:
        if config.encoder.load_from is not None:
            raise RuntimeError('`initial_q_decoder` conflicts with `encoder.load_from`.')
        if config.decoder.load_from is not None:
            raise RuntimeError('`initial_q_decoder` conflicts with `decoder.load_from`.')
        if not config.initial_q_decoder.exists():
            raise RuntimeError(f'{config.initial_q_decoder}: Does not exist.')
        if not config.initial_q_decoder.is_file():
            raise RuntimeError(f'{config.initial_q_decoder}: Not a file.')

    if config.initial_model_prefix is not None:
        if config.encoder.load_from is not None:
            raise RuntimeError('`initial_model_prefix` conflicts with `encoder.load_from`.')
        if config.decoder.load_from is not None:
            raise RuntimeError('`initial_model_prefix` conflicts with `decoder.load_from`.')
        if config.initial_q_encoder is not None:
            raise RuntimeError('`initial_model_prefix` conflicts with `initial_q_encoder`.')
        if config.initial_q_decoder is not None:
            raise RuntimeError('`initial_model_prefix` conflicts with `initial_q_decoder`.')
        if not config.initial_model_prefix.exists():
            raise RuntimeError(f'{config.initial_model_prefix}: Does not exist.')
        if not config.initial_model_prefix.is_dir():
            raise RuntimeError(f'{config.initial_model_prefix}: Not a directory.')

    if config.initial_model_index is not None:
        if config.initial_model_prefix is None:
            raise RuntimeError('`initial_model_index` must be combined with `initial_model_prefix`.')
        if config.initial_model_index < 0:
            raise RuntimeError(f'{config.initial_model_index}: An invalid initial model index.')

    num_samples = 0

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None
        assert config.decoder.load_from is None
        assert config.initial_q_encoder is None
        assert config.initial_q_decoder is None

        if config.initial_model_index is None:
            for child in os.listdir(config.initial_model_prefix):
                match = re.search(
                    '^(?:value|q[12]-source|q[12]-target|value-optimizer|q[12]-optimizer)(?:\\.(\\d+))?\\.pth$',
                    child)
                if match is None:
                    continue
                if match[1] is None:
                    config.initial_model_index = sys.maxsize
                    continue
                if config.initial_model_index is None or int(match[1]) > config.initial_model_index:
                    config.initial_model_index = int(match[1])
                    continue
        if config.initial_model_index is None:
            raise RuntimeError(f'{config.initial_model_prefix}: No model snapshot found.')

        if config.initial_model_index == sys.maxsize:
            config.initial_model_index = 0
            infix = ''
        else:
            num_samples = config.initial_model_index
            infix = f'.{num_samples}'

        value_snapshot_path: Path = config.initial_model_prefix / f'value{infix}.pth'
        if not value_snapshot_path.exists():
            raise RuntimeError(f'{value_snapshot_path}: Does not exist.')
        if not value_snapshot_path.is_file():
            raise RuntimeError(f'{value_snapshot_path}: Not a file.')

        q1_source_snapshot_path: Path = config.initial_model_prefix / f'q1-source{infix}.pth'
        if not q1_source_snapshot_path.exists():
            raise RuntimeError(f'{q1_source_snapshot_path}: Does not exist.')
        if not q1_source_snapshot_path.is_file():
            raise RuntimeError(f'{q1_source_snapshot_path}: Not a file.')

        q2_source_snapshot_path: Path = config.initial_model_prefix / f'q2-source{infix}.pth'
        if not q2_source_snapshot_path.exists():
            raise RuntimeError(f'{q2_source_snapshot_path}: Does not exist.')
        if not q2_source_snapshot_path.is_file():
            raise RuntimeError(f'{q2_source_snapshot_path}: Not a file.')

        q1_target_snapshot_path: Path = config.initial_model_prefix / f'q1-target{infix}.pth'
        if not q1_target_snapshot_path.exists():
            raise RuntimeError(f'{q1_target_snapshot_path}: Does not exist.')
        if not q1_target_snapshot_path.is_file():
            raise RuntimeError(f'{q1_target_snapshot_path}: Not a file.')

        q2_target_snapshot_path: Path = config.initial_model_prefix / f'q2-target{infix}.pth'
        if not q2_target_snapshot_path.exists():
            raise RuntimeError(f'{q2_target_snapshot_path}: Does not exist.')
        if not q2_target_snapshot_path.is_file():
            raise RuntimeError(f'{q2_target_snapshot_path}: Not a file.')

        value_optimizer_snapshot_path: Optional[Path] = config.initial_model_prefix / f'value-optimizer{infix}.pth'
        if not value_optimizer_snapshot_path.is_file() or config.optimizer.initialize:
            value_optimizer_snapshot_path = None

        q1_optimizer_snapshot_path: Optional[Path] = config.initial_model_prefix / f'q1-optimizer{infix}.pth'
        if not q1_optimizer_snapshot_path.is_file() or config.optimizer.initialize:
            q1_optimizer_snapshot_path = None

        q2_optimizer_snapshot_path: Optional[Path] = config.initial_model_prefix / f'q2-optimizer{infix}.pth'
        if not q2_optimizer_snapshot_path.is_file() or config.optimizer.initialize:
            q2_optimizer_snapshot_path = None

    if not config.reward_plugin.exists():
        raise RuntimeError(f'{config.reward_plugin}: Does not exist.')
    if not config.reward_plugin.is_file():
        raise RuntimeError(f'{config.reward_plugin}: Not a file.')

    if config.discount_factor <= 0.0 or 1.0 < config.discount_factor:
        raise RuntimeError(f'{config.discount_factor}: An invalid value for `discount_factor`.')

    if config.expectile <= 0.0 or 1.0 <= config.expectile:
        raise RuntimeError(f'{config.expectile}: An invalid value for `expectile`.')

    if config.batch_size < 1:
        raise RuntimeError(f'{config.batch_size}: An invalid value for `batch_size`.')
    if config.batch_size % world_size != 0:
        raise RuntimeError(f'`batch_size` must be divisible by the world size ({world_size}).')

    if config.gradient_accumulation_steps < 1:
        raise RuntimeError(
            f'{config.gradient_accumulation_steps}: '
            'An invalid value for `gradient_accumulation_steps`.')
    if config.gradient_accumulation_steps >= 2 and config.optimizer.type == 'mtadam':
        raise RuntimeError('`mtadam` does not support for gradient accumulation.')

    if config.q_max_gradient_norm <= 0.0:
        raise RuntimeError(
            f'{config.q_max_gradient_norm}: An invalid value for `q_max_gradient_norm`.')

    if config.v_max_gradient_norm <= 0.0:
        raise RuntimeError(
            f'{config.v_max_gradient_norm}: An invalid value for `v_max_gradient_norm`.')

    if config.optimizer.type in ('sgd',):
        if config.optimizer.momentum is None:
            raise RuntimeError('`optimizer.momentum` must be specified for `sgd`.')
        if config.optimizer.momentum < 0.0 or 1.0 <= config.optimizer.momentum:
            raise RuntimeError(
                f'{config.optimizer.momentum}: An invalid value for `optimizer.momentum`.')
    else:
        if config.optimizer.momentum is not None:
            raise RuntimeError(f'`optimizer.momentum` is useless for `{config.optimizer.type}`.')

    if config.optimizer.type in ('sgd',):
        if config.optimizer.epsilon is not None:
            raise RuntimeError(f'`optimizer.epsilon` is useless for `{config.optimizer.type}`.')
    else:
        if config.optimizer.epsilon is None:
            if config.optimizer.type in ('adam', 'radam', 'mtadam'):
                config.optimizer.epsilon = 1.0e-8
            elif config.optimizer in ('lamb',):
                config.optimizer.epsilon = 1.0e-6
            else:
                raise NotImplementedError(config.optimizer.type)
    if config.optimizer.epsilon is not None and config.optimizer.epsilon <= 0.0:
        raise RuntimeError(f'{config.optimizer.epsilon}: An invalid value for `optimizer.epsilon`.')

    if config.optimizer.learning_rate <= 0.0:
        raise RuntimeError(
            f'{config.optimizer.learning_rate}: An invalid value for `optimizer.learning_rate`.')

    if config.optimizer.warmup_start_factor <= 0.0 or 1.0 <= config.optimizer.warmup_start_factor:
        raise RuntimeError(
            f'{config.optimizer.warmup_start_factor}: '
            '`optimizer.warmup_start_factor` must be a real number with the range (0.0, 1.0).')

    if config.optimizer.warmup_steps < 0:
        raise RuntimeError(
            f'{config.optimizer.warmup_steps}: '
            '`optimizer.warmup_steps` must be a non-negative integer.')

    if config.optimizer.annealing_steps <= 0:
        raise RuntimeError(
            f'{config.optimizer.annealing_steps}: '
            '`optimizer.annealing_steps` must be a positive integer.')

    if config.optimizer.annealing_steps_factor <= 0:
        raise RuntimeError(
            f'{config.optimizer.annealing_steps_factor}: '
            '`optimizer.annealing_steps_factor` must be a positive integer.')

    if config.target_update_interval <= 0:
        raise RuntimeError(
            f'{config.target_update_interval}: An invalid value for `target_update_interval`.')

    if config.target_update_rate <= 0.0 or 1.0 < config.target_update_rate:
        raise RuntimeError(
            f'{config.target_update_rate}: An invalid value for `target_update_rate`.')

    if config.snapshot_interval < 0:
        raise RuntimeError(f'{config.snapshot_interval}: An invalid value for `snapshot_interval`.')

    experiment_path = Path(HydraConfig.get().runtime.output_dir)

    tensorboard_path = experiment_path / 'tensorboard'
    tensorboard_path.mkdir(parents=True, exist_ok=True)

    snapshots_path = experiment_path / 'snapshots'
    snapshots_path.mkdir(parents=True, exist_ok=True)

    if is_main_process:
        if world_size is None:
            assert rank is None
            logging.info('World size: N/A (single process)')
            logging.info('Process rank: N/A (single process)')
        else:
            assert rank is not None
            logging.info('World size: %d', world_size)
            logging.info('Process rank: %d', rank)
        logging.info('Training data: %s', config.training_data)
        if num_samples > 0:
            logging.info('# of training samples consumed so far: %d', num_samples)
        logging.info('# of workers: %d', config.num_workers)
        logging.info('Device: %s', config.device.type)
        if backends.cudnn.is_available():
            logging.info('cuDNN: available')
        else:
            logging.info('cuDNN: N/A')
        logging.info('dtype: %s', dtype)
        logging.info('AMP dtype: %s', amp_dtype)
        logging.info('Position encoder: %s', config.encoder.position_encoder)
        logging.info('Encoder dimension: %d', config.encoder.dimension)
        logging.info('# of heads for encoder: %d', config.encoder.num_heads)
        logging.info(
            'Dimension of feedforward networks for encoder: %d', config.encoder.dim_feedforward)
        logging.info('Activation function for encoder: %s', config.encoder.activation_function)
        logging.info('Dropout for encoder: %f', config.encoder.dropout)
        logging.info('# of encoder layers: %d', config.encoder.num_layers)
        if config.encoder.load_from is not None:
            logging.info('Load encoder from: %s', config.encoder.load_from)
        if config.decoder.num_layers >= 2:
            logging.info(
                'Dimension of feedforward networks for decoder: %d', config.decoder.dim_feedforward)
        logging.info('Activation function for decoder: %s', config.decoder.activation_function)
        logging.info('Dropout for decoder: %f', config.decoder.dropout)
        logging.info('# of decoder layers: %d', config.decoder.num_layers)
        if config.decoder.load_from is not None:
            logging.info('Load decoder from: %s', config.decoder.load_from)
        if config.initial_q_encoder is not None:
            logging.info('Load Q encoder from: %s', config.initial_q_encoder)
        if config.initial_q_decoder is not None:
            logging.info('Load Q decoder from: %s', config.initial_q_decoder)
        if config.initial_model_prefix is not None:
            logging.info('Initial model prefix: %s', config.initial_model_prefix)
            logging.info('Initlal model index: %d', config.initial_model_index)
            if config.optimizer.initialize:
                logging.info('(Will not load optimizer)')
        logging.info('Reward plugin: %s', config.reward_plugin)
        logging.info('Discount factor: %f', config.discount_factor)
        logging.info('Expectile: %f', config.expectile)
        logging.info('Checkpointing: %s', config.checkpointing)
        if world_size is None:
            logging.info('Batch size: %d', config.batch_size)
        else:
            logging.info('Local batch size: %d', config.batch_size // world_size)
            logging.info('World batch size: %d', config.batch_size)
        logging.info('# of steps for gradient accumulation: %d', config.gradient_accumulation_steps)
        logging.info(
            'Virtual batch size: %d', config.batch_size * config.gradient_accumulation_steps)
        logging.info('Norm threshold for gradient clipping on Q: %E', config.q_max_gradient_norm)
        logging.info('Norm threshold for gradient clipping on V: %E', config.v_max_gradient_norm)
        logging.info('Optimizer: %s', config.optimizer.type)
        if config.optimizer in ('sgd',):
            logging.info('Momentum factor: %f', config.optimizer.momentum)
        if config.optimizer in ('adam', 'radam', 'mtadam', 'lamb'):
            logging.info('Epsilon parameter: %E', config.optimizer.epsilon)
        logging.info('Learning rate: %E', config.optimizer.learning_rate)
        if config.optimizer.warmup_steps == 0:
            logging.info('LR warm-up: (disabled)')
        else:
            logging.info('LR warm-up start factor: %E', config.optimizer.warmup_start_factor)
            logging.info('# of steps for LR warm-up: %d', config.optimizer.warmup_steps)
        if config.optimizer.annealing_steps is None:
            logging.info('LR annealing: (disabled)')
        else:
            logging.info('# of steps for LR annealing: %d', config.optimizer.annealing_steps)
            logging.info(
                'Step factor for LR annealing: %d', config.optimizer.annealing_steps_factor)
        logging.info('Target update interval: %d', config.target_update_interval)
        logging.info('Target update rate: %f', config.target_update_rate)
        if config.initial_model_prefix is not None:
            logging.info('Initial value network snapshot: %s', value_snapshot_path)
            logging.info('Initial Q1 source network snapshot: %s', q1_source_snapshot_path)
            logging.info('Initial Q2 source network snapshot: %s', q2_source_snapshot_path)
            logging.info('Initial Q1 target network snapshot: %s', q1_target_snapshot_path)
            logging.info('Initial Q2 target network snapshot: %s', q2_target_snapshot_path)
            if value_optimizer_snapshot_path is not None:
                logging.info('Initial value optimizer snapshot: %s', value_optimizer_snapshot_path)
            if q1_optimizer_snapshot_path is not None:
                logging.info('Initial Q1 optimizer snapshot: %s', q1_optimizer_snapshot_path)
            if q2_optimizer_snapshot_path is not None:
                logging.info('Initial Q2 optimizer snapshot: %s', q2_optimizer_snapshot_path)
        logging.info('Experiment output: %s', experiment_path)
        if config.snapshot_interval == 0:
            logging.info('Snapshot interval: N/A')
        else:
            logging.info('Snapshot interval: %d', config.snapshot_interval)

    value_encoder = Encoder(
        position_encoder=config.encoder.position_encoder, dimension=config.encoder.dimension,
        num_heads=config.encoder.num_heads, dim_feedforward=config.encoder.dim_feedforward,
        num_layers=config.encoder.num_layers,
        activation_function=config.encoder.activation_function, dropout=config.encoder.dropout,
        checkpointing=config.checkpointing, device=config.device.type, dtype=dtype)
    value_decoder = ValueDecoder(
        dimension=config.encoder.dimension, dim_feedforward=config.decoder.dim_feedforward,
        activation_function=config.decoder.activation_function, dropout=config.decoder.dropout,
        num_layers=config.decoder.num_layers, device=config.device.type, dtype=dtype)
    value_model = ValueModel(value_encoder, value_decoder)
    value_model.to(device=config.device.type, dtype=dtype)

    q1_source_encoder = Encoder(
        position_encoder=config.encoder.position_encoder, dimension=config.encoder.dimension,
        num_heads=config.encoder.num_heads, dim_feedforward=config.encoder.dim_feedforward,
        num_layers=config.encoder.num_layers,
        activation_function=config.encoder.activation_function, dropout=config.encoder.dropout,
        checkpointing=config.checkpointing, device=config.device.type, dtype=dtype)
    q1_source_decoder = QDecoder(
        dimension=config.encoder.dimension, dim_feedforward=config.decoder.dim_feedforward,
        activation_function=config.decoder.activation_function, dropout=config.decoder.dropout,
        num_layers=config.decoder.num_layers, device=config.device.type, dtype=dtype)
    q1_source_model = QModel(q1_source_encoder, q1_source_decoder)
    q1_source_model.to(device=config.device.type, dtype=dtype)

    q2_source_encoder = Encoder(
        position_encoder=config.encoder.position_encoder, dimension=config.encoder.dimension,
        num_heads=config.encoder.num_heads, dim_feedforward=config.encoder.dim_feedforward,
        num_layers=config.encoder.num_layers,
        activation_function=config.encoder.activation_function, dropout=config.encoder.dropout,
        checkpointing=config.checkpointing, device=config.device.type, dtype=dtype)
    q2_source_decoder = QDecoder(
        dimension=config.encoder.dimension, dim_feedforward=config.decoder.dim_feedforward,
        activation_function=config.decoder.activation_function, dropout=config.decoder.dropout,
        num_layers=config.decoder.num_layers, device=config.device.type, dtype=dtype)
    q2_source_model = QModel(q2_source_encoder, q2_source_decoder)
    q2_source_model.to(device=config.device.type, dtype=dtype)

    q1_target_encoder = Encoder(
        position_encoder=config.encoder.position_encoder, dimension=config.encoder.dimension,
        num_heads=config.encoder.num_heads, dim_feedforward=config.encoder.dim_feedforward,
        num_layers=config.encoder.num_layers,
        activation_function=config.encoder.activation_function, dropout=config.encoder.dropout,
        checkpointing=config.checkpointing, device=config.device.type, dtype=dtype)
    q1_target_decoder = QDecoder(
        dimension=config.encoder.dimension, dim_feedforward=config.decoder.dim_feedforward,
        activation_function=config.decoder.activation_function, dropout=config.decoder.dropout,
        num_layers=config.decoder.num_layers, device=config.device.type, dtype=dtype)
    q1_target_model = QModel(q1_target_encoder, q1_target_decoder)
    q1_target_model.requires_grad_(False)
    q1_target_model.to(device=config.device.type, dtype=dtype)

    q2_target_encoder = Encoder(
        position_encoder=config.encoder.position_encoder, dimension=config.encoder.dimension,
        num_heads=config.encoder.num_heads, dim_feedforward=config.encoder.dim_feedforward,
        num_layers=config.encoder.num_layers,
        activation_function=config.encoder.activation_function, dropout=config.encoder.dropout,
        checkpointing=config.checkpointing, device=config.device.type, dtype=dtype)
    q2_target_decoder = QDecoder(
        dimension=config.encoder.dimension, dim_feedforward=config.decoder.dim_feedforward,
        activation_function=config.decoder.activation_function, dropout=config.decoder.dropout,
        num_layers=config.decoder.num_layers, device=config.device.type, dtype=dtype)
    q2_target_model = QModel(q2_target_encoder, q2_target_decoder)
    q2_target_model.requires_grad_(False)
    q2_target_model.to(device=config.device.type, dtype=dtype)

    if config.optimizer.type == 'sgd':
        def construct_optimizer(model: nn.Module) -> Optimizer:
            if config.device.type == 'cpu':
                return SGD(
                    model.parameters(), lr=config.optimizer.learning_rate,
                    momentum=config.optimizer.momentum)
            return FusedSGD(
                model.parameters(), lr=config.optimizer.learning_rate,
                momentum=config.optimizer.momentum)
    elif config.optimizer.type == 'adam':
        def construct_optimizer(model: nn.Module) -> Optimizer:
            if config.device.type == 'cpu':
                return Adam(
                    model.parameters(), lr=config.optimizer.learning_rate,
                    eps=config.optimizer.epsilon)
            return FusedAdam(
                model.parameters(), lr=config.optimizer.learning_rate, eps=config.optimizer.epsilon)
    elif config.optimizer.type == 'radam':
        def construct_optimizer(model: nn.Module) -> Optimizer:
            return RAdam(
                model.parameters(), lr=config.optimizer.learning_rate, eps=config.optimizer.epsilon)
    elif config.optimizer.type == 'lamb':
        def construct_optimizer(model: nn.Module) -> Optimizer:
            return FusedLAMB(
                model.parameters(), lr=config.optimizer.learning_rate, eps=config.optimizer.epsilon)
    else:
        raise NotImplementedError(config.optimizer.type)
    value_optimizer = construct_optimizer(value_model)
    q1_optimizer = construct_optimizer(q1_source_model)
    q2_optimizer = construct_optimizer(q2_source_model)

    v_warmup_lr_scheduler = lr_scheduler.LinearLR(
        value_optimizer, start_factor=config.optimizer.warmup_start_factor,
        total_iters=config.optimizer.warmup_steps)
    v_annealing_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        value_optimizer, config.optimizer.annealing_steps, config.optimizer.annealing_steps_factor)
    v_lr_scheduler = lr_scheduler.SequentialLR(
        value_optimizer, [v_warmup_lr_scheduler, v_annealing_lr_scheduler],
        [v_warmup_lr_scheduler.total_iters])

    q1_warmup_lr_scheduler = lr_scheduler.LinearLR(
        q1_optimizer, start_factor=config.optimizer.warmup_start_factor,
        total_iters=config.optimizer.warmup_steps)
    q1_annealing_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        q1_optimizer, config.optimizer.annealing_steps, config.optimizer.annealing_steps_factor)
    q1_lr_scheduler = lr_scheduler.SequentialLR(
        q1_optimizer, [q1_warmup_lr_scheduler, q1_annealing_lr_scheduler],
        [q1_warmup_lr_scheduler.total_iters])

    q2_warmup_lr_scheduler = lr_scheduler.LinearLR(
        q2_optimizer, start_factor=config.optimizer.warmup_start_factor,
        total_iters=config.optimizer.warmup_steps)
    q2_annealing_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        q2_optimizer, config.optimizer.annealing_steps, config.optimizer.annealing_steps_factor)
    q2_lr_scheduler = lr_scheduler.SequentialLR(
        q2_optimizer, [q2_warmup_lr_scheduler, q2_annealing_lr_scheduler],
        [q2_warmup_lr_scheduler.total_iters])

    if config.encoder.load_from is not None:
        assert config.initial_q_encoder is None
        assert config.initial_q_decoder is None
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        encoder_state_dict = torch.load(config.encoder.load_from, map_location='cpu')
        value_encoder.load_state_dict(encoder_state_dict)
        q1_source_encoder.load_state_dict(encoder_state_dict)
        q2_source_encoder.load_state_dict(encoder_state_dict)
        q1_target_encoder.load_state_dict(encoder_state_dict)
        q2_target_encoder.load_state_dict(encoder_state_dict)
        if config.device.type != 'cpu':
            value_encoder.cuda()
            q1_source_encoder.cuda()
            q2_source_encoder.cuda()
            q1_target_encoder.cuda()
            q2_target_encoder.cuda()

    if config.decoder.load_from is not None:
        assert config.initial_q_encoder is None
        assert config.initial_q_decoder is None
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        decoder_state_dict = torch.load(config.decoder.load_from, map_location='cpu')
        value_decoder.load_state_dict(decoder_state_dict)
        q1_source_decoder.load_state_dict(decoder_state_dict)
        q2_source_decoder.load_state_dict(decoder_state_dict)
        q1_target_decoder.load_state_dict(decoder_state_dict)
        q2_target_decoder.load_state_dict(decoder_state_dict)
        if config.device.type != 'cpu':
            value_decoder.cuda()
            q1_source_decoder.cuda()
            q2_source_decoder.cuda()
            q1_target_decoder.cuda()
            q2_target_decoder.cuda()

    if config.initial_q_encoder is not None:
        assert config.initial_q_decoder is not None
        assert config.encoder.load_from is None
        assert config.decoder.load_from is None
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        q_encoder_state_dict = torch.load(config.initial_q_encoder, map_location='cpu')
        q_decoder_state_dict = torch.load(config.initial_q_decoder, map_location='cpu')
        q1_source_encoder.load_state_dict(q_encoder_state_dict)
        q1_source_decoder.load_state_dict(q_decoder_state_dict)
        q2_source_encoder.load_state_dict(q_encoder_state_dict)
        q2_source_decoder.load_state_dict(q_decoder_state_dict)
        q1_target_encoder.load_state_dict(q_encoder_state_dict)
        q1_target_decoder.load_state_dict(q_decoder_state_dict)
        q2_target_encoder.load_state_dict(q_encoder_state_dict)
        q2_target_decoder.load_state_dict(q_decoder_state_dict)
        if config.device.type != 'cpu':
            q1_source_model.cuda()
            q2_source_model.cuda()
            q1_target_model.cuda()
            q2_target_model.cuda()

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None
        assert config.decoder.load_from is None
        assert config.initial_q_model is None

        value_state_dict = torch.load(value_snapshot_path, map_location='cpu')
        value_model.load_state_dict(value_state_dict)
        if config.device.type != 'cpu':
            value_model.cuda()

        q1_source_state_dict = torch.load(q1_source_snapshot_path, map_location='cpu')
        q1_source_model.load_state_dict(q1_source_state_dict)
        if config.device.type != 'cpu':
            q1_source_model.cuda()

        q2_source_state_dict = torch.load(q2_source_snapshot_path, map_location='cpu')
        q2_source_model.load_state_dict(q2_source_state_dict)
        if config.device.type != 'cpu':
            q2_source_model.cuda()

        q1_target_state_dict = torch.load(q1_target_snapshot_path, map_location='cpu')
        q1_target_model.load_state_dict(q1_target_state_dict)
        if config.device.type != 'cpu':
            q1_target_model.cuda()

        q2_target_state_dict = torch.load(q2_target_snapshot_path, map_location='cpu')
        q2_target_model.load_state_dict(q2_target_state_dict)
        if config.device.type != 'cpu':
            q2_target_model.cuda()

        if value_optimizer_snapshot_path is not None:
            value_optimizer.load_state_dict(
                torch.load(value_optimizer_snapshot_path, map_location='cpu'))

        if q1_optimizer_snapshot_path is not None:
            q1_optimizer.load_state_dict(
                torch.load(q1_optimizer_snapshot_path, map_location='cpu'))

        if q2_optimizer_snapshot_path is not None:
            q2_optimizer.load_state_dict(
                torch.load(q2_optimizer_snapshot_path, map_location='cpu'))

    if is_multiprocess:
        init_process_group(backend='nccl')
        value_model = DistributedDataParallel(value_model)
        value_model = nn.SyncBatchNorm.convert_sync_batchnorm(value_model)
        q1_source_model = DistributedDataParallel(q1_source_model)
        q1_source_model = nn.SyncBatchNorm.convert_sync_batchnorm(q1_source_model)
        q2_source_model = DistributedDataParallel(q2_source_model)
        q2_source_model = nn.SyncBatchNorm.convert_sync_batchnorm(q2_source_model)

    def snapshot_writer(
            value_model: nn.Module, q1_source_model: nn.Module, q2_source_model: nn.Module,
            q1_target_model: QModel, q2_target_model: QModel, value_optimizer: Optimizer,
            q1_optimizer: Optimizer, q2_optimizer: Optimizer,
            num_samples: Optional[int]=None) -> None:
        if isinstance(value_model, DistributedDataParallel):
            value_model = value_model.module
        if isinstance(q1_source_model, DistributedDataParallel):
            q1_source_model = q1_source_model.module
        if isinstance(q2_source_model, DistributedDataParallel):
            q2_source_model = q2_source_model.module

        infix = '' if num_samples is None else f'.{num_samples}'

        torch.save(value_model.state_dict(), snapshots_path / f'value{infix}.pth')
        torch.save(q1_source_model.state_dict(), snapshots_path / f'q1-source{infix}.pth')
        torch.save(q2_source_model.state_dict(), snapshots_path / f'q2-source{infix}.pth')
        torch.save(q1_target_model.state_dict(), snapshots_path / f'q1-target{infix}.pth')
        torch.save(q2_target_model.state_dict(), snapshots_path / f'q2-target{infix}.pth')
        torch.save(value_optimizer.state_dict(), snapshots_path / f'value-optimizer{infix}.pth')
        torch.save(q1_optimizer.state_dict(), snapshots_path / f'q1-optimizer{infix}.pth')
        torch.save(q2_optimizer.state_dict(), snapshots_path / f'q2-optimizer{infix}.pth')
        torch.save(v_lr_scheduler.state_dict(), snapshots_path / f'v_lr_scheduler{infix}.pth')
        torch.save(q1_lr_scheduler.state_dict(), snapshots_path / f'q1_lr_scheduler{infix}.pth')
        torch.save(q2_lr_scheduler.state_dict(), snapshots_path / f'q2_lr_scheduler{infix}.pth')

        q_model = QQModel(q1_model=q1_target_model, q2_model=q2_target_model)
        state = dump_object(
            q_model,
            [
                dump_object(
                    q_model.q1_model,
                    [
                        dump_model(
                            q_model.q1_model.encoder,
                            [],
                            {
                                'position_encoder': config.encoder.position_encoder,
                                'dimension': config.encoder.dimension,
                                'num_heads': config.encoder.num_heads,
                                'dim_feedforward': config.encoder.dim_feedforward,
                                'activation_function': config.encoder.activation_function,
                                'dropout': config.encoder.dropout,
                                'num_layers': config.encoder.num_layers,
                                'checkpointing': config.checkpointing,
                                'device': config.device.type,
                                'dtype': dtype
                            }
                        ),
                        dump_model(
                            q_model.q1_model.decoder,
                            [],
                            {
                                'dimension': config.encoder.dimension,
                                'dim_feedforward': config.decoder.dim_feedforward,
                                'activation_function': config.decoder.activation_function,
                                'dropout': config.decoder.dropout,
                                'num_layers': config.decoder.num_layers,
                                'device': config.device.type,
                                'dtype': dtype
                            }
                        )
                    ],
                    {}
                ),
                dump_object(
                    q_model.q2_model,
                    [
                        dump_model(
                            q_model.q2_model.encoder,
                            [],
                            {
                                'position_encoder': config.encoder.position_encoder,
                                'dimension': config.encoder.dimension,
                                'num_heads': config.encoder.num_heads,
                                'dim_feedforward': config.encoder.dim_feedforward,
                                'num_layers': config.encoder.num_layers,
                                'activation_function': config.encoder.activation_function,
                                'dropout': config.encoder.dropout,
                                'checkpointing': config.checkpointing,
                                'device': config.device.type,
                                'dtype': dtype
                            }
                        ),
                        dump_model(
                            q_model.q2_model.decoder,
                            [],
                            {
                                'dimension': config.encoder.dimension,
                                'dim_feedforward': config.decoder.dim_feedforward,
                                'activation_function': config.decoder.activation_function,
                                'dropout': config.decoder.dropout,
                                'num_layers': config.decoder.num_layers,
                                'device': config.device.type,
                                'dtype': dtype
                            }
                        )
                    ],
                    {}
                )
            ],
            {}
        )
        torch.save(state, snapshots_path / f'q-model{infix}.kanachan')

    with SummaryWriter(log_dir=tensorboard_path) as summary_writer:
        _training(
            is_multiprocess=is_multiprocess, world_size=world_size, rank=rank,
            is_main_process=is_main_process, training_data=config.training_data,
            num_workers=config.num_workers, device=config.device.type, dtype=dtype,
            amp_dtype=amp_dtype, value_model=value_model, q1_source_model=q1_source_model,
            q2_source_model=q2_source_model, q1_target_model=q1_target_model,
            q2_target_model=q2_target_model, reward_plugin=config.reward_plugin,
            discount_factor=config.discount_factor, expectile=config.expectile,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            v_max_gradient_norm=config.v_max_gradient_norm,
            q_max_gradient_norm=config.q_max_gradient_norm, value_optimizer=value_optimizer,
            q1_optimizer=q1_optimizer, q2_optimizer=q2_optimizer, v_lr_scheduler=v_lr_scheduler,
            q1_lr_scheduler=q1_lr_scheduler, q2_lr_scheduler=q2_lr_scheduler,
            target_update_interval=config.target_update_interval,
            target_update_rate=config.target_update_rate,
            snapshot_interval=config.snapshot_interval, num_samples=num_samples,
            summary_writer=summary_writer, snapshot_writer=snapshot_writer)


if __name__ == '__main__':
    _main() # pylint: disable=no-value-for-parameter
    sys.exit(0)
