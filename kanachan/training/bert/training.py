import re
import datetime
import math
from pathlib import Path
import os
import logging
from typing import Optional, Tuple, Type, Callable
import sys
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import torch
from torch import backends
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer, SGD, Adam, RAdam
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, all_reduce, barrier
from torch.utils.tensorboard.writer import SummaryWriter
from apex.optimizers import FusedAdam, FusedSGD, FusedLAMB
from kanachan.training.common import Dataset
from kanachan.training.iterator_adaptor_base import IteratorAdaptorBase
from kanachan.training.bert.encoder import Encoder
from kanachan.model_loader import dump_model, dump_object


LossFunction = Callable[[torch.Tensor, float], torch.Tensor]
PredictionFunction = Callable[[torch.Tensor], torch.Tensor]


def _validate(
        *, is_multiprocess: bool, world_size: Optional[int], rank: Optional[int],
        is_main_process: bool, device: str, validation_data: Path,
        iterator_adaptor_type: Type[IteratorAdaptorBase], num_workers: int, batch_size: int,
        model: nn.Module, loss_function: LossFunction,
        prediction_function: PredictionFunction) -> Tuple[float, float]:
    start_time = datetime.datetime.now()

    # Prepare the validation data loader. Note that this data loader must
    # iterate the validation data set only once.
    dataset = Dataset(validation_data, iterator_adaptor_type)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        pin_memory=(num_workers >= 1), drop_last=is_multiprocess)

    validation_loss = 0.0
    num_correct_predictions = 0
    batch_count = 0

    for annotation in data_loader:
        if is_multiprocess:
            assert world_size is not None
            assert rank is not None
            assert batch_size % world_size == 0
            first = (batch_size // world_size) * rank
            last = (batch_size // world_size) * (rank + 1)
            annotation = tuple(x[first:last] for x in annotation)

        if device != 'cpu':
            annotation = tuple(x.cuda() for x in annotation)

        weights = model(*(annotation[:4]))
        loss = loss_function(weights, *(annotation[4:]))
        prediction = prediction_function(weights)

        if math.isnan(loss.item()):
            raise RuntimeError('Validation loss becomes NaN.')
        if is_multiprocess:
            all_reduce(loss)
            loss /= world_size
        validation_loss += loss.item()

        num_correct_predictions += torch.sum((prediction == annotation[4]).long()).item()

        batch_count += 1

    validation_loss /= batch_count
    precision = num_correct_predictions / (batch_size * batch_count)

    elapsed_time = datetime.datetime.now() - start_time
    if is_main_process:
        logging.info('Validation has finished (elapsed time = %f).', elapsed_time)
        logging.info('Validation loss = %E', validation_loss)

    return validation_loss, precision


SnapshotWriter = Callable[[nn.Module, nn.Module, Optimizer, Optional[int]], None]


def _train(
        *, is_multiprocess: bool, world_size: Optional[int], rank: Optional[int],
        is_main_process: bool, training_data: Path,
        iterator_adaptor_type: Type[IteratorAdaptorBase], num_workers: int, device: str,
        dtype: torch.dtype, amp_dtype: torch.dtype, encoder: Encoder, decoder: nn.Module,
        model: nn.Module, loss_function: LossFunction, batch_size: int,
        gradient_accumulation_steps: int, max_gradient_norm: float, optimizer: Optimizer,
        snapshot_interval: int, num_samples: int, summary_writer: SummaryWriter,
        snapshot_writer: SnapshotWriter) -> None:
    start_time = datetime.datetime.now()

    # Prepare the training data loader. Note that this data loader must iterate
    # the training data set only once.
    dataset = Dataset(training_data, iterator_adaptor_type)
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

    for annotation in data_loader:
        if num_consumed_samples < num_samples:
            num_consumed_samples = (int(num_samples/batch_size)+1)*batch_size
            continue

        if is_multiprocess:
            barrier()

        if is_multiprocess:
            assert world_size is not None
            assert rank is not None
            assert batch_size % world_size == 0
            first = (batch_size // world_size) * rank
            last = (batch_size // world_size) * (rank + 1)
            annotation = tuple(x[first:last] for x in annotation)

        if device != 'cpu':
            annotation = tuple(x.cuda() for x in annotation)

        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device != 'cpu' and dtype != amp_dtype)):
            prediction = model(*(annotation[:4]))
        loss = loss_function(prediction, *(annotation[4:]))
        if math.isnan(loss.item()):
            raise RuntimeError('Training loss becomes NaN.')

        batch_loss = loss
        if is_multiprocess:
            all_reduce(batch_loss)
            batch_loss /= world_size
        batch_loss = batch_loss.item()

        loss = loss / gradient_accumulation_steps
        if grad_scaler is None:
            loss.backward()
        else:
            grad_scaler.scale(loss).backward()

        num_samples += batch_size
        num_consumed_samples += batch_size
        batch_count += 1

        if batch_count % gradient_accumulation_steps == 0:
            if grad_scaler is not None:
                grad_scaler.unscale_(optimizer)
            gradient = nn.utils.parameters_to_vector(model.parameters())
            gradient_norm: float = torch.linalg.vector_norm(gradient).item()
            nn.utils.clip_grad_norm_(
                model.parameters(), max_gradient_norm, error_if_nonfinite=False)
            if grad_scaler is None:
                optimizer.step()
            else:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            optimizer.zero_grad()

            if is_main_process:
                logging.info(
                    'sample = %d, training loss = %E, gradient norm = %E',
                    num_samples, batch_loss, gradient_norm)
            summary_writer.add_scalar('Training loss', batch_loss, num_samples)
            summary_writer.add_scalar('Gradient norm', gradient_norm, num_samples)
        else:
            if is_main_process:
                logging.info('sample = %d, training loss = %E', num_samples, batch_loss)
            summary_writer.add_scalar('Training loss', batch_loss, num_samples)

        if is_main_process and last_snapshot is not None and num_samples - last_snapshot >= snapshot_interval:
            snapshot_writer(encoder, decoder, optimizer, num_samples)
            last_snapshot = num_samples

    if is_multiprocess:
        barrier()

    elapsed_time = datetime.datetime.now() - start_time

    if is_main_process:
        logging.info('A training epoch has finished (elapsed time = %s).', elapsed_time)
        snapshot_writer(encoder, decoder, optimizer)


def main(
        *, config: DictConfig, iterator_adaptor_type: Type[IteratorAdaptorBase],
        decoder_type: Type[nn.Module], model_type: Type[nn.Module], loss_function: LossFunction,
        prediction_function: PredictionFunction) -> None:
    if 'LOCAL_RANK' in os.environ:
        if os.environ['WORLD_SIZE'] != os.environ['LOCAL_WORLD_SIZE']:
            raise RuntimeError('Multi-node not supported')
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

    if config.validation_data is not None:
        if not config.validation_data.exists():
            raise RuntimeError(f'{config.validation_data}: Does not exist.')
        if not config.validation_data.is_file():
            raise RuntimeError(f'{config.validation_data}: Not a file.')

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

    if config.initial_model is not None:
        if config.encoder.load_from is not None:
            raise RuntimeError('`initial_model` conflicts with `encoder.load_from`.')
        if config.decoder.load_from is not None:
            raise RuntimeError('`initial_model` conflicts with `decoder.load_from`.')
        if not config.initial_model.exists():
            raise RuntimeError(f'{config.initial_model}: Does not exist.')
        if not config.initial_model.is_file():
            raise RuntimeError(f'{config.initial_model}: Not a file.')

    if config.initial_model_prefix is not None:
        if config.encoder.load_from is not None:
            raise RuntimeError('`initial_model_prefix` conflicts with `encoder.load_from`.')
        if config.decoder.load_from is not None:
            raise RuntimeError('`initial_model_prefix` conflicts with `decoder.load_from`.')
        if config.initial_model is not None:
            raise RuntimeError('`initial_model_prefix` conflicts with `initial_model`.')
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
        assert config.initial_model is None

        if config.initial_model_index is None:
            for child in os.listdir(config.initial_model_prefix):
                match = re.search('^(?:encoder|decoder|optimizer)(?:\\.(\\d+))?\\.pth$', child)
                if match is None:
                    continue
                if match[1] is None:
                    config.initial_model_index = sys.maxsize
                    continue
                if int(match[1]) > config.initial_model_index:
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

        encoder_snapshot_path = config.initial_model_prefix / f'encoder{infix}.pth'
        if not encoder_snapshot_path.exists():
            raise RuntimeError(f'{encoder_snapshot_path}: Does not exist.')
        if not encoder_snapshot_path.is_file():
            raise RuntimeError(f'{encoder_snapshot_path}: Not a file.')

        decoder_snapshot_path = config.initial_model_prefix / f'decoder{infix}.pth'
        if not decoder_snapshot_path.exists():
            raise RuntimeError(f'{decoder_snapshot_path}: Does not exist.')
        if not decoder_snapshot_path.is_file():
            raise RuntimeError(f'{decoder_snapshot_path}: Not a file.')

        optimizer_snapshot_path = config.initial_model_prefix / f'optimizer{infix}.pth'
        if not optimizer_snapshot_path.is_file() or config.optimizer.initialize:
            optimizer_snapshot_path = None

    if config.training_batch_size < 1:
        raise RuntimeError(f'{config.training_batch_size}: An invalid training batch size.')
    if config.training_batch_size % world_size != 0:
        raise RuntimeError(
            f'`training_batch_size` must be divisible by the world size ({world_size}).')

    if config.validation_data is not None and config.validation_batch_size is None:
        raise RuntimeError('Specify `validation_batch_size`.')
    if config.validation_data is None and config.validation_batch_size is not None:
        raise RuntimeError('`validation_batch_size` must be combined with `validation_data`.')
    if config.validation_batch_size is not None and config.validation_batch_size < 1:
        raise RuntimeError(f'{config.validation_batch_size}: An invalid validation batch size.')
    if config.validation_batch_size is not None and config.validation_batch_size % world_size != 0:
        raise RuntimeError(
            f'`validation_batch_size` must be divisible by the world size ({world_size}).')

    if config.gradient_accumulation_steps < 1:
        raise RuntimeError(
            f'{config.gradient_accumulation_steps}:'
            ' An invalid value for `gradient_accumulation_steps`')
    if config.max_gradient_norm <= 0.0:
        raise RuntimeError(
            f'{config.max_gradient_norm}: invalid norm for gradient clipping')

    if config.gradient_accumulation_steps < 1:
        raise RuntimeError(
            f'{config.gradient_accumulation_steps}: '
            'An invalid value for `gradient_accumulation_steps`.')
    if config.gradient_accumulation_steps >= 2 and config.optimizer.type == 'mtadam':
        raise RuntimeError('`mtadam` does not support for gradient accumulation.')

    if config.max_gradient_norm <= 0.0:
        raise RuntimeError(f'{config.max_gradient_norm}: An invalid value for `max_gradient_norm`.')

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
            if config.optimizer.type in ('adam', 'radam'):
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

    experiment_path = Path(HydraConfig.get().runtime.output_dir)

    tensorboard_path = experiment_path / 'tensorboard'
    if is_multiprocess:
        tensorboard_path /= str(rank).zfill(2)
    tensorboard_path.mkdir(parents=True, exist_ok=True)

    snapshots_path = experiment_path / 'snapshots'
    snapshots_path.mkdir(parents=True, exist_ok=True)

    if config.snapshot_interval < 0:
        raise RuntimeError(f'{config.snapshot_interval}: An invalid value for `snapshot_interval`.')

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
        if config.validation_data is None:
            logging.info('Validation data: N/A')
        else:
            logging.info('Validation data: %s', config.validation_data)
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
        if config.initial_model is not None:
            logging.info('Load model from: %s', config.initial_model)
        if config.initial_model_prefix is not None:
            logging.info('Initial model prefix: %s', config.initial_model_prefix)
            logging.info('Initlal model index: %d', config.initial_model_index)
            if config.optimizer.initialize:
                logging.info('(Will not load optimizer)')
        logging.info('Checkpointing: %s', config.checkpointing)
        if world_size is None:
            logging.info('Training batch size: %d', config.training_batch_size)
        else:
            logging.info('Local training batch size: %d', config.training_batch_size // world_size)
            logging.info('World training batch size: %d', config.training_batch_size)
        if config.validation_batch_size is not None:
            if world_size is None:
                logging.info('Validation batch size: %d', config.validation_batch_size)
            else:
                logging.info(
                    'Local validation batch size: %d', config.validation_batch_size // world_size)
                logging.info('World validation batch size: %d', config.validation_batch_size)
        logging.info('# of steps for gradient accumulation: %d', config.gradient_accumulation_steps)
        logging.info(
            'Virtual training batch size: %d',
            config.training_batch_size * config.gradient_accumulation_steps)
        logging.info('Norm threshold for gradient clipping: %E', config.max_gradient_norm)
        logging.info('Optimizer: %s', config.optimizer.type)
        if config.optimizer in ('sgd',):
            logging.info('Momentum factor: %f', config.optimizer.momentum)
        if config.optimizer in ('adam', 'radam', 'mtadam', 'lamb'):
            logging.info('Epsilon parameter: %E', config.optimizer.epsilon)
        logging.info('Learning rate: %E', config.optimizer.learning_rate)
        if config.initial_model_prefix is not None:
            logging.info('Initial encoder snapshot: %s', encoder_snapshot_path)
            logging.info('Initial decoder snapshot: %s', decoder_snapshot_path)
            if optimizer_snapshot_path is not None:
                logging.info('Initial optimizer snapshot: %s', optimizer_snapshot_path)
        logging.info('Experiment output: %s', experiment_path)
        if config.snapshot_interval == 0:
            logging.info('Snapshot interval: N/A')
        else:
            logging.info('Snapshot interval: %d', config.snapshot_interval)

    encoder = Encoder(
        position_encoder=config.encoder.position_encoder, dimension=config.encoder.dimension,
        num_heads=config.encoder.num_heads, dim_feedforward=config.encoder.dim_feedforward,
        activation_function=config.encoder.activation_function, dropout=config.encoder.dropout,
        num_layers=config.encoder.num_layers, checkpointing=config.checkpointing,
        device=config.device.type, dtype=dtype)
    decoder = decoder_type(
        dimension=config.encoder.dimension, dim_feedforward=config.decoder.dim_feedforward,
        activation_function=config.decoder.activation_function, dropout=config.decoder.dropout,
        num_layers=config.decoder.num_layers, device=config.device.type, dtype=dtype)
    model = model_type(encoder, decoder)
    model.to(device=config.device.type, dtype=dtype)

    if config.optimizer.type == 'sgd':
        if config.device.type == 'cpu':
            optimizer = SGD(
                model.parameters(), lr=config.optimizer.learning_rate,
                momentum=config.optimizer.momentum)
        else:
            optimizer = FusedSGD(
                model.parameters(), lr=config.optimizer.learning_rate,
                momentum=config.optimizer.momentum)
    elif config.optimizer.type == 'adam':
        if config.device.type == 'cpu':
            optimizer = Adam(
                model.parameters(), lr=config.optimizer.learning_rate, eps=config.optimizer.epsilon)
        else:
            optimizer = FusedAdam(
                model.parameters(), lr=config.optimizer.learning_rate, eps=config.optimizer.epsilon)
    elif config.optimizer.type == 'radam':
        optimizer = RAdam(
            model.parameters(), lr=config.optimizer.learning_rate, eps=config.optimizer.epsilon)
    elif config.optimizer.type == 'lamb':
        optimizer = FusedLAMB(
            model.parameters(), lr=config.optimizer.learning_rate, eps=config.optimizer.epsilon)
    else:
        raise NotImplementedError(config.optimizer.type)

    if config.encoder.load_from is not None:
        assert config.initial_model is None
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        encoder_state_dict = torch.load(config.encoder.load_from, map_location='cpu')
        encoder_new_state_dict = {}
        for key, value in encoder_state_dict.items():
            new_key = re.sub('^module\\.', '', key)
            encoder_new_state_dict[new_key] = value
        encoder.load_state_dict(encoder_new_state_dict)
        if config.device.type != 'cpu':
            encoder.cuda()

    if config.decoder.load_from is not None:
        assert config.initial_model is None
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        decoder_state_dict = torch.load(config.decoder.load_from, map_location='cpu')
        decoder_new_state_dict = {}
        for key, value in decoder_state_dict.items():
            new_key = re.sub('^module\\.', '', key)
            decoder_new_state_dict[new_key] = value
        decoder.load_state_dict(decoder_new_state_dict)
        if config.device.type != 'cpu':
            decoder.cuda()

    if config.initial_model is not None:
        assert config.encoder.load_from is None
        assert config.decoder.load_from is None
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        model_state_dict = torch.load(config.initial_model, map_location='cpu')
        model_new_state_dict = {}
        for key, value in model_state_dict.items():
            new_key = re.sub('^module\\.', '', key)
            model_new_state_dict[new_key] = value
        model.load_state_dict(model_new_state_dict)
        if config.device.type != 'cpu':
            model.cuda()

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None
        assert config.decoder.load_from is None
        assert config.initial_model is None

        encoder_state_dict = torch.load(encoder_snapshot_path, map_location='cpu')
        encoder_new_state_dict = {}
        for key, value in encoder_state_dict.items():
            new_key = re.sub('^module\\.', '', key)
            encoder_new_state_dict[new_key] = value
        encoder.load_state_dict(encoder_new_state_dict)
        if config.device.type != 'cpu':
            encoder.cuda()

        decoder_state_dict = torch.load(decoder_snapshot_path, map_location='cpu')
        decoder_new_state_dict = {}
        for key, value in decoder_state_dict.items():
            new_key = re.sub('^module\\.', '', key)
            decoder_new_state_dict[new_key] = value
        decoder.load_state_dict(decoder_new_state_dict)
        if config.device.type != 'cpu':
            decoder.cuda()

        if optimizer_snapshot_path is not None:
            optimizer_state_dict = torch.load(optimizer_snapshot_path, map_location='cpu')
            optimizer.load_state_dict(optimizer_state_dict)

    if is_multiprocess:
        init_process_group(backend='nccl')
        model = DistributedDataParallel(model)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    def snapshot_writer(
            encoder: nn.Module, decoder: nn.Module, optimizer: Optimizer,
            num_samples: Optional[int]=None) -> None:
        infix = '' if num_samples is None else f'.{num_samples}'

        torch.save(encoder.state_dict(), snapshots_path / f'encoder{infix}.pth')
        torch.save(decoder.state_dict(), snapshots_path / f'decoder{infix}.pth')
        torch.save(optimizer.state_dict(), snapshots_path / f'optimizer{infix}.pth')

        state = dump_object(
            model.module if isinstance(model, DistributedDataParallel) else model,
            [
                dump_model(
                    encoder,
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
                    decoder,
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
        torch.save(state, snapshots_path / f'model{infix}.kanachan')

    with SummaryWriter(log_dir=tensorboard_path) as summary_writer:
        _train(
            is_multiprocess=is_multiprocess, world_size=world_size, rank=rank,
            is_main_process=is_main_process, training_data=config.training_data,
            iterator_adaptor_type=iterator_adaptor_type, num_workers=config.num_workers,
            device=config.device.type, dtype=dtype, amp_dtype=amp_dtype, encoder=encoder,
            decoder=decoder, model=model, loss_function=loss_function,
            batch_size=config.training_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_gradient_norm=config.max_gradient_norm, optimizer=optimizer,
            snapshot_interval=config.snapshot_interval, num_samples=num_samples,
            summary_writer=summary_writer, snapshot_writer=snapshot_writer)

        if config.validation_data is not None:
            assert config.validation_batch_size is not None
            model.eval()
            with torch.no_grad():
                validation_loss, precision = _validate(
                    is_multiprocess=is_multiprocess, world_size=world_size, rank=rank,
                    is_main_process=is_main_process, device=config.device.type,
                    validation_data=config.validation_data,
                    iterator_adaptor_type=iterator_adaptor_type, num_workers=config.num_workers,
                    batch_size=config.validation_batch_size, model=model,
                    loss_function=loss_function, prediction_function=prediction_function)
            model.train()
            if is_main_process:
                logging.info('validation loss = %E, precision = %f', validation_loss, precision)
                summary_writer.add_scalar('Validation loss', validation_loss)
                summary_writer.add_scalar('Precision', precision)
