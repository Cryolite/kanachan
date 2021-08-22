#!/usr/bin/env python3

import re
import datetime
import math
import pathlib
import os
from argparse import ArgumentParser
import logging
import sys
from typing import Optional
import torch
from torch import backends
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import tensorboard
from kanachan.constants import (
    MAX_LENGTH_OF_POSITIONAL_FEATURES, MAX_NUM_ACTION_CANDIDATES,)
from kanachan import common
from kanachan.common import (Dataset,)
from kanachan.encoder import Encoder
from kanachan.pretraining.iterator_adaptor import IteratorAdaptor
from apex import amp
from apex.optimizers import (FusedAdam, FusedSGD,)


class Decoder(nn.Module):
    def __init__(self, num_dimensions: int) -> None:
        super(Decoder, self).__init__()
        self.__linear = nn.Linear(num_dimensions, 1)

    def forward(self, encode):
        encode = encode[:, -MAX_NUM_ACTION_CANDIDATES:, :]
        prediction = self.__linear(encode)
        prediction = torch.squeeze(prediction, dim=2)
        return prediction


class Model(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super(Model, self).__init__()
        self.__encoder = encoder
        self.__decoder = decoder

    def forward(self, x):
        encode = self.__encoder(x)
        prediction = self.__decoder(encode)
        return prediction


def _training_epoch(
        data_loader: DataLoader, model: Model, optimizer,
        lr_scheduler: ReduceLROnPlateau, device: str, writer,
        num_batches: Optional[int], epoch: int) -> None:
    logging.info(f'The {epoch}-th epoch starts.')

    start_time = datetime.datetime.now()

    loss_function = nn.CrossEntropyLoss()

    training_loss = 0.0
    for batch_count, annotation in enumerate(data_loader):
        # The following assertion is sometimes violated.
        #assert(num_batches is None or batch_count < num_batches)

        if device != 'cpu':
            annotation = tuple(x.to(device=device) for x in annotation)

        prediction = model(annotation[:-1])
        loss = loss_function(prediction, annotation[-1])
        if math.isnan(loss.item()):
            raise RuntimeError('Training loss becomes NaN in training.')

        batch_loss = loss.item()
        training_loss += batch_loss

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        logging.info(f'epoch = {epoch}, batch = {batch_count}, training loss = {loss.item()}')
        if num_batches is None:
            writer.add_scalar('Training batch loss', batch_loss, batch_count)
        else:
            writer.add_scalar('Training batch loss', batch_loss, num_batches * epoch + batch_count)
    num_batches = batch_count

    elapsed_time = datetime.datetime.now() - start_time
    logging.info(f'The {epoch}-th epoch has finished (elapsed time = {elapsed_time}).')

    training_loss /= num_batches
    logging.info(f'epoch = {epoch}, training loss = {training_loss}')
    writer.add_scalar('Training epoch loss', training_loss, epoch)

    lr_scheduler.step(training_loss)

    return num_batches


if __name__ == '__main__':
    ap = ArgumentParser(
        description='Pre-train the encoder by imitating higher level players.')
    ap_data = ap.add_argument_group(title='Data')
    ap_data.add_argument(
        '--training-data', type=pathlib.Path, required=True,
        help='path to training data', metavar='PATH')
    ap_data.add_argument(
        '--num-workers', default=2, type=int,
        help='# of worker processes in data loading (defaults to 2)',
        metavar='NWORKERS')
    ap_device = ap.add_argument_group(title='Device')
    ap_device.add_argument('--device', help='device', metavar='DEV')
    ap_device.add_argument(
        '--dtype', default='float32', choices=('float16','float32'),
        help='floating point type (defaults to `float32`)')
    ap_model = ap.add_argument_group(title='Model')
    ap_model.add_argument(
        '--num-dimensions', default=768, type=int,
        help='# of embedding dimensions (defaults to 768)', metavar='DIM')
    ap_model.add_argument(
        '--num-heads', default=12, type=int, help='# of heads (defaults to 12)',
        metavar='NHEAD')
    ap_model.add_argument(
        '--num-layers', default=12, type=int,
        help='# of layers (defaults to 12)', metavar='NLAYERS')
    ap_model.add_argument(
        '--initial-encoder', type=pathlib.Path,
        help='path to the initial encoder; mutually exclusive to `--resume`',
        metavar='PATH')
    ap_model.add_argument(
        '--initial-decoder', type=pathlib.Path,
        help='path to the initial decoder; mutually exclusive to `--resume`',
        metavar='PATH')
    ap_training = ap.add_argument_group(title='Training')
    ap_training.add_argument(
        '--batch-size', default=32, type=int, help='batch size (defaults to 32)',
        metavar='N')
    ap_training.add_argument(
        '--max-epoch', default=100, type=int,
        help='the maximum epoch (defaults to 100)', metavar='N')
    ap_training.add_argument(
        '--optimizer', default='adam', choices=('adam', 'sgd'),
        help='optimizer (defaults to `adam`)')
    ap_training.add_argument(
        '--learning-rate', type=float,
        help='learning rate (defaults to 0.001 for `adam`, 0.01 for `sgd`)',
        metavar='LR')
    ap_training.add_argument(
        '--epsilon', default=1.0e-5, type=float,
        help='epsilon parameter; only meaningful for Adam (defaults to 1.0e-5)',
        metavar='EPS')
    ap_training.add_argument(
        '--momentum', default=0.9, type=float,
        help='momentum factor; only meaningful for SGD (defaults to 0.9)',
        metavar='MOMENTUM')
    ap_training.add_argument(
        '--dropout', default=0.1, type=float, help='defaults to 0.1',
        metavar='DROPOUT')
    ap_output = ap.add_argument_group(title='Output')
    ap_output.add_argument('--output-prefix', type=pathlib.Path, required=True, metavar='PATH')
    ap_output.add_argument('--experiment-name', metavar='NAME')
    ap_output.add_argument('--resume', action='store_true')

    config = ap.parse_args()

    if not config.training_data.exists():
        raise RuntimeError(f'{config.training_data}: does not exist.')
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
        raise RuntimeError(f'{config.dtype}: An invalid value for `--dtype`.')

    if config.num_dimensions < 1:
        raise RuntimeError(
            f'{config.num_dimensions}: An invalid number of dimensions.')
    if config.num_heads < 1:
        raise RuntimeError(f'{config.num_heads}: An invalid number of heads.')
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
        if config.momentum == 0.0:
            sparse = True
        else:
            # See https://github.com/pytorch/pytorch/issues/29814
            sparse = False
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
            learning_rate = 0.01
        else:
            learning_rate = config.learning_rate

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
    logging.info(f'# of workers: {config.num_workers}')
    logging.info(f'Device: {device}')
    if backends.cudnn.is_available():
        logging.info(f'cuDNN: available')
        backends.cudnn.benchmark = True
    else:
        logging.info(f'cuDNN: N/A')
    logging.info(f'dtype: {dtype}')
    logging.info(f'# of dimensions: {config.num_dimensions}')
    logging.info(f'# of heads: {config.num_heads}')
    logging.info(f'# of layers: {config.num_layers}')
    if config.initial_encoder is None and not config.resume:
        logging.info(f'Initial encoder: (initialized randomly)')
    elif config.initial_encoder is not None:
        logging.info(f'Initial encoder: {config.initial_encoder}')
    if config.initial_decoder is None and not config.resume:
        logging.info(f'Initial decoder: (initialized randomly)')
    elif config.initial_decoder is not None:
        logging.info(f'Initial decoder: {config.initial_decoder}')
    logging.info(f'Batch size: {config.batch_size}')
    logging.info(f'Max epoch: {config.max_epoch}')
    logging.info(f'Optimizer: {config.optimizer}')
    if config.optimizer == 'adam':
        logging.info(f'Epsilon parameter: {config.epsilon}')
    if config.optimizer == 'sgd':
        logging.info(f'Momentum factor: {config.momentum}')
    logging.info(f'Sparse: {sparse}')
    logging.info(f'Learning rate: {learning_rate}')
    logging.info(f'Dropout: {config.dropout}')
    if config.resume:
        logging.info(f'Resume from {experiment_path}')
    else:
        logging.info(f'Experiment output: {experiment_path}')

    config = {
        'training_data': config.training_data,
        'num_workers': config.num_workers,
        'device': device,
        'dtype': dtype,
        'num_dimensions': config.num_dimensions,
        'num_heads': config.num_heads,
        'num_layers': config.num_layers,
        'initial_encoder': config.initial_encoder,
        'initial_decoder': config.initial_decoder,
        'batch_size': config.batch_size,
        'max_epoch': config.max_epoch,
        'optimizer': config.optimizer,
        'epsilon': config.epsilon,
        'momentum': config.momentum,
        'sparse': sparse,
        'learning_rate': learning_rate,
        'dropout': config.dropout,
        'experiment_name': experiment_name,
        'experiment_path': experiment_path,
        'snapshots_path': snapshots_path,
        'tensorboard_path': tensorboard_path,
        'resume': config.resume,
    }

    encoder = Encoder(
        config['num_dimensions'], config['num_heads'], config['num_layers'],
        dropout=config['dropout'], sparse=config['sparse'])
    decoder = Decoder(config['num_dimensions'])
    model = Model(encoder, decoder)
    model.to(device=config['device'], dtype=config['dtype'])

    if config['optimizer'] == 'adam':
        optimizer = FusedAdam(
            model.parameters(), lr=config['learning_rate'],
            eps=config['epsilon'])
    else:
        assert(config['optimizer'] == 'sgd')
        optimizer = FusedSGD(
            model.parameters(), lr=config['learning_rate'],
            momentum=config['momentum'])

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=0)

    epoch = 0
    if config['resume']:
        if not config['snapshots_path'].exists():
            raise RuntimeError(f'{config["snapshots_path"]}: does not exist.')
        for child in os.listdir(config['snapshots_path']):
            m = re.search('^encoder\\.(\\d+)\\.pth$', child)
            if m is None:
                continue
            if int(m[1]) > epoch:
                epoch = int(m[1])
        latest_encoder_snapshot_path = config['snapshots_path'] / f'encoder.{epoch}.pth'
        encoder.load_state_dict(torch.load(latest_encoder_snapshot_path))
        latest_decoder_snapshot_path = config['snapshots_path'] / f'decoder.{epoch}.pth'
        decoder.load_state_dict(torch.load(latest_decoder_snapshot_path))
        epoch += 1

    if config['initial_encoder'] is not None:
        encoder.load_state_dict(torch.load(config['initial_encoder']))
    if config['initial_decoder'] is not None:
        decoder.load_state_dict(torch.load(config['initial_decoder']))

    with tensorboard.SummaryWriter(log_dir=config['tensorboard_path']) as writer:
        config['tensorboard_path'].mkdir(parents=True, exist_ok=True)
        if not config['resume']:
            (config['experiment_path'] / 'tensorboard').symlink_to(
                f'../tensorboard/{config["experiment_name"]}',
                target_is_directory=True)

        num_batches = None
        for i in range(epoch, config['max_epoch'] + 1):
            # Prepare the data loader. Note that this data loader must be
            # created anew for each epoch.
            iterator_adaptor = lambda fp: IteratorAdaptor(
                fp, config['num_dimensions'], config['dtype'])
            dataset = Dataset(config['training_data'], iterator_adaptor)
            data_loader = DataLoader(
                dataset, batch_size=config['batch_size'],
                num_workers=config['num_workers'], pin_memory=True)

            num_batches = _training_epoch(
                data_loader, model, optimizer, lr_scheduler, config['device'],
                writer, num_batches, i)
            config['snapshots_path'].mkdir(parents=False, exist_ok=True)
            torch.save(encoder.state_dict(), config['snapshots_path'] / f'encoder.{i}.pth')
            torch.save(decoder.state_dict(), config['snapshots_path'] / f'decoder.{i}.pth')

    sys.exit(0)
