#!/usr/bin/env python3

import re
import datetime
import pathlib
import os
from argparse import ArgumentParser
import logging
import sys
import torch
from torch import backends
from torch import nn
from torch.utils import tensorboard
from kanachan import common
import kanachan.round_result.common as round_result_common


class Decoder(common.DecoderBase):
    def __init__(self, dimension, num_heads, num_layers, sparse) -> None:
        super(Decoder, self).__init__(
            common.NUM_POST_ZIMO_ACTIONS, dimension, num_heads, num_layers, sparse)
        self.__activation = nn.ReLU()
        self.__full_connection = nn.Linear(dimension, round_result_common.NUM_ROUND_RESULT_CATEGORIES)

    def forward(self, encode, action):
        decode = super(Decoder, self).forward(encode, action)
        decode = self.__activation(decode)
        decode = torch.flatten(decode, start_dim=1)
        prediction = self.__full_connection(decode)
        return prediction


class Model(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(Model, self).__init__()
        self.__encoder = encoder
        self.__decoder = decoder

    def forward(self, sparse_feature, float_feature, action):
        encode = self.__encoder(sparse_feature, float_feature)
        prediction = self.__decoder(encode, action)
        return prediction


if __name__ == '__main__':
    ap = ArgumentParser(
        description='Train a model predicting the round result by post-zimo action.')
    ap.add_argument_group(title='Data')
    ap.add_argument('--training-data', type=pathlib.Path, required=True, metavar='PATH')
    ap.add_argument('--validation-data', type=pathlib.Path, metavar='PATH')
    ap.add_argument('--data-prefetch-size', default=256, type=int, metavar='N')
    ap.add_argument_group(title='Device')
    ap.add_argument('--device', metavar='DEV')
    ap.add_argument(
        '--dtype', default='float32', choices=('float16','float32', 'float64'),
        metavar='TYPE')
    ap.add_argument_group(title='Model')
    ap.add_argument('--dimension', default=128, type=int, metavar='N')
    ap.add_argument('--num-heads', default=8, type=int, metavar='N')
    ap.add_argument('--num-layers', default=5, type=int, metavar='N')
    ap.add_argument_group(title='Training')
    ap.add_argument('--batch-size', default=32, type=int, metavar='N')
    ap.add_argument('--max-epoch', default=100, type=int, metavar='N')
    ap.add_argument('--optimizer', default='sgd', choices=('adam', 'sgd'))
    ap.add_argument('--learning-rate', type=float, metavar='LR')
    ap.add_argument_group(title='Output')
    ap.add_argument('--output-prefix', type=pathlib.Path, required=True, metavar='PATH')
    ap.add_argument('--experiment-name', metavar='NAME')
    ap.add_argument('--resume', action='store_true')

    config = ap.parse_args()
    if config.device is not None:
        if re.search('^(?:cpu)|(?:cuda(?::\\d+)?)', config.device) is None:
            raise RuntimeError(f'{config.device}: invalid device.')
        device = config.device
    elif backends.cuda.is_built():
        device = 'cuda'
    else:
        device = 'cpu'
    if not config.training_data.exists():
        raise RuntimeError(f'{config.training_data}: does not exist.')
    if config.validation_data is not None and not config.validation_data.exists():
        raise RuntimeError(f'{config.validation_data}: does not exist.')
    if config.dtype == 'float16':
        dtype = torch.float16
    elif config.dtype == 'float32':
        dtype = torch.float32
    else:
        assert(config.dtype == 'float64')
        dtype = torch.float64
    if config.dimension < 1:
        raise RuntimeError(f'{config.dimension}: invalid dimension.')

    iterator_adaptor = lambda fp: round_result_common.IteratorAdaptor(
        fp, config.dimension, config.data_prefetch_size,
        device=device, dtype=dtype)

    if config.optimizer == 'sgd':
        sparse = True
    else:
        assert(config.optimizer == 'adam')
        sparse = False
    encoder = common.Encoder(config.dimension, config.num_heads, config.num_layers, sparse)
    decoder = Decoder(config.dimension, config.num_heads, config.num_layers, sparse)
    model = Model(encoder, decoder)
    model.to(device=device, dtype=dtype)

    loss_function = nn.CrossEntropyLoss()

    if config.optimizer == 'adam':
        if config.learning_rate is None:
            learning_rate = 0.001
        else:
            learning_rate = config.learning_rate
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        assert(config.optimizer == 'sgd')
        if config.learning_rate is None:
            learning_rate = 0.1
        else:
            learning_rate = config.learning_rate
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    if config.experiment_name is None:
        now = datetime.datetime.now()
        experiment_name = now.strftime('%Y-%m-%d-%H-%M-%S')
    else:
        experiment_name = config.experiment_name

    experiment_path = pathlib.Path(config.output_prefix / experiment_name)
    if experiment_path.exists() and not config.resume:
        raise RuntimeError(f'{experiment_path}: already exists. Did you mean `--resume`?')
    snapshots_path = experiment_path / 'snapshots'
    tensorboard_path = config.output_prefix / 'tensorboard' / experiment_name

    common.initialize_logging(experiment_path / 'training.log')

    logging.info(f'training data: {config.training_data}')
    if config.validation_data is None:
        logging.info(f'Validation data: N/A')
    else:
        logging.info(f'Validation data: {config.validation_data}')
    logging.info(f'Data prefetch size: {config.data_prefetch_size}')
    logging.info(f'Device: {device}')
    if backends.cudnn.is_available():
        logging.info(f'cuDNN: available')
    else:
        logging.info(f'cuDNN: N/A')
    logging.info(f'dtype: {dtype}')
    logging.info(f'Dimension: {config.dimension}')
    logging.info(f'# of heads: {config.num_heads}')
    logging.info(f'# of layers: {config.num_layers}')
    logging.info(f'Batch size: {config.batch_size}')
    logging.info(f'Max epoch: {config.max_epoch}')
    logging.info(f'Optimizer: {config.optimizer}')
    logging.info(f'Learning rate: {learning_rate}')
    if config.resume:
        logging.info(f'Resume from {experiment_path}')
    else:
        logging.info(f'Experiment output: {experiment_path}')

    epoch = 0
    if config.resume:
        if not snapshots_path.exists():
            raise RuntimeError(f'{snapshots_path}: does not exist.')
        for child in os.listdir(snapshots_path):
            m = re.search('^encoder\\.(\\d+)\\.pth$', child)
            if m is None:
                continue
            if epoch is None or int(m[1]) > epoch:
                epoch = int(m[1])
        latest_encoder_snapshot_path = snapshots_path / f'encoder.{epoch}.pth'
        latest_decoder_snapshot_path = snapshots_path / f'decoder.{epoch}.pth'
        if not latest_decoder_snapshot_path.exists():
            raise RuntimeError(
                f'{latest_decoder_snapshot_path}: does not exist.')
        encoder.load_state_dict(torch.load(latest_encoder_snapshot_path))
        decoder.load_state_dict(torch.load(latest_decoder_snapshot_path))

    with tensorboard.SummaryWriter(log_dir=tensorboard_path) as writer:
        experiment_path.mkdir(parents=True, exist_ok=True)
        tensorboard_path.mkdir(parents=True, exist_ok=True)
        if not config.resume:
            (experiment_path / 'tensorboard').symlink_to(
                f'../tensorboard/{experiment_name}', target_is_directory=True)

        if config.validation_data is not None:
            validation_loss = common.get_validation_loss(
                config.validation_data, iterator_adaptor, model, loss_function,
                config.batch_size)
            logging.info(f'epoch = {epoch}, validation loss = {validation_loss}')
            writer.add_scalar('Validation epoch loss', validation_loss, epoch)

        epoch += 1

        for i in range(epoch, config.max_epoch + 1):
            validation_loss = common.training_epoch(
                config.training_data, config.validation_data, iterator_adaptor,
                model, loss_function, optimizer, config.batch_size, writer, i)
            snapshots_path.mkdir(parents=False, exist_ok=True)
            torch.save(encoder.state_dict(), snapshots_path / f'encoder.{i}.pth')
            torch.save(decoder.state_dict(), snapshots_path / f'decoder.{i}.pth')

            if validation_loss is not None:
                best_tsv_path = experiment_path / 'best.tsv'
                if not best_tsv_path.exists():
                    with open(best_tsv_path, 'w') as f:
                        f.write(f'{i}\t{validation_loss}\n')
                    best_epoch = i
                else:
                    with open(best_tsv_path, 'r') as f:
                        best_epoch, best_loss = f.read().rstrip('\n').split('\t')
                    best_epoch, best_loss = int(best_epoch), float(best_loss)
                    if validation_loss < best_loss:
                        with open(best_tsv_path, 'w') as f:
                            f.write(f'{i}\t{validation_loss}\n')
                        best_epoch = i
                (experiment_path / 'encoder.best.pth').unlink(missing_ok=True)
                (experiment_path / 'decoder.best.pth').unlink(missing_ok=True)
                (experiment_path / 'encoder.best.pth').symlink_to(
                    f'snapshots/encoder.{best_epoch}.pth', target_is_directory=False)
                (experiment_path / 'decoder.best.pth').symlink_to(
                    f'snapshots/decoder.{best_epoch}.pth', target_is_directory=False)

    sys.exit(0)
