import math
from collections import OrderedDict
import torch
from torch import nn
from kanachan.training.constants import (
    NUM_TYPES_OF_ACTIONS, MAX_NUM_ACTION_CANDIDATES, ENCODER_WIDTH
)
from kanachan.training.bert.encoder import Encoder
from kanachan.training.iql.value_model import ValueDecoder


class ThetaDecoder(nn.Module):
    def __init__(
            self, *, dimension: int, dim_feedforward: int, activation_function: str, dropout: float,
            num_layers: int, num_qr_intervals: int, device: torch.device,
            dtype: torch.dtype) -> None:
        if dimension < 1:
            raise ValueError(dimension)
        if dim_feedforward < 1:
            raise ValueError(dim_feedforward)
        if activation_function not in ('relu', 'gelu'):
            raise ValueError(activation_function)
        if dropout < 0.0 or 1.0 <= dropout:
            raise ValueError(dropout)
        if num_layers < 1:
            raise ValueError(num_layers)
        if num_qr_intervals < 1:
            raise ValueError(num_qr_intervals)

        super(ThetaDecoder, self).__init__()

        self.value_decoder = ValueDecoder(
            dimension=dimension, dim_feedforward=dim_feedforward,
            activation_function=activation_function, dropout=dropout, num_layers=num_layers,
            device=device, dtype=dtype)

        self.layers_list = nn.ModuleList()
        for _ in range(num_qr_intervals):
            layers = OrderedDict()
            for i in range(num_layers - 1):
                layers[f'layer{i}'] = nn.Linear(
                    dimension if i == 0 else dim_feedforward, dim_feedforward,
                    device=device, dtype=dtype)
                if activation_function == 'relu':
                    layers[f'activation{i}'] = nn.ReLU()
                elif activation_function == 'gelu':
                    layers[f'activation{i}'] = nn.GELU()
                else:
                    raise ValueError(activation_function)
                layers[f'dropout{i}'] = nn.Dropout(p=dropout)
            suffix = '' if num_layers == 1 else str(num_layers - 1)
            layers['layer' + suffix] = nn.Linear(
                dimension if num_layers == 1 else dim_feedforward, 1, device=device, dtype=dtype)
            self.layers_list.append(nn.Sequential(layers))

    def forward(
            self, candidates: torch.Tensor, encode: torch.Tensor) -> torch.Tensor:
        assert candidates.dim() == 2
        assert encode.dim() == 3
        assert encode.size(1) == ENCODER_WIDTH
        assert candidates.size(0) == encode.size(0)
        
        value: torch.Tensor = self.value_decoder(candidates, encode)
        assert value.dim() == 1
        assert value.size(0) == candidates.size(0)
        value = torch.unsqueeze(value, dim=1)
        value = value.expand(-1, MAX_NUM_ACTION_CANDIDATES)

        encode = encode[:, -MAX_NUM_ACTION_CANDIDATES:]

        theta = torch.zeros(
            (candidates.size(0), len(self.layers_list), MAX_NUM_ACTION_CANDIDATES),
            device=encode.device, dtype=encode.dtype)
        for i, layers in enumerate(self.layers_list):
            i: int
            layers: nn.Module
            advantage: torch.Tensor = layers(encode)
            advantage = torch.squeeze(advantage, dim=2)
            assert advantage.dim() == 2
            assert advantage.size(0) == candidates.size(0)
            assert advantage.size(1) == MAX_NUM_ACTION_CANDIDATES
            q = value + advantage
            theta[:, i, :] = torch.where(candidates < NUM_TYPES_OF_ACTIONS, q, -math.inf)

        return theta


class ThetaModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: ThetaDecoder) -> None:
        super(ThetaModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
            self, sparse: torch.Tensor, numeric: torch.Tensor, progression: torch.Tensor,
            candidates: torch.Tensor) -> torch.Tensor:
        encode = self.encoder(sparse, numeric, progression, candidates)
        return self.decoder(candidates, encode)


class QModel(nn.Module):
    def __init__(self, theta_model: ThetaModel) -> None:
        super(QModel, self).__init__()
        self.theta = theta_model

    def forward(
            self, sparse: torch.Tensor, numeric: torch.Tensor, progression: torch.Tensor,
            candidates: torch.Tensor) -> torch.Tensor:
        theta: torch.Tensor = self.theta(sparse, numeric, progression, candidates)

        q = torch.sum(theta * (1.0 / len(self.theta.decoder.layers_list)), dim=1)
        assert q.dim() == 2
        assert q.size(0) == sparse.size(0)
        assert q.size(1) == MAX_NUM_ACTION_CANDIDATES

        return q
