#!/usr/bin/env python3

import sys
from omegaconf import DictConfig
import hydra
import torch
from torch import nn
from kanachan.training.constants import MAX_NUM_ACTION_CANDIDATES
import kanachan.training.bert.phase1.config # type: ignore pylint: disable=unused-import
from kanachan.training.bert.phase1.decoder import Decoder
from kanachan.training.bert.phase1.model import Model
from kanachan.training.bert.phase2.iterator_adaptor import IteratorAdaptor
from kanachan.training.bert import training


@hydra.main(version_base=None, config_name='config')
def _main(config: DictConfig) -> None:
    _loss_function = nn.MSELoss()

    def loss_function(
            prediction: torch.Tensor, index: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert prediction.dim() == 2
        batch_size = prediction.size(0)
        assert prediction.size(1) == MAX_NUM_ACTION_CANDIDATES
        assert index.dim() == 1
        assert index.size(0) == batch_size
        assert target.dim() == 1
        assert target.size(0) == batch_size
        prediction = prediction[torch.arange(batch_size), index]
        return _loss_function(prediction.to(dtype=torch.float32), target.to(dtype=torch.float32))

    def prediction_function(weights: torch.Tensor) -> torch.Tensor:
        assert weights.dim() == 2
        assert weights.size(1) == MAX_NUM_ACTION_CANDIDATES
        return torch.argmax(weights, dim=1)

    training.main(
        config=config, iterator_adaptor_type=IteratorAdaptor, decoder_type=Decoder,
        model_type=Model, loss_function=loss_function, prediction_function=prediction_function)


if __name__ == '__main__':
    _main() # pylint: disable=no-value-for-parameter
    sys.exit(0)
