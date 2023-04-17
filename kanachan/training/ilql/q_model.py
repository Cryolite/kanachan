#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from kanachan.training.ilql.qv_model import QVModel


class QModel(nn.Module):
    def __init__(self, qv1_model: QVModel, qv2_model: QVModel) -> None:
        super(QModel, self).__init__()

        if isinstance(qv1_model, DistributedDataParallel):
            self.qv1_model = qv1_model.module
        else:
            self.qv1_model = qv1_model
        if isinstance(qv2_model, DistributedDataParallel):
            self.qv2_model = qv2_model.module
        else:
            self.qv2_model = qv2_model

    def forward(
            self, sparse: torch.Tensor, numeric: torch.Tensor, progression: torch.Tensor,
            candidates: torch.Tensor) -> torch.Tensor:
        q1, _ = self.qv1_model(sparse, numeric, progression, candidates)
        q2, _ = self.qv2_model(sparse, numeric, progression, candidates)
        return torch.minimum(q1, q2)
