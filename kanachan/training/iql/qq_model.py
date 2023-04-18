import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from kanachan.training.iql.q_model import QModel


class QQModel(nn.Module):
    def __init__(self, q1_model: QModel, q2_model: QModel) -> None:
        super(QQModel, self).__init__()

        if isinstance(q1_model, DistributedDataParallel):
            self.q1_model = q1_model.module
        else:
            self.q1_model = q1_model
        if isinstance(q2_model, DistributedDataParallel):
            self.q2_model = q2_model.module
        else:
            self.q2_model = q2_model

    def forward(
            self, sparse: torch.Tensor, numeric: torch.Tensor, progression: torch.Tensor,
            candidates: torch.Tensor) -> torch.Tensor:
        q1 = self.q1_model(sparse, numeric, progression, candidates)
        q2 = self.q2_model(sparse, numeric, progression, candidates)
        return torch.minimum(q1, q2)
