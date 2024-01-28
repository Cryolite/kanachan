import torch
from torch import Tensor, nn
from kanachan.constants import MAX_NUM_ACTIVE_SPARSE_FEATURES, NUM_NUMERIC_FEATURES


class FeatureAdaptor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, sparse: Tensor, numeric: Tensor) -> tuple[Tensor, Tensor]:
        assert sparse.dim() == 2
        batch_size = sparse.size(0)
        assert sparse.size(1) == MAX_NUM_ACTIVE_SPARSE_FEATURES
        assert numeric.dim() == 2
        assert numeric.size(0) == batch_size
        assert numeric.size(1) == NUM_NUMERIC_FEATURES

        _sparse = torch.zeros(batch_size, 8, device=sparse.device, dtype=torch.int32)
        _sparse[:, 0] = sparse[:, 0] # room
        _sparse[:, 1] = sparse[:, 1] - 5 + 5 # game style
        _sparse[:, 2] = sparse[:, 2] - 7 + 7 # grading[0]
        _sparse[:, 3] = sparse[:, 3] - 23 + 23 # grading[1]
        _sparse[:, 4] = sparse[:, 4] - 39 + 39 # grading[2]
        _sparse[:, 5] = sparse[:, 5] - 55 + 55 # grading[3]
        _sparse[:, 6] = sparse[:, 6] - 75 + 71 # chang (game wind)
        _sparse[:, 7] = sparse[:, 7] - 78 + 74 # ju (round)

        return _sparse, numeric
