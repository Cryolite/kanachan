import math
import torch
from torch import Tensor, nn
from kanachan.constants import NUM_TYPES_OF_ACTIONS, MAX_NUM_ACTION_CANDIDATES


class DecodeConverter(nn.Module):
    def __init__(self, mode: str) -> None:
        super().__init__()

        if mode not in ("scores", "probs", "log_probs", "argmax"):
            msg = f"{mode}: An invalid mode."
            raise RuntimeError(msg)
        self.__mode = mode

    def forward(self, candidates: Tensor, decode: Tensor) -> Tensor:
        assert candidates.dim() == 2
        batch_size = candidates.size(0)
        assert candidates.size(1) == MAX_NUM_ACTION_CANDIDATES
        assert decode.dim() == 2
        assert decode.size(0) == batch_size
        assert decode.size(1) == MAX_NUM_ACTION_CANDIDATES

        if self.__mode in ("probabilities", "log_probabilities"):
            mask = candidates >= NUM_TYPES_OF_ACTIONS
            decode = decode.masked_fill(mask, -math.inf)
            if self.__mode == "probabilities":
                result = torch.softmax(decode, 1)
                result = result.masked_fill(mask, 0.0)
            else:
                assert self.__mode == "log_probabilities"
                result = torch.log_softmax(decode, 1)
                result = result.masked_fill(mask, -math.inf)
            return result

        assert self.__mode in ("scores", "argmax")
        mask = candidates >= NUM_TYPES_OF_ACTIONS
        score = decode.masked_fill(mask, -math.inf)
        if self.__mode == "scores":
            return score
        return score.argmax(1)
