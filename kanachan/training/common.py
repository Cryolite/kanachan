import re
import pickle
import os
from typing import Any
import torch
from torch import Tensor, nn
from torch.distributed import (
    is_initialized,
    get_world_size,
    get_rank,
    ProcessGroup,
    send,
    recv,
)


def get_distributed_environment() -> tuple[int, int, int]:
    if not is_initialized():
        return 1, 0, 0

    if "LOCAL_RANK" not in os.environ:
        errmsg = "The `LOCAL_RANK` environment variable must be defined."
        raise RuntimeError(errmsg)

    world_size = get_world_size()
    rank = get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    return world_size, rank, local_rank


def make_noisy(state_dict: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in state_dict.items():
        if re.search("\\.weight$", key) is not None:
            new_key = re.sub("\\.weight$", ".weight_mu", key)
            result[new_key] = value
            continue
        if re.search("\\.bias$", key) is not None:
            new_key = re.sub("\\.bias$", ".bias_mu", key)
            result[new_key] = value
            continue
    return result


def get_gradient(model: nn.Module) -> Tensor:
    gradient = [
        param.grad.view(-1)
        for param in model.parameters()
        if param.grad is not None
    ]
    return torch.cat(gradient)


def is_gradient_nan(model: nn.Module) -> Tensor:
    gradient = get_gradient(model)
    return gradient.isnan().any()


def send_object(
    obj, *, device: torch.device, dst: int, group: ProcessGroup | None = None
) -> None:
    data = pickle.dumps(obj)
    buf = torch.frombuffer(bytearray(data), dtype=torch.int8).to(device=device)
    length = torch.tensor(buf.size(0), device=device, dtype=torch.int64)

    send(length, dst, group=group)
    send(buf, dst, group=group)


def recv_object(
    *,
    device: torch.device,
    src: int | None = None,
    group: ProcessGroup | None = None,
) -> tuple[Any, int]:
    length = torch.tensor(-1, device=device, dtype=torch.int64)
    src = recv(length, src, group=group)

    buf = torch.empty((int(length.item()),), device=device, dtype=torch.int8)
    recv(buf, src, group=group)
    obj = pickle.loads(buf.cpu().numpy().tobytes())

    return obj, src
