import logging
import os
from dataclasses import dataclass
from omegaconf import DictConfig
from hydra.core.config_store import ConfigStore
import torch
from torch import backends


@dataclass
class CpuConfig:
    type: str = "cpu"
    dtype: str = "float32"
    amp_dtype: str | None = None


@dataclass
class CudaConfig:
    type: str = "cuda"
    dtype: str = "float32"
    amp_dtype: str = "float16"


config_store = ConfigStore.instance()
config_store.store(name="cpu", node=CpuConfig, group="device")
config_store.store(name="cuda", node=CudaConfig, group="device")


def validate(
    config: DictConfig,
) -> tuple[int, int, int, torch.device, torch.dtype, torch.dtype]:
    if "WORLD_SIZE" not in os.environ:
        errmsg = "Set the `WORLD_SIZE` environment variable."
        raise RuntimeError(errmsg)
    world_size = int(os.environ["WORLD_SIZE"])

    if "RANK" not in os.environ:
        errmsg = "Set the `RANK` environment variable."
        raise RuntimeError(errmsg)
    rank = int(os.environ["RANK"])

    if "LOCAL_RANK" not in os.environ:
        errmsg = "Set the `LOCAL_RANK` environment variable."
        raise RuntimeError(errmsg)
    local_rank = int(os.environ["LOCAL_RANK"])

    if config.device.type not in ("cpu", "cuda"):
        errmsg = f"{config.device.type}: An invalid value for `device.type`."
        raise RuntimeError(errmsg)
    if config.device.type == "cuda":
        if local_rank < torch.cuda.device_count():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    elif config.device.type == "cpu":
        device = torch.device("cpu")
    else:
        errmsg = f"{config.device.type}: An invalid value for `device.type`."
        raise RuntimeError(errmsg)

    if config.device.dtype not in (
        "float64",
        "double",
        "float32",
        "float",
        "float16",
        "half",
    ):
        errmsg = (
            f"{config.device.dtype}: An invalid value for" " `device.dtype`."
        )
        raise RuntimeError(errmsg)
    dtype = {
        "float64": torch.float64,
        "double": torch.float64,
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
    }[config.device.dtype]

    if device.type == "cpu":
        amp_dtype = config.device.dtype
    if config.device.amp_dtype is None:
        amp_dtype = dtype
    else:
        if config.device.amp_dtype not in (
            "float64",
            "double",
            "float32",
            "float",
            "float16",
            "half",
        ):
            errmsg = (
                f"{config.device.amp_dtype}: An invalid value for"
                " `device.amp_dtype`."
            )
            raise RuntimeError(errmsg)
        amp_dtype = {
            "float64": torch.float64,
            "double": torch.float64,
            "float32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "half": torch.float16,
        }[config.device.amp_dtype]
    if amp_dtype == torch.float64 and dtype in (torch.float32, torch.float16):
        errmsg = (
            "An invalid combination of `device.dtype`"
            f" (`{config.device.dtype}`) and `device.amp_dtype`"
            f" (`{config.device.amp_dtype}`)."
        )
        raise RuntimeError(errmsg)
    if amp_dtype == torch.float32 and dtype == torch.float16:
        errmsg = (
            "An invalid combination of `device.dtype`"
            f" (`{config.device.dtype}`) and `device.amp_dtype`"
            f" (`{config.device.amp_dtype}`)."
        )
        raise RuntimeError(errmsg)

    if backends.cudnn.is_available():
        backends.cudnn.benchmark = True

    return world_size, rank, local_rank, device, dtype, amp_dtype


def dump(
    *,
    world_size: int,
    rank: int,
    local_rank: int,
    device: torch.device,
    dtype: torch.dtype,
    amp_dtype: torch.dtype,
) -> None:
    logging.info("World size: %d", world_size)
    logging.info("Process rank: %d", rank)
    logging.info("Local process rank: %d", local_rank)

    logging.info("Device: %s", device)
    logging.info("dtype: %s", dtype)
    if dtype == amp_dtype:
        logging.info("AMP dtype: (AMP is disabled)")
    else:
        logging.info("AMP dtype: %s", amp_dtype)

    if backends.cudnn.is_available():
        logging.info("cuDNN: available")
    else:
        logging.info("cuDNN: N/A")
