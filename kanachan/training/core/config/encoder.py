from pathlib import Path
import logging
from typing import Any
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class EncoderConfig:
    position_encoder: str
    dimension: int
    num_heads: int
    dim_feedforward: None | int = None
    activation_function: str = "gelu"
    dropout: float = 0.1
    num_layers: int = 4
    layer_normalization: bool = False
    load_from: None | Path = None


@dataclass
class BertTinyEncoderConfig(EncoderConfig):
    position_encoder: str = "position_embedding"
    dimension: int = 128
    num_heads: int = 2
    dim_feedforward: None | int = None
    activation_function: str = "gelu"
    dropout: float = 0.1
    num_layers: int = 2
    layer_normalization: bool = False
    load_from: None | Path = None


@dataclass
class BertMiniEncoderConfig(EncoderConfig):
    position_encoder: str = "position_embedding"
    dimension: int = 256
    num_heads: int = 4
    dim_feedforward: None | int = None
    activation_function: str = "gelu"
    dropout: float = 0.1
    num_layers: int = 4
    layer_normalization: bool = False
    load_from: None | Path = None


@dataclass
class BertSmallEncoderConfig(EncoderConfig):
    position_encoder: str = "position_embedding"
    dimension: int = 512
    num_heads: int = 8
    dim_feedforward: None | int = None
    activation_function: str = "gelu"
    dropout: float = 0.1
    num_layers: int = 4
    layer_normalization: bool = False
    load_from: None | Path = None


@dataclass
class BertMediumEncoderConfig(EncoderConfig):
    position_encoder: str = "position_embedding"
    dimension: int = 512
    num_heads: int = 8
    dim_feedforward: None | int = None
    activation_function: str = "gelu"
    dropout: float = 0.1
    num_layers: int = 8
    layer_normalization: bool = False
    load_from: None | Path = None


@dataclass
class BertBaseEncoderConfig(EncoderConfig):
    position_encoder: str = "position_embedding"
    dimension: int = 768
    num_heads: int = 12
    dim_feedforward: None | int = None
    activation_function: str = "gelu"
    dropout: float = 0.1
    num_layers: int = 12
    layer_normalization: bool = False
    load_from: None | Path = None


@dataclass
class BertLargeEncoderConfig(EncoderConfig):
    position_encoder: str = "position_embedding"
    dimension: int = 1024
    num_heads: int = 16
    dim_feedforward: None | int = None
    activation_function: str = "gelu"
    dropout: float = 0.1
    num_layers: int = 24
    layer_normalization: bool = False
    load_from: None | Path = None


config_store = ConfigStore.instance()
config_store.store(
    name="bert_tiny", node=BertTinyEncoderConfig, group="encoder"
)
config_store.store(
    name="bert_mini", node=BertMiniEncoderConfig, group="encoder"
)
config_store.store(
    name="bert_small", node=BertSmallEncoderConfig, group="encoder"
)
config_store.store(
    name="bert_medium", node=BertMediumEncoderConfig, group="encoder"
)
config_store.store(
    name="bert_base", node=BertBaseEncoderConfig, group="encoder"
)
config_store.store(
    name="bert_large", node=BertLargeEncoderConfig, group="encoder"
)


def validate(encoder_config: Any) -> None:
    if encoder_config.position_encoder not in (
        "positional_encoding",
        "position_embedding",
    ):
        errmsg = (
            f"{encoder_config.position_encoder}: "
            "An invalid position encoder."
        )
        raise RuntimeError(errmsg)

    if encoder_config.dimension <= 0:
        errmsg = (
            f"{encoder_config.dimension}: "
            "`encoder.dimension` must be a positive integer."
        )
        raise RuntimeError(errmsg)

    if encoder_config.num_heads <= 0:
        errmsg = (
            f"{encoder_config.num_heads}: "
            "`encoder.num_heads` must be a positive integer."
        )
        raise RuntimeError(errmsg)

    if encoder_config.dim_feedforward is None:
        encoder_config.dim_feedforward = 4 * encoder_config.dimension
    if encoder_config.dim_feedforward <= 0:
        errmsg = (
            f"{encoder_config.dim_feedforward}: "
            "`encoder.dim_feedforward` must be a positive integer."
        )
        raise RuntimeError(errmsg)

    if encoder_config.activation_function not in ("relu", "gelu"):
        errmsg = (
            f"{encoder_config.activation_function}: "
            "An invalid activation function for the encoder."
        )
        raise RuntimeError(errmsg)

    if encoder_config.dropout < 0.0 or 1.0 <= encoder_config.dropout:
        errmsg = (
            f"{encoder_config.dropout}: `encoder.dropout` must be a real "
            "number within the range [0.0, 1.0)."
        )
        raise RuntimeError(errmsg)

    if encoder_config.num_layers <= 0:
        errmsg = (
            f"{encoder_config.num_layers}: "
            "`encoder.num_layers` must be a positive integer."
        )
        raise RuntimeError(errmsg)

    if encoder_config.load_from is not None:
        if not encoder_config.load_from.exists():
            errmsg = f"{encoder_config.load_from}: Does not exist."
            raise RuntimeError(errmsg)
        if not encoder_config.load_from.is_file():
            errmsg = f"{encoder_config.load_from}: Not a file."
            raise RuntimeError(errmsg)


def dump(encoder_config: Any, prefix: str = "") -> None:
    logging.info(
        "%sPosition encoder: %s", prefix, encoder_config.position_encoder
    )
    logging.info("%sEncoder dimension: %d", prefix, encoder_config.dimension)
    logging.info(
        "%s# of heads for encoder: %d", prefix, encoder_config.num_heads
    )
    logging.info(
        "%sDimension of feedforward networks for encoder: %d",
        prefix,
        encoder_config.dim_feedforward,
    )
    logging.info(
        "%sActivation function for encoder: %s",
        prefix,
        encoder_config.activation_function,
    )
    logging.info("%sDropout for encoder: %f", prefix, encoder_config.dropout)
    logging.info(
        "%sLayer normalization for encoder: %s",
        prefix,
        encoder_config.layer_normalization,
    )
    logging.info(
        "%s# of encoder layers: %d", prefix, encoder_config.num_layers
    )
    if encoder_config.load_from is not None:
        logging.info(
            "%sLoad encoder from: %s", prefix, encoder_config.load_from
        )
