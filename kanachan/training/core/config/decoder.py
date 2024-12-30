import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass
class DecoderConfig:
    dimension: None | int
    activation_function: None | str
    dropout: None | float
    layer_normalization: bool
    num_layers: int
    load_from: Path | None


@dataclass
class SingleDecoderConfig(DecoderConfig):
    dimension: None | int = None
    activation_function: None | str = None
    dropout: None | float = None
    layer_normalization: bool = False
    num_layers: int = 1
    load_from: Path | None = None


@dataclass
class DoubleDecoderConfig(DecoderConfig):
    dimension: None | int = None
    activation_function: str = "relu"
    dropout: float = 0.1
    layer_normalization: bool = False
    num_layers: int = 2
    load_from: Path | None = None


@dataclass
class TripleDecoderConfig(DecoderConfig):
    dimension: None | int = None
    activation_function: str = "relu"
    dropout: float = 0.1
    layer_normalization: bool = False
    num_layers: int = 3
    load_from: Path | None = None


config_store = ConfigStore.instance()
config_store.store(name="single", node=SingleDecoderConfig, group="decoder")
config_store.store(name="double", node=DoubleDecoderConfig, group="decoder")
config_store.store(name="triple", node=TripleDecoderConfig, group="decoder")


def validate(config: Any) -> None:
    if config.decoder.dimension is None and config.decoder.num_layers >= 2:
        config.decoder.dimension = config.encoder.dimension
    if config.decoder.dimension is not None and config.decoder.dimension <= 0:
        errmsg = (
            f"{config.decoder.dimension}: "
            "`decoder.dimension` must be a positive integer."
        )
        raise RuntimeError(errmsg)

    if config.decoder.activation_function not in (None, "relu", "gelu"):
        errmsg = (
            f"{config.decoder.activation_function}:"
            " An invalid activation function for the decoder."
        )
        raise RuntimeError(errmsg)

    if config.decoder.dropout is not None and (
        config.decoder.dropout < 0.0 or 1.0 <= config.decoder.dropout
    ):
        errmsg = (
            f"{config.decoder.dropout}: `decoder.dropout` must be a real value"
            " within the range [0.0, 1.0)."
        )
        raise RuntimeError(errmsg)

    if config.decoder.num_layers <= 0:
        errmsg = (
            f"{config.decoder.num_layers}: "
            "`decoder.num_layers` must be a positive integer."
        )
        raise RuntimeError(errmsg)

    if config.decoder.num_layers == 1:
        if config.decoder.dimension is not None:
            errmsg = (
                "`decoder.dimension` cannot be specified for a "
                "single-layer decoder."
            )
            raise RuntimeError(errmsg)
        if config.decoder.activation_function is not None:
            errmsg = (
                "`decoder.activation_function` cannot be specified for "
                "a single-layer decoder."
            )
            raise RuntimeError(errmsg)
        if config.decoder.dropout is not None:
            errmsg = (
                "`decoder.dropout` cannot be specified for a "
                "single-layer decoder."
            )
            raise RuntimeError(errmsg)
        if config.decoder.layer_normalization:
            errmsg = (
                "`decoder.layer_normalization` cannot be specified for a "
                "single-layer decoder."
            )
            raise RuntimeError(errmsg)

    if config.decoder.load_from is not None:
        if not config.decoder.load_from.exists():
            errmsg = f"{config.decoder.load_from}: Does not exist."
            raise RuntimeError(errmsg)
        if not config.decoder.load_from.is_file():
            errmsg = f"{config.decoder.load_from}: Not a file."
            raise RuntimeError(errmsg)

    if hasattr(config.decoder, "num_qr_intervals"):
        if config.decoder.num_qr_intervals is not None and (
            config.decoder.num_qr_intervals <= 0
        ):
            errmsg = (
                f"{config.decoder.num_qr_intervals}: "
                "`decoder.num_qr_intervals` must be a positive integer."
            )
            raise RuntimeError(errmsg)


def dump(config: Any, prefix: str = "") -> None:
    if config.decoder.num_layers >= 2:
        logging.info(
            "%sDecoder dimension: %d", prefix, config.decoder.dimension
        )
        logging.info(
            "%sActivation function for decoder: %s",
            prefix,
            config.decoder.activation_function,
        )
        logging.info(
            "%sDropout for decoder: %f", prefix, config.decoder.dropout
        )
        logging.info(
            "%sLayer normalization for decoder: %s",
            prefix,
            config.decoder.layer_normalization,
        )
    logging.info(
        "%s# of decoder layers: %d", prefix, config.decoder.num_layers
    )
    if config.decoder.load_from is not None:
        logging.info(
            "%sLoad decoder from: %s", prefix, config.decoder.load_from
        )
    if hasattr(config.decoder, "num_qr_intervals"):
        logging.info(
            "%s# of quantile regression intervals: %d",
            prefix,
            config.decoder.num_qr_intervals,
        )
