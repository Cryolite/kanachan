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


def validate(decoder_config: Any, encoder_dimension: int) -> None:
    if decoder_config.dimension is None and decoder_config.num_layers >= 2:
        decoder_config.dimension = encoder_dimension
    if decoder_config.dimension is not None and decoder_config.dimension <= 0:
        errmsg = (
            f"{decoder_config.dimension}: "
            "`dimension` must be a positive integer."
        )
        raise RuntimeError(errmsg)

    if decoder_config.activation_function not in (None, "relu", "gelu"):
        errmsg = (
            f"{decoder_config.activation_function}: "
            "An invalid activation function for the decoder."
        )
        raise RuntimeError(errmsg)

    if decoder_config.dropout is not None and (
        decoder_config.dropout < 0.0 or 1.0 <= decoder_config.dropout
    ):
        errmsg = (
            f"{decoder_config.dropout}: "
            "`dropout` must be a real number within the range [0.0, 1.0)."
        )
        raise RuntimeError(errmsg)

    if decoder_config.num_layers <= 0:
        errmsg = (
            f"{decoder_config.num_layers}: "
            "`num_layers` must be a positive integer."
        )
        raise RuntimeError(errmsg)

    if decoder_config.num_layers == 1:
        if decoder_config.dimension is not None:
            errmsg = (
                "`dimension` cannot be specified for a single-layer decoder."
            )
            raise RuntimeError(errmsg)
        if decoder_config.activation_function is not None:
            errmsg = (
                "`activation_function` cannot be specified for a single-layer "
                "decoder."
            )
            raise RuntimeError(errmsg)
        if decoder_config.dropout is not None:
            errmsg = (
                "`dropout` cannot be specified for a single-layer decoder."
            )
            raise RuntimeError(errmsg)
        if decoder_config.layer_normalization:
            errmsg = (
                "`layer_normalization` cannot be specified for a single-layer "
                "decoder."
            )
            raise RuntimeError(errmsg)

    if decoder_config.load_from is not None:
        if not decoder_config.load_from.exists():
            errmsg = f"{decoder_config.load_from}: Does not exist."
            raise RuntimeError(errmsg)
        if not decoder_config.load_from.is_file():
            errmsg = f"{decoder_config.load_from}: Not a file."
            raise RuntimeError(errmsg)

    if hasattr(decoder_config, "num_qr_intervals"):
        if decoder_config.num_qr_intervals < 0:
            errmsg = (
                f"{decoder_config.num_qr_intervals}: "
                "`num_qr_intervals` must be a non-negative integer."
            )
            raise RuntimeError(errmsg)


def dump(decoder_config: Any, prefix: str = "") -> None:
    if decoder_config.num_layers >= 2:
        logging.info(
            "%sDecoder dimension: %d", prefix, decoder_config.dimension
        )
        logging.info(
            "%sActivation function for decoder: %s",
            prefix,
            decoder_config.activation_function,
        )
        logging.info(
            "%sDropout for decoder: %f", prefix, decoder_config.dropout
        )
        logging.info(
            "%sLayer normalization for decoder: %s",
            prefix,
            decoder_config.layer_normalization,
        )
    logging.info(
        "%s# of decoder layers: %d", prefix, decoder_config.num_layers
    )
    if decoder_config.load_from is not None:
        logging.info(
            "%sLoad decoder from: %s", prefix, decoder_config.load_from
        )
    if hasattr(decoder_config, "num_qr_intervals"):
        if decoder_config.num_qr_intervals == 0:
            logging.info(
                "%sQuantile regression: (N/A)", prefix
            )
        else:
            logging.info(
                "%s# of quantile regression intervals: %d",
                prefix,
                decoder_config.num_qr_intervals,
            )
    if hasattr(decoder_config, "dueling_network"):
        logging.info(
            "%sDueling network: %s",
            prefix,
            decoder_config.dueling_network,
        )
