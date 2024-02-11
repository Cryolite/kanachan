import logging
from typing import Any
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class SingleDecoderConfig:
    dimension: None | int = None
    activation_function: None | str = None
    dropout: None | float = None
    num_layers: int = 1


@dataclass
class DoubleDecoderConfig:
    dimension: None | int = None
    activation_function: str = "relu"
    dropout: float = 0.1
    num_layers: int = 2


@dataclass
class TripleDecoderConfig:
    dimension: None | int = None
    activation_function: str = "relu"
    dropout: float = 0.1
    num_layers: int = 3


config_store = ConfigStore.instance()
config_store.store(name="single", node=SingleDecoderConfig, group="decoder")
config_store.store(name="double", node=DoubleDecoderConfig, group="decoder")
config_store.store(name="triple", node=TripleDecoderConfig, group="decoder")


def validate(config: Any) -> None:
    if config.decoder.dimension is None and config.decoder.num_layers >= 2:
        config.decoder.dimension = config.encoder.dimension
    if config.decoder.dimension is not None and config.decoder.dimension <= 0:
        errmsg = (
            f"{config.decoder.dimension}: `decoder.dimension` must be a"
            " positive integer."
        )
        raise RuntimeError(errmsg)

    if config.decoder.activation_function not in (None, "relu", "gelu"):
        errmsg = (
            f"{config.decoder.activation_function}: An invalid"
            " activation function for the decoder."
        )
        raise RuntimeError(errmsg)

    if config.decoder.dropout is not None and (
        config.decoder.dropout < 0.0 or 1.0 <= config.decoder.dropout
    ):
        errmsg = (
            f"{config.decoder.dropout}: `decoder.dropout` must be a"
            " real value within the range [0.0, 1.0)."
        )
        raise RuntimeError(errmsg)

    if config.decoder.num_layers <= 0:
        errmsg = (
            f"{config.decoder.num_layers}: `decoder.num_layers` must be"
            " a positive integer."
        )
        raise RuntimeError(errmsg)

    if config.decoder.num_layers == 1:
        if config.decoder.dimension is not None:
            errmsg = (
                "`decoder.dimension` cannot be specified for a"
                " single-layer decoder."
            )
            raise RuntimeError(errmsg)
        if config.decoder.activation_function is not None:
            errmsg = (
                "`decoder.activation_function` cannot be specified for"
                " a single-layer decoder."
            )
            raise RuntimeError(errmsg)
        if config.decoder.dropout is not None:
            errmsg = (
                "`decoder.dropout` cannot be specified for a"
                " single-layer decoder."
            )
            raise RuntimeError(errmsg)


def dump(config: Any) -> None:
    if config.decoder.num_layers >= 2:
        logging.info("Decoder dimension: %d", config.decoder.dimension)
        logging.info(
            "Activation function for decoder: %s",
            config.decoder.activation_function,
        )
        logging.info("Dropout for decoder: %f", config.decoder.dropout)
    logging.info("# of decoder layers: %d", config.decoder.num_layers)
