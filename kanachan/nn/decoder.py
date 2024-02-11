from collections import OrderedDict
import torch
from torch import Tensor, nn
from torchrl.modules import NoisyLinear
from kanachan.constants import (
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_NUMERIC_FEATURES,
    MAX_NUM_ACTION_CANDIDATES,
    ENCODER_WIDTH,
    ROUND_NUM_SPARSE_FEATURES,
    ROUND_NUM_NUMERIC_FEATURES,
    ROUND_ENCODER_WIDTH,
)


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        input_dimension: int,
        dimension: int | None,
        activation_function: str | None,
        dropout: float | None,
        num_layers: int,
        output_mode: str,
        noise_init_std: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if input_dimension <= 0:
            raise ValueError(input_dimension)
        if dimension is not None and dimension <= 0:
            raise ValueError(dimension)
        if activation_function not in (None, "relu", "gelu"):
            raise ValueError(activation_function)
        if dropout is not None and (dropout < 0.0 or 1.0 <= dropout):
            raise ValueError(dropout)
        if num_layers <= 0:
            raise ValueError(num_layers)
        if num_layers == 1:
            if dimension is not None:
                raise ValueError(dimension)
            if activation_function is not None:
                raise ValueError(activation_function)
            if dropout is not None:
                raise ValueError(dropout)
        if output_mode not in ("state", "scores", "candidates", "ranking"):
            raise ValueError(output_mode)
        if noise_init_std < 0.0:
            raise ValueError(noise_init_std)

        super().__init__()

        self.__output_mode = output_mode

        self.__output_width = 1
        if self.__output_mode == "ranking":
            self.__output_width = 4

        layers = OrderedDict()

        if noise_init_std == 0.0:
            linear_layer = nn.Linear
        else:
            linear_layer = NoisyLinear

        if num_layers == 1:
            layers["linear"] = linear_layer(
                input_dimension,
                self.__output_width,
                device=device,
                dtype=dtype,
            )
        else:
            assert dimension is not None
            layers["linear0"] = linear_layer(
                input_dimension, dimension, device=device, dtype=dtype
            )
        if num_layers >= 2:
            if activation_function == "relu":
                layers["activation0"] = nn.ReLU()
            elif activation_function == "gelu":
                layers["activation0"] = nn.GELU()
            else:
                raise AssertionError(activation_function)
            assert dropout is not None
            layers["dropout0"] = nn.Dropout(p=dropout)

        for i in range(1, num_layers):
            assert dimension is not None
            assert dropout is not None
            final_layer = i == num_layers - 1
            layers[f"linear{i}"] = linear_layer(
                dimension,
                self.__output_width if final_layer else dimension,
                device=device,
                dtype=dtype,
            )
            if not final_layer:
                if activation_function == "relu":
                    layers[f"activation{i}"] = nn.ReLU()
                elif activation_function == "gelu":
                    layers[f"activation{i}"] = nn.GELU()
                else:
                    raise AssertionError(activation_function)
                layers[f"dropout{i}"] = nn.Dropout(p=dropout)

        self.layers = nn.Sequential(layers)

    def forward(self, encode: Tensor) -> Tensor:
        assert encode.dim() == 3
        batch_size = encode.size(0)
        input_width = encode.size(1)

        original_dtype = encode.dtype

        if self.__output_mode == "state":
            decode: Tensor = self.layers(encode)
            decode = decode.squeeze(2)
            decode = decode.sum(1)
            assert decode.dim() == 1
            assert decode.size(0) == batch_size
        elif self.__output_mode == "scores":
            first: int = -1
            last: int = -1
            if input_width == ENCODER_WIDTH:
                first = MAX_NUM_ACTIVE_SPARSE_FEATURES + 2
                last = MAX_NUM_ACTIVE_SPARSE_FEATURES + NUM_NUMERIC_FEATURES
            elif input_width == ROUND_ENCODER_WIDTH:
                first = ROUND_NUM_SPARSE_FEATURES + 2
                last = ROUND_NUM_SPARSE_FEATURES + ROUND_NUM_NUMERIC_FEATURES
            else:
                raise NotImplementedError()
            assert last - first == 4
            scores_encode = encode[:, first:last]
            decode: Tensor = self.layers(scores_encode)
            decode = decode.squeeze(2)
            assert decode.dim() == 2
            assert decode.size(0) == batch_size
            assert decode.size(1) == 4
        elif self.__output_mode == "ranking":
            first: int = -1
            last: int = -1
            if input_width == ENCODER_WIDTH:
                first = MAX_NUM_ACTIVE_SPARSE_FEATURES + 2
                last = MAX_NUM_ACTIVE_SPARSE_FEATURES + NUM_NUMERIC_FEATURES
            elif input_width == ROUND_ENCODER_WIDTH:
                first = ROUND_NUM_SPARSE_FEATURES + 2
                last = ROUND_NUM_SPARSE_FEATURES + ROUND_NUM_NUMERIC_FEATURES
            else:
                raise NotImplementedError()
            assert last - first == 4
            scores_encode = encode[:, first:last]
            decode: Tensor = self.layers(scores_encode)
            assert decode.dim() == 3
            assert decode.size(0) == batch_size
            assert decode.size(1) == 4
            assert decode.size(2) == 4
        elif self.__output_mode == "candidates":
            if input_width != ENCODER_WIDTH:
                msg = "An invalid output mode."
                raise RuntimeError(msg)
            candidates_encode = encode[:, -MAX_NUM_ACTION_CANDIDATES:]
            decode: Tensor = self.layers(candidates_encode)
            decode = decode.squeeze(2)
            assert decode.dim() == 2
            assert decode.size(0) == batch_size
            assert decode.size(1) == MAX_NUM_ACTION_CANDIDATES
        else:
            raise ValueError(self.__output_mode)

        return decode.to(dtype=original_dtype)
