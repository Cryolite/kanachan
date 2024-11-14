from collections import OrderedDict
import torch
from torch import Tensor, nn
from kanachan.constants import (
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    NUM_NUMERIC_FEATURES,
    MAX_NUM_ACTION_CANDIDATES,
    ENCODER_WIDTH,
    EOR_NUM_SPARSE_FEATURES,
    EOR_NUM_NUMERIC_FEATURES,
    EOR_ENCODER_WIDTH,
)
from kanachan.nn.noisy_linear import NoisyLinear


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        input_dimension: int,
        dimension: int | None,
        activation_function: str | None,
        dropout: float | None,
        layer_normalization: bool,
        num_layers: int,
        output_mode: str,
        noise_init_std: float | None,
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
        if noise_init_std is not None and noise_init_std < 0.0:
            raise ValueError(noise_init_std)

        super().__init__()

        self.__output_mode = output_mode

        self.__output_width = 1
        if self.__output_mode == "ranking":
            self.__output_width = 4

        layers: OrderedDict[str, nn.Module] = OrderedDict()

        def create_linear_layer(
            in_features, out_features, device, dtype
        ) -> nn.Linear | NoisyLinear:
            if noise_init_std is None:
                return nn.Linear(
                    in_features, out_features, device=device, dtype=dtype
                )
            else:
                return NoisyLinear(
                    in_features,
                    out_features,
                    device=device,
                    dtype=dtype,
                    std_init=noise_init_std,
                )

        if num_layers == 1:
            layers["linear"] = create_linear_layer(
                input_dimension,
                self.__output_width,
                device=device,
                dtype=dtype,
            )
        else:
            assert dimension is not None
            layers["linear0"] = create_linear_layer(
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
            if layer_normalization:
                assert dimension is not None
                layers["layer_normalization0"] = nn.LayerNorm([dimension])

        for i in range(1, num_layers):
            assert dimension is not None
            assert dropout is not None
            final_layer = i == num_layers - 1
            layers[f"linear{i}"] = create_linear_layer(
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
                if layer_normalization:
                    layers[f"layer_normalization{i}"] = nn.LayerNorm(
                        [dimension]
                    )

        self.layers = nn.Sequential(layers)

    def forward(self, encode: Tensor) -> Tensor:
        assert encode.dim() == 3
        batch_size = encode.size(0)
        input_width = encode.size(1)

        original_dtype = encode.dtype

        first: int
        last: int
        decode: Tensor
        if self.__output_mode == "state":
            decode = self.layers(encode)
            decode = decode.squeeze(2)
            decode = decode.sum(1)
            assert decode.dim() == 1
            assert decode.size(0) == batch_size
        elif self.__output_mode == "scores":
            first = -1
            last = -1
            if input_width == ENCODER_WIDTH:
                first = MAX_NUM_ACTIVE_SPARSE_FEATURES + 2
                last = MAX_NUM_ACTIVE_SPARSE_FEATURES + NUM_NUMERIC_FEATURES
            elif input_width == EOR_ENCODER_WIDTH:
                first = EOR_NUM_SPARSE_FEATURES + 2
                last = EOR_NUM_SPARSE_FEATURES + EOR_NUM_NUMERIC_FEATURES
            else:
                raise NotImplementedError()
            assert last - first == 4
            scores_encode = encode[:, first:last]
            decode = self.layers(scores_encode)
            decode = decode.squeeze(2)
            assert decode.dim() == 2
            assert decode.size(0) == batch_size
            assert decode.size(1) == 4
        elif self.__output_mode == "ranking":
            first = -1
            last = -1
            if input_width == ENCODER_WIDTH:
                first = MAX_NUM_ACTIVE_SPARSE_FEATURES + 2
                last = MAX_NUM_ACTIVE_SPARSE_FEATURES + NUM_NUMERIC_FEATURES
            elif input_width == EOR_ENCODER_WIDTH:
                first = EOR_NUM_SPARSE_FEATURES + 2
                last = EOR_NUM_SPARSE_FEATURES + EOR_NUM_NUMERIC_FEATURES
            else:
                raise NotImplementedError()
            assert last - first == 4
            scores_encode = encode[:, first:last]
            decode = self.layers(scores_encode)
            assert decode.dim() == 3
            assert decode.size(0) == batch_size
            assert decode.size(1) == 4
            assert decode.size(2) == 4
        elif self.__output_mode == "candidates":
            if input_width != ENCODER_WIDTH:
                msg = "An invalid output mode."
                raise RuntimeError(msg)
            candidates_encode = encode[:, -MAX_NUM_ACTION_CANDIDATES:]
            decode = self.layers(candidates_encode)
            decode = decode.squeeze(2)
            assert decode.dim() == 2
            assert decode.size(0) == batch_size
            assert decode.size(1) == MAX_NUM_ACTION_CANDIDATES
        else:
            raise ValueError(self.__output_mode)

        return decode.to(dtype=original_dtype)

    def reset_parameters(self) -> None:
        for layer in self.layers:
            if isinstance(layer, (nn.Linear, NoisyLinear)):
                layer.reset_parameters()
