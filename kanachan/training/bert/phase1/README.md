# `kanachan.training.bert.phase1` Python submodule

## Prerequisits

The following items are required to run the program in this directory:

- [Docker](https://www.docker.com/),
- NVIDIA driver, and
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html).
- ([CUDA](https://developer.nvidia.com/cuda-toolkit) is not required.)

For detailed installation instructions for the above prerequisite items, refer to those for each OS and distribution.

After the installation of the prerequisite items, [build the `cryolite/kanachan` Docker image](https://github.com/Cryolite/kanachan/blob/main/kanachan/README.md#cryolitekanachan-docker-image).

## `train.py`

#### Usage

```
$ docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /path/to/host-data:/workspace/data --rm cryolite/kanachan python3 -m kanachan.training.bert.phase1.train OPTIONS...
```

If you want to run this program on multiple GPUs, see [Running programs on multiple GPUs](https://github.com/Cryolite/kanachan/wiki/Running-programs-on-multiple-GPUs).

#### Options

`--training-data PATH`: Specify the path to the training data. The path must be one that can be interpreted within the Docker guest.

`--validation-data PAHT`: Specify the path to the validation data. The path must be one that can be interpreted within the Docker guest.

`--num-workers NWORKERS`: Specify the number of workers used in data loading. The argument must be a positive integer. Default to `2`.

`--device {cpu|cuda|cudaN}`: Specify the device on which the training is performed.

`--amp-optimization-level {O0|O1|O2|O3}`: **THIS OPTION IS DEPRECATED.** Specify the optimization level of [Automatic Mixed Precision](https://developer.nvidia.com/automatic-mixed-precision). Each optimization level corresponds to the one defined in [apex.amp](https://nvidia.github.io/apex/amp.html#opt-levels). Default to `O2`.

`--model-preset {base|large}`: Specify the model preset. `base` is the transformer encoder layers used in BERT BASE, and `large` is the one used in BERT LARGE. See the table below for the meaning of the presets:

|         | `DIM` | `NHEADS` | `DIM_FEEDFORWARD` | `NLAYERS` |
|---------|-------|----------|-------------------|-----------|
| `base`  | 768   | 12       | 3072              | 12        |
| `large` | 1024  | 16       | 4096              | 24        |

`--dimension DIM`: Specify the embedding dimension for the model. The argument must be a positive integer. Override the value by the preset.

`--num-heads NHEADS`: Specify the number of heads in each layer. The argument must be a positive integer. Override the value by the preset.

`--dim-feedforward DIM_FEEDFORWARD`: Specify the dimension of the feedforward network in each layer. The argument must be a positive integer. Override the value by the preset. Default to `4 * DIM`.

`--num-layers NLAYERS`: Specify the number of layers. The argument must be a positive integer. Override the value by the preset.

`--dim-final-feedforward DIM_FINAL_FEEDFORWARD`: Specify the dimension of the final feedforward network. The argument must be a positive integer. Default to `DIM_FEEDFORWARD`.

`--activation-function {relu|gelu}`: Specify the activation function for the feedforward networks. Defaults to `gelu`.

`--dropout DROPOUT`: Specify the dropout ratio. The argument must be a real value in the range \[0.0, 1.0\). Default to `0.1`.

`--initial-encoder PATH`: Specify the path to the initial encoder. The path must be one that can be interpreted within the Docker guest. Mutually exclusive to `--resume`.

`--initial-decoder PATH`: Specify the path to the initial decoder. The path must be one that can be interpreted within the Docker guest. Mutually exclusive to `--resume`.

`--training-batch-size N`: Specify the batch size for training. The argument must be a positive integer.

`--training-batch-size N`: Specify the batch size for validation. The argument must be a positive integer.

`--freeze-encoder`: Freeze encoder parameters during training.

`--optimizer {sgd|adam|radam|lamb}`: Specify the optimizer. Default to `lamb`.

`--momentum MOMENTUM`: Specify the momentum factor. Only meaningful for `sgd`. Default to `0.9`.

`--learning-rate LR`: Specify the learning rate. Default to `0.1` for `sgd`, `0.001` for `adam`, `radam`, and `lamb`.

`--epsilon EPS`: Specify the epsilon parameter. Only meaningful for `adam`, `radam`, and `lamb`. Default to `1.0e-8` for `adam` and `radam`, `1.0e-6` for `lamb`.

`--checkpointing`: Enable checkpointing.

`--gradient-accumulation-steps NSTEPS`: Specify the number of steps for gradient accumulation. Defaults to `1`.

`--max-gradient-norm NORM`: Specify the norm threshold for gradient clipping. Default to `1.0`.

`--num-epochs`: Specify the number of epochs to iterate. Default to `1`.

`--initial-optimizer PATH`: Specify the path to the initial optimizer state. The path must be one that can be interpreted within the Docker guest. Mutually exclusive to `--resume`.

`--output-prefix PATH`: Specify the output prefix. The path must be one that can be interpreted within the Docker guest.

`--experiment-name NAME`: Specify the experiment name. Default to the start time of the experiment in the format `YYYY-MM-DD-hh-mm-ss`. The final path to the output will becomes `PATH/NAME`.

`--num-epoch-digits`: Specify the number of digits to index epochs. Default to `2`.

`--snapshot-interval NSAMPLES`: Specify the interval between snapshots. The argument must be a positive integer. `0` means that no snapshot is taken at all. Default to `0`.

`--resume`: Resume the experiment from the latest snapshot in the path `PATH/NAME`.
