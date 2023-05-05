# `kanachan.training.ilql` Python submodule

A training program with [ILQL](https://arxiv.org/abs/2206.11871).

## Prerequisits

The following items are required to run the programs in this directory:

- [Docker](https://www.docker.com/),
- NVIDIA driver, and
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html).
- ([CUDA](https://developer.nvidia.com/cuda-toolkit) is not required.)

For detailed installation instructions for the above prerequisite items, refer to those for each OS and distribution.

After the installation of the prerequisite items, first [build the `cryolite/kanachan` Docker image](https://github.com/Cryolite/kanachan/blob/main/kanachan/README.md#cryolitekanachan-docker-image). Then, execute the following command with the top directory of the working tree of this repository as the current directory:

```sh
kanachan$ docker build -f kanachan/training/ilql/Dockerfile -t cryolite/kanachan.training.ilql .
```

## `train.py`

#### Usage

```
$ docker run --gpus all -v /path/to/host-data:/workspace/data --rm cryolite/kanachan.training.ilql OPTIONS...
```

If you want to run this program on specific GPUs, modify the `--gpus` option for the `docker run` command (see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#gpu-enumeration) or `device.type` option (see below).

#### Options

Options are specified in the [Hydra](https://hydra.cc/) manner.

`training_data=PATH`: Specify the path to the training data. The path must be one that can be interpreted within the Docker guest.

`num_workers=NWORKERS`: Specify the number of workers used in data loading. The argument must be a non-negative integer. `0` means that the main process is used to load data. Default to `0` for CPU, and `2` for CUDA.

`device={cpu|cuda}`: Specify the device on which the training is performed. Default to the value guessed from PyTorch build information. See the table below for the detailed meaning of the options:

| `device` | `device.type` | `device.dtype` | `device.amp_dtype` |
|----------|---------------|----------------|--------------------|
| `cpu`    | `cpu`         | `float64`      | (N/A)              |
| `cuda`   | `cuda`        | `float32`      | `float16`          |

`device.type={cpu|cuda|cudaN}`: Specify the device on which the training is performed. Override the value by the `device` option.

`device.dtype={float64|double|float32|float|float16|half}`: Specify the PyTorch [`dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch-dtype). Override the value by the `device` option.

`device.amp_dtype={float64|double|float32|float|float16|half}`: Specify the PyTorch `dtype` for [automatic mixed precision (AMP)](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html). Override the value by the `device` option.

`encoder={bert_base|bert_large}`: Specify the encoder structure. Default to `bert_base`. See the table below for the detailed meaning of the options:

| `encoder`    | `encoder.position_encoder` | `encoder.dimension` | `encoder.num_heads` | `encoder.dim_feedforward` | `encoder.activation_function` | `encoder.dropout` | `encoder.num_layers` | `encoder.load_from` |
|--------------|----------------------------|---------------------|---------------------|---------------------------|-------------------------------|-------------------|----------------------|---------------------|
| `bert_base`  | `positional_encoding`      | `768`               | `12`                | `3072`                    | `gelu`                        | `0.1`             | `12`                 | (N/A)               |
| `bert_large` | `positional_encoding`      | `1024`              | `16`                | `4096`                    | `gelu`                        | `0.1`             | `24`                 | (N/A)               |

`encoder.position_encoder={positional_encoding|position_embedding}`: Specify the method of encoding positions. `positional_encoding` is the method used in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) to encode positions with a sinusoidal function. `position_embedding` is the method used in the paper ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805) to encode positions with embeddings. Override the value specified by the `encoder` option. Default to `positional_encoding`.

`encoder.dimension=DIM`: Specify the embedding dimension for the encoder. The argument must be a positive integer. Override the value by the `encoder` option.

`encoder.num_heads=NHEADS`: Specify the number of heads in each encoder layer. The argument must be a positive integer. Override the value by the `encoder` option.

`encoder.dim_feedforward=DIM_FEEDFORWARD`: Specify the dimension of the feedforward networks in each encoder layer. The argument must be a positive integer. Override the value by the `encoder` option. Default to `4 * DIM`.

`encoder.activation_function={relu|gelu}`: Specify the activation function for the feedforward networks in each encoder layer. Override the value specified by the `encoder` option. Default to `gelu`.

`encoder.dropout=DROPOUT`: Specify the dropout ratio for the feedforward networks in each encoder layer. The argument must be a real number in the range \[0.0, 1.0\). Override the value by the `encoder` option. Default to `0.1`.

`encoder.num_layers=NLAYERS`: Specify the number of encoder layers. The argument must be a positive integer. Override the value by the `encoder` option.

`encoder.load_from=INITIAL_ENCODER`: Specify the path to the initial encoder snapshot. The path must be one that can be interpreted within the Docker guest. Mutually exclusive to the `initial_model` and `initial_model_prefix` options.

`decoder={single|double|triple}`: Specify the decoder structure. Default to `double`. See the table below for the detailed meaning of the options:

| `decoder` | `decoder.dim_feedforward`   | `decoder.activation_function` | `decoder.dropout` | `decoder.num_layers` | `decoder.load_from` |
|-----------|-----------------------------|-------------------------------|-------------------|----------------------|---------------------|
| `single`  | (N/A)                       | `gelu`                        | `0.1`             | `1`                  | (N/A)               |
| `double`  | (`encoder.dim_feedforward`) | `gelu`                        | `0.1`             | `2`                  | (N/A)               |
| `triple`  | (`encoder.dim_feedforward`) | `gelu`                        | `0.1`             | `3`                  | (N/A)               |

`decoder.dim_feedforward=DIM_FEEDFORWARD`: Specify the dimension of the feedforward networks in each decoder layer. The argument must be a positive integer. Override the value by the `decoder` option. Default to the value specified by the `encoder.dim_feedforward` option.

`decoder.activation_function={relu|gelu}`: Specify the activation function for the feedforward networks in each decoder layer. Override the value by the `decoder` option. Default to `gelu`.

`decoder.dropout=DROPOUT`: Specify the dropout ratio for the feedforward networks in each decoder layer. The argument must be a real number in the range \[0.0, 1.0\). Override the value by the `decoder` option. Default to `0.1`.

`decoder.num_layers=NLAYERS`: Specify the number of decoder layers. The argument must be a positive integer. Override the value by the `decoder` option.

`decoder.load_from=INITIAL_DECODER`: Specify the path to the initial decoder snapshot. The path must be one that can be interpreted within the Docker guest. Mutually exclusive to the `initial_model` and `initial_model_prefix` options.

`initial_model=INITIAL_MODEL`: Specify the path to the initial model snapshot. The path must be one that can be interpreted within the Docker guest. Mutually exclusive to the `encoder.load_from`, `decoder.load_from`, and `initial_model_prefix` options.

`initial_model_prefix=PATH`: Specify the prefix to the initial model snapshot. The path must be one that can be interpreted within the Docker guest. Mutually exclusive to the `encoder.load_from`, `decoder.load_from`, and `initial_model` options.

`initial_model_index=N`: Speficy the index of the initial model snapshot. The argument must be a non-negative integer. Must be used with `initial_model_prefix` option.

`reward_plugin=REWARD_PLUGIN`: Specify the path to the reward plugin. The path must be one that can be interpreted within the Docker guest.

`discount_factor=GAMMA`: Specify the discount factor (in the sense of reinforcement learning). The argument must be a real number in the range \[0.0, 1.0\]. Default to `1.0`.

`expectile=TAU`: Specify the expectile hyperparameter for ILQL. The argument must be a real number in the range \[0.0, 1.0\].

`v_loss_scaling=SCALE`: Specify the coefficient for the `V` loss term. The argument must be a non-negative real number. Default to `1.0`.

`checkpointing={false|true}`: Enable [checkpointing](https://pytorch.org/docs/stable/checkpoint.html). Default to `false`.

`batch_size=N`: Specify the batch size. The argument must be a positive integer.

`gradient_accumulation_steps=NSTEPS`: Specify the number of steps for gradient accumulation. The argument must be a positive integer. Default to `1`.

`max_gradient_norm=NORM`: Specify the norm threshold for gradient clipping. The argument must be a positive real number. Default to `1.0`.

`optimizer={sgd|adam|radam|mtadam|lamb}`: Specify the optimizer preset. Default to `lamb`. See the table below for the detailed meaning of the options:

| `optimizer` | `optimizer.type`                             | `optimizer.momentum` | `optimizer.epsilon` | `optimizer.learning_rate` | `optimizer.initialize` |
|-------------|----------------------------------------------|----------------------|---------------------|---------------------------|------------------------|
| `sgd`       | `sgd`                                        | `0.0`                | (N/A)               | (EXPLICITLY REQUIRED)     | `false`                |
| `adam`      | [`adam`](https://arxiv.org/abs/1412.6980)    | (N/A)                | `1.0e-8`            | `0.001`                   | `false`                |
| `radam`     | [`radam`](https://arxiv.org/abs/1908.03265)  | (N/A)                | `1.0e-8`            | `0.001`                   | `false`                |
| `mtadam`    | [`mtadam`](https://arxiv.org/abs/2006.14683) | (N/A)                | `1.0e-8`            | `0.001`                   | `true`                 |
| `lamb`      | [`lamb`](https://arxiv.org/abs/1904.00962)   | (N/A)                | `1.0e-6`            | `0.001`                   | `false`                |

`optimizer.type={sgd|adam|radam|mtadam|lamb}` Specify the optimizer type. Override the value specified by the `optimizer` option.

`optimizer.momentum=MOMENTUM`: Specify the momentum factor. Only meaningful for `optimizer.type=sgd`. The argument must be a real number in the range \[0.0, 1.0\). Override the value specified by the `optimizer` option. Default to `0.0`.

`optimizer.epsilon=EPS`: Specify the epsilon parameter. Only meaningful for `adam`, `radam`, `mtadam`, and `lamb`. The argument must be a positive real number. Override the value specified by the `optimizer` option. Default to `1.0e-8` for `adam`, `radam`, and `mtadam`, and `1.0e-6` for `lamb`.

`optimizer.learning_rate=LR`: Specify the learning rate. The argument must be a positive real number. Override the value specified by the `optimizer` option. Default to `0.001` for `adam`, `radam`, `mtadam`, and `lamb`.

`optimizer.initialize={false|true}`: Specify whether to start from the initialized optimizer without using a snapshot, even if one is found. Override the value specified by the `optimizer` option.

`target_update_interval=N`: Specify the interval of updating the target networks. The argument must be a positive integer.

`target_update_rate=ALPHA`: Specify the rate of the Polyak averaging when updating the target networks. The argument must be a real number in the range \[0.0, 1.0\].

`snapshot_interval=NSAMPLES`: Specify the interval between snapshots. The argument must be a non-negative integer. `0` means that no snapshot is taken at all. Default to `0`.

## `extract_policy.py`

#### Usage

```
$ docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /path/to/host-data:/workspace/data --rm cryolite/kanachan python3 -m kanachan.training.iql.extract_policy OPTIONS...
```

If you want to run this program on multiple GPUs, see [Running programs on multiple GPUs](https://github.com/Cryolite/kanachan/wiki/Running-programs-on-multiple-GPUs).

#### Options

`--training-data PATH`: Specify the path to the training data. The path must be one that can be interpreted within the Docker guest.

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

`--dropout DROPOUT`: Specify the dropout ratio. The argument must be a real number in the range \[0.0, 1.0\). Default to `0.1`.

`--initial-encoder PATH`: Specify the path to the initial encoder. The path must be one that can be interpreted within the Docker guest. Mutually exclusive to `--resume`.

`--value-model PATH`: Specify the path to the YAML configuration file for the *V* model. The path must be one that can be interpreted within the Docker guest.

`--q1-model PATH`: Specify the path to the YAML configuration file for the *Q1* model. The path must be one that can be interpreted within the Docker guest.

`--q2-model PATH`: Specify the path to the YAML configuration file for the *Q2* model. The path must be one that can be interpreted within the Docker guest.

`--inverse-temperature BETA`: Specify the inverse temperature hyperparameter of AWR. The argument must be a positive real number.

`--advantage-threshold ADV_THRESHOLD`: Specify the threshold for advantage clipping. The argument must be a positive real number.

`--batch-size N`: Specify the batch size. The argument must be a positive integer.

`--optimizer {sgd|adam|radam|lamb}`: Specify the optimizer. Default to `lamb`.

`--momentum MOMENTUM`: Specify the momentum factor. Only meaningful for `sgd`. The argument must be a real number in the range (0.0, 1.0). Default to `0.9`.

`--learning-rate LR`: Specify the learning rate. The argument must be a positive real number. Default to `0.1` for `sgd`, `0.001` for `adam`, `radam`, and `lamb`.

`--epsilon EPS`: Specify the epsilon parameter. Only meaningful for `adam`, `radam`, and `lamb`. The argument must be a positive real number. Default to `1.0e-8` for `adam` and `radam`, `1.0e-6` for `lamb`.

`--checkpointing`: Enable checkpointing.

`--gradient-accumulation-steps NSTEPS`: Specify the number of steps for gradient accumulation. The argument must be a positive integer. Default to `1`.

`--max-gradient-norm NORM`: Specify the norm threshold for gradient clipping. The argument must be a positive real number. Default to `10.0`.

`--num-epochs NEPOCHS`: Specify the number of epochs to iterate. The argument must be a positive integer. Default to `1`.

`--output-prefix PATH`: Specify the output prefix. The path must be one that can be interpreted within the Docker guest.

`--experiment-name NAME`: Specify the experiment name. Default to the start time of the experiment in the format `YYYY-MM-DD-hh-mm-ss`. The final path to the output will becomes `PATH/NAME`.

`--num-epoch-digits NDIGITS`: Specify the number of digits to index epochs. The argument must be an integer greater or equal to `1`. Default to `2`.

`--snapshot-interval NSAMPLES`: Specify the interval between snapshots. The argument must be a non-negative integer. `0` means that no snapshot is taken at all. Default to `0`.

`--resume`: Resume the experiment from the latest snapshot in the path `PATH/NAME`.
