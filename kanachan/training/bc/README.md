# `kanachan.training.bc` Python submodule

An implementation of training for supervised learning to mimic actions recorded in game records (a.k.a. behavioral cloning).

## Prerequisits

The following items are required to run the program in this directory:

- Linux or Microsoft Windows
- [Docker](https://www.docker.com/),
- NVIDIA driver, and
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html). In the case of Docker Desktop, there is no need to install it as it is bundled.
- ([CUDA](https://developer.nvidia.com/cuda-toolkit) is not required.)

For detailed installation instructions for the above prerequisite items, refer to those for each OS and distribution.

After the installation of the prerequisite items, execute the following command with the top directory of the working tree of this repository as the current directory:

```sh
docker build -f kanachan/Dockerfile -t cryolite/kanachan . && docker build -f kanachan/training/bc/Dockerfile -t cryolite/kanachan.training.bc .
```

## `train.py`

#### Usage

```
docker run --gpus all -v /path/to/host-data:/workspace/data --rm cryolite/kanachan.training.bc OPTIONS...
```

If you want to run this program on specific GPUs, modify the `--gpus` option for the `docker run` command (see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/user-guide.html#gpu-enumeration) or specify `device.type` option (see below).

#### Options

Options are specified in the [Hydra](https://hydra.cc/) manner.

`device={cpu|cuda}`: Specify the device on which the training is performed. Default to the value guessed from PyTorch build information. See the table below for the detailed meaning of the options:

| `device` | `device.type` | `device.dtype` | `device.amp_dtype` |
|----------|---------------|----------------|--------------------|
| `cpu`    | `cpu`         | `float32`      | (N/A)              |
| `cuda`   | `cuda`        | `float32`      | `float16`          |

`device.type={cpu|cuda}`: Specify the device on which the training is performed. Override the value by the `device` option.

`device.dtype={float64|double|float32|float|float16|half}`: Specify the PyTorch [`dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch-dtype). Override the value by the `device` option.

`device.amp_dtype={float64|double|float32|float|float16|half}`: Specify the PyTorch `dtype` for [automatic mixed precision (AMP)](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html). Setting this option to the same as `device.dtype` disables AMP. Override the value by the `device` option.

`training_data=PATH`: Specify the path to the training data. `PATH` must be a path that can be interpreted within the Docker guest environment.

`num_workers=NWORKERS`: Specify the number of workers used in data loading. `NWORKERS` must be a non-negative integer. `0` means that the main process is used to load data. Default to `0` for CPU, and `2` for CUDA.

`encoder={bert_tiny|bert_mini|bert_small|bert_medium|bert_base|bert_large}`: Specify the encoder structure. Default to `bert_base`. See the table below for the detailed meaning of the options:

| `encoder`     | `encoder.position_encoder` | `encoder.dimension` | `encoder.num_heads` | `encoder.dim_feedforward` | `encoder.activation_function` | `encoder.dropout` | `encoder.num_layers` | `encoder.load_from` |
|---------------|----------------------------|---------------------|---------------------|---------------------------|-------------------------------|-------------------|----------------------|---------------------|
| `bert_tiny`   | `position_embedding`       | `128`               | `2`                 | `512`                     | `gelu`                        | `0.1`             | `2`                  | (N/A)               |
| `bert_mini`   | `position_embedding`       | `256`               | `4`                 | `1024`                    | `gelu`                        | `0.1`             | `4`                  | (N/A)               |
| `bert_small`  | `position_embedding`       | `512`               | `8`                 | `2048`                    | `gelu`                        | `0.1`             | `4`                  | (N/A)               |
| `bert_medium` | `position_embedding`       | `512`               | `8`                 | `2048`                    | `gelu`                        | `0.1`             | `8`                  | (N/A)               |
| `bert_base`   | `position_embedding`       | `768`               | `12`                | `3072`                    | `gelu`                        | `0.1`             | `12`                 | (N/A)               |
| `bert_large`  | `position_embedding`       | `1024`              | `16`                | `4096`                    | `gelu`                        | `0.1`             | `24`                 | (N/A)               |

`encoder.position_encoder={positional_encoding|position_embedding}`: Specify the method of encoding positions. `positional_encoding` is the method used in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) to encode positions with a sinusoidal function. `position_embedding` is the method used in the paper ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805) to encode positions with embeddings. Override the value specified by the `encoder` option.

`encoder.dimension=DIM`: Specify the embedding dimension for the encoder. `DIM` must be a positive integer. Override the value by the `encoder` option.

`encoder.num_heads=NHEADS`: Specify the number of heads in each encoder layer. `NHEADS` must be a positive integer. Override the value by the `encoder` option.

`encoder.dim_feedforward=DIM_FEEDFORWARD`: Specify the dimension of the feedforward networks in each encoder layer. `DIM_FEEDFORWARD` must be a positive integer. Override the value by the `encoder` option.

`encoder.activation_function={relu|gelu}`: Specify the activation function for the feedforward networks in each encoder layer. Override the value specified by the `encoder` option.

`encoder.dropout=DROPOUT`: Specify the dropout ratio for the feedforward networks in each encoder layer. `DROPOUT` must be a real number in the range \[0.0, 1.0\). Override the value by the `encoder` option.

`encoder.num_layers=NLAYERS`: Specify the number of encoder layers. `NLAYERS` must be a positive integer. Override the value by the `encoder` option.

`encoder.load_from=INITIAL_ENCODER`: Specify the path to the initial encoder snapshot. `INITIAL_ENCODER` must be a path that can be interpreted within the Docker guest environment. Mutually exclusive to the `initial_model` and `initial_model_prefix` options.

`decoder={single|double|triple}`: Specify the decoder structure. Default to `single`. See the table below for the detailed meaning of the options:

| `decoder` | `decoder.dimension`         | `decoder.activation_function` | `decoder.dropout` | `decoder.num_layers` | `decoder.load_from` |
|-----------|-----------------------------|-------------------------------|-------------------|----------------------|---------------------|
| `single`  | (N/A)                       | (N/A)                         | (N/A)             | `1`                  | (N/A)               |
| `double`  | (`encoder.dim_feedforward`) | `gelu`                        | `0.1`             | `2`                  | (N/A)               |
| `triple`  | (`encoder.dim_feedforward`) | `gelu`                        | `0.1`             | `3`                  | (N/A)               |

`decoder.dim_feedforward=DIM_FEEDFORWARD`: Specify the dimension of the feedforward networks in each decoder layer. `DIM_FEEDFORWARD` must be a positive integer. Override the value by the `decoder` option.

`decoder.activation_function={relu|gelu}`: Specify the activation function for the feedforward networks in each decoder layer. Override the value by the `decoder` option.

`decoder.dropout=DROPOUT`: Specify the dropout ratio for the feedforward networks in each decoder layer. `DROPOUT` must be a real number in the range \[0.0, 1.0\). Override the value by the `decoder` option.

`decoder.num_layers=NLAYERS`: Specify the number of decoder layers. `NLAYERS` must be a positive integer. Override the value by the `decoder` option.

`decoder.load_from=INITIAL_DECODER`: Specify the path to the initial decoder snapshot. `INITIAL_DECODER` must be a path that can be interpreted within the Docker guest environment. Mutually exclusive to the `initial_model` and `initial_model_prefix` options.

`initial_model_prefix=PREFIX`: Specify the prefix to the initial model snapshot. `PREFIX` must be a path that can be interpreted within the Docker guest environment. Mutually exclusive to the `encoder.load_from` and `decoder.load_from` options.

`initial_model_index=INDEX`: Speficy the index of the initial model snapshot. `INDEX` must be a non-negative integer. Must be used with `initial_model_prefix` option.

`checkpointing={false|true}`: Enable [checkpointing](https://pytorch.org/docs/stable/checkpoint.html). Default to `false`.

`batch_size=N`: Specify the batch size. `N` must be a positive integer.

`gradient_accumulation_steps=NSTEPS`: Specify the number of steps for [gradient accumulation](https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation). `NSTEPS` must be a positive integer. Default to `1`.

`max_gradient_norm=NORM`: Specify the norm threshold for gradient clipping. `NORM` must be a positive real number. Default to `1.0`.

`optimizer={sgd|adam|radam|lamb}`: Specify the optimizer preset. Default to `adam`. See the table below for the detailed meaning of the options:

| `optimizer` | `optimizer.type`                             | `optimizer.momentum` | `optimizer.epsilon` | `optimizer.learning_rate` | `optimizer.warmup_start_lr` | `optimizer.warmup_steps` | `optimizer.annealing_steps` | `optimizer.annealing_steps_factor` | `optimizer.use_zero` | `optimizer.initialize` |
|-------------|----------------------------------------------|----------------------|---------------------|---------------------------|-----------------------------|--------------------------|-----------------------------|------------------------------------|----------------------|------------------------|
| `sgd`       | `sgd`                                        | `0.0`                | (N/A)               | (EXPLICITLY REQUIRED)     | `1.0e-8`                    | `0`                      | `0`                         | `1`                                | `false`              | `false`                |
| `adam`      | [`adam`](https://arxiv.org/abs/1412.6980)    | (N/A)                | `1.0e-8`            | `1.0e-4`                  | `1.0e-8`                    | `0`                      | `0`                         | `1`                                | `false`              | `false`                |
| `radam`     | [`radam`](https://arxiv.org/abs/1908.03265)  | (N/A)                | `1.0e-8`            | `1.0e-4`                  | `1.0e-8`                    | `0`                      | `0`                         | `1`                                | `false`              | `false`                |
| `lamb`      | [`lamb`](https://arxiv.org/abs/1904.00962)   | (N/A)                | `1.0e-6`            | `1.0e-4`                  | `1.0e-8`                    | `0`                      | `0`                         | `1`                                | `false`              | `false`                |

`optimizer.type={sgd|adam|radam|lamb}` Specify the optimizer type. Override the value specified by the `optimizer` option.

`optimizer.momentum=MOMENTUM`: Specify the momentum factor. Only meaningful for `optimizer.type=sgd`. `MOMENTUM` must be a real number in the range \[0.0, 1.0\). Override the value specified by the `optimizer` option.

`optimizer.epsilon=EPS`: Specify the epsilon parameter. Only meaningful for `adam`, `radam`, and `lamb`. `EPS` must be a positive real number. Override the value specified by the `optimizer` option.

`optimizer.learning_rate=LR`: Specify the learning rate. `LR` must be a positive real number. Override the value specified by the `optimizer` option.

`optimizer.warmup_start_lr=START_LR`: Specify the learning rate at the beginning of the warm-up. `START_LR` must not be greater than `LR`. Override the value specified by the `optimizer` option.

`optimizer.warmup_steps=WARMUP_STEPS`: Specify the warm-up steps. `WARMUP_STEPS` must be a non-negative integer. Specifying `0` disables the warm-up. Override the value specified by the `optimizer` option.

`optimizer.annealing_steps=ANNEALING_STEPS`: Specify the steps for the cosine anneaning. `ANNEALING_STEPS` must be a non-negative integer. Specifying `0` disables the cosine annealing. Override the value specified by the `optimizer` option.

`optimizer.annealing_steps_factor=ANNEALING_STEPS_FACTOR`: Specify the multiplier for the number of steps at each cycle of annealing. `ANNEALING_STEPS_FACTOR` must be a positive integer. For example, specifying `2` will increase the number of annealing steps in a sequence of `1, 2, 4, 8, ...`. Override the value specified by the `optimizer` option.

`optimizer.use_zero={false|true}`: Specify whether to enable [ZeRO](https://arxiv.org/abs/1910.02054). Override the value specified by the `optimizer` option.

`optimizer.initialize={false|true}`: Specify whether to start from the initialized optimizer without using a snapshot, even if one is found. Override the value specified by the `optimizer` option.

`snapshot_interval=NSAMPLES`: Specify the interval between snapshots. `NSAMPLES` must be a non-negative integer. `0` means that no snapshot is taken at all. Default to `0`.
