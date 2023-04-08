# `cryolite/kanachan.simulation` Docker image

### How to Build

Make the top (`kanachan`) directory of the working tree of this repository the current directory, and execute the following commands:

```bash
kanachan$ docker build --pull -f kanachan/Dockerfile -t cryolite/kanachan .
kanachan$ docker build --pull -f kanachan/simulation -t cryolite/kanachan.simulation .
```

# `kanachan.simulation.run` Python program

The `kanachan.simulation.run` Python program simulates Mahjong Soul ranking matches to compare the performance of two given models.

### Prerequisits

The following items are required to run this program:

- [Docker](https://www.docker.com/),
- NVIDIA driver, and
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html).
- ([CUDA](https://developer.nvidia.com/cuda-toolkit) is not required.)

For detailed installation instructions for the above prerequisite items, refer to those for each OS and distribution.

After the installation of the prerequisite items, build the `cryolite/kanachan.simulation` Docker image (see above).

### Usage

```
$ docker run --gpus all -v /path/to/host-data:/workspace/data -it --rm cryolite/kanachan python3 -m kanachan.simulation.run OPTIONS...
```

If you want to run this program on a specific GPU, use the `--gpus device=n` option, where `n` is the GPU number displayed by the [`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface) command, or `--device` option (see below).

#### Options

`--device DEVICE`: Specify the device on which to run simulation. The allowed values are `cpu`, `cuda`, and `cuda:n`, where `n` is the GPU number displayed by the `nvidia-smi` command. If no value is specified for this option, a suitable value will be inferred from the PyTorch build information.

`--dtype DTYPE`: Specify the [PyTorch `dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) for simulation. The allowed values are `float16`, `half`, `float32`, `float`, `float64`, or `double`. Default to `float32` for CPU, and `float16` for CUDA.

`--baseline-model BASELINE_MODEL`: Specify the path to the model file of the [baseline model](https://github.com/Cryolite/kanachan/wiki/Methods-and-Metrics-in-Performance-Comparison-and-Evaluation). The path must be one that can be interpreted within the Docker guest.

`--baseline-grade BASELINE_GRADE`: Specify which ranking grade the baseline model should behave as in simulation. `0` means Novice 1 (初心1), `1` Novice 2 (初心2), ..., `14` Saint 3 (雀聖3), and `15` Celestial (魂天). The value must be in the range `0` (Novice 1, 初心1) to `15` (Celestial, 魂天).

`--proposed-model PROPOSED_MODEL`: Specify the path to the model file of the [proposed model](https://github.com/Cryolite/kanachan/wiki/Methods-and-Metrics-in-Performance-Comparison-and-Evaluation). The path must be one that can be interpreted within the Docker guest.

`--proposed-grade PROPOSED_GRADE`: Specify which ranking grade the proposed model should behave as in simulation. The meaning of the value is the same as one for the `--baseline-grade` option.

`--room ROOM`: Specify which room simulation emulates. `0` means Bronze Room (銅の間), `1` Silver Room (銀の間), `2` Gold Room (金の間), `3` Jade Room (玉の間), and `4` Throne Room (王座の間).

`--dongfengzhan`: Specify simulation emulates quater-length games (Dong Feng Zhan, 東風戦) instead of half-length games (Ban Zhuang Zhan, 半荘戦).

`--mode MODE`: Specify the [comparison style](https://github.com/Cryolite/kanachan/wiki/Methods-and-Metrics-in-Performance-Comparison-and-Evaluation) of simulation. The allowed values are `2vs2` or `1vs3`. Default to `2vs2`. Ignored if the `--non-duplicated` option is specified.

`--non-duplicated`: Disable [duplicated mahjong](https://github.com/Cryolite/kanachan/wiki/Methods-and-Metrics-in-Performance-Comparison-and-Evaluation).

`-n N`: Specify the number of _sets_ of simulation. A set consists of 6 games in the 2vs2 mode, 4 games in the 1vs3 mode, or 1 game in the non-duplicated mode. See [Methods and Metrics in Performance Comparison and Evaluation](https://github.com/Cryolite/kanachan/wiki/Methods-and-Metrics-in-Performance-Comparison-and-Evaluation)
for detail.

`--batch-size BATCH_SIZE`: Specify the batch size of simulation. The larger the batch size is, the higher it will result in simulation throughput but the more main/GPU memory consumption. Default to `1`.

`--concurrency CONCURRENCY`: Specify the number of threads used for simulation. The `CONCURRENCY` must be greater or equal to `BATCH_SIZE * 2 - 1`. Default to `BATCH_SIZE * 2`.
