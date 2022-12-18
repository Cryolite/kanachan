# `cryolite/kanachan` Docker image

This directory becomes the `kanachan` Python module. The module is intended to run on a docker container. Therefore, it is necessary to build the Docker image before using the module. The command to build the image is, with the top-level directory of the working tree of this repository as the current directory, as follows:

```
$ docker build -f kanachan/Dockerfile -t cryolite/kanachan .
```

If the image fails to build, try lowering the version of the base image as follows:

```
$ docker build -f kanachan/Dockerfile --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:xx.yy-py3 -t cryolite/kanachan .
```

# `kanachan` Python module

## Submodules

### [`kanachan.training`](training)

Training programs and prediction modules with [PyTorch](https://pytorch.org/).

### [`kanachan.simulation`](simulation)

A simulation program.
