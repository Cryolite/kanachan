# `cryolite/kanachan.annotate4rl` Docker image

A Docker image to create annotation data for offline reinforcement learning.

### How to Build

First [build the cryolite/kanachan Docker image](../../kanachan/README.md#cryolitekanachan-docker-image). Then, execute the following command with the top directory of the working tree of this repository as the current directory:

```bash
kanachan$ docker build -f bin/annotate4rl/Dockerfile -t cryolite/kanachan.annotate4rl .
```

# `annotate4rl.py` Python program

### Usage

```sh
$ docker run --rm cryolite/kanachan.annotate4rl [OPTION]... [INPUT_FILE]
```

or

```sh
$ another-command | docker run -i --rm cryolite/kanachan.annotate4rl [OPTION]... [-]
```

Convert the [training data format for behavioral cloning](https://github.com/Cryolite/kanachan/wiki/Notes-on-Training-Data#training-data-format-for-behavioral-cloning) in the file specified by the `INPUT_FILE` argument to the [training data format for offline reinforcement learning](https://github.com/Cryolite/kanachan/wiki/Notes-on-Training-Data#training-data-format-for-offline-reinforcement-learning). If `-` is specified for this argument or omitted, the conversion will be performed on the standard input. When using standard input, don't forget to add the `-i` option to the `docker run` command.

Note that the input must contain all the annotations for each game. The easiest way to ensure this precondition is met is by piping the Mahjong Soul game record data converted with [`cryolite/kanachan.annotate`](../../src/annotation#annotation), as follows:

```sh
$ docker run -v /path/to/data:/data:ro --rm cryolite/kanachan.annotate | docker run -i --rm cryolite/kanachan.annotate4rl [OPTION]...
```
