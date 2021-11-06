ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:21.10-py3
FROM $BASE_IMAGE

RUN apt-get update && \
    apt-get -y dist-upgrade && \
    apt-get -y install \
      git \
      locales \
      locales-all && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8 ja_JP.UTF-8

RUN git clone https://github.com/NVIDIA/apex.git && \
    (cd apex && \
     pip install --use-feature=in-tree-build -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./) && \
    rm -rf apex

COPY . /workspace/kanachan
RUN cd /workspace/kanachan && pip install --use-feature=in-tree-build .
