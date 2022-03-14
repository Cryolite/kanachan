ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:22.02-py3
FROM $BASE_IMAGE

RUN apt-get update && \
    apt-get -y dist-upgrade && \
    apt-get -y install \
      git \
      libboost-python-dev \
      libboost-stacktrace-dev \
      locales \
      locales-all \
      python3-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    locale-gen en_US.UTF-8 ja_JP.UTF-8 && \
    pip3 install -U pip && \
    useradd -ms /bin/bash ubuntu && \
    chown ubuntu:ubuntu /workspace

USER ubuntu

ENV PATH="/home/ubuntu/.local/bin:${PATH}"

RUN pip3 install -U \
      mahjong \
      setuptools \
      wheel

# `marisa-trie` only supports for in-tree build.
RUN mkdir -p /home/ubuntu/.local/src && \
    pushd /home/ubuntu/.local/src && \
    git clone 'https://github.com/s-yata/marisa-trie.git' && \
    popd && \
    pushd /home/ubuntu/.local/src/marisa-trie && \
    autoreconf -i && \
    CFLAGS='-DNDEBUG -O3 -flto' CXXFLAGS='-DNDEBUG -O3 -flto' ./configure --prefix=/home/ubuntu/.local --enable-native-code --disable-static && \
    make -j && \
    make install && \
    popd

RUN mkdir -p /home/ubuntu/.local/src && \
    pushd /home/ubuntu/.local/src && \
    git clone 'https://github.com/tomohxx/shanten-number' && \
    pushd shanten-number && \
    tar xzvf index.tar.gz && \
    popd && \
    popd

RUN git clone https://github.com/NVIDIA/apex.git && \
    (cd apex && \
     MAX_JOBS=$(nproc) pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./) && \
    rm -rf apex

COPY --chown=ubuntu . /workspace/kanachan

RUN pushd /workspace/kanachan/src/common && \
    protoc -I. --cpp_out=. mahjongsoul.proto && \
    popd && \
    mkdir -p /workspace/kanachan/build && \
    pushd /workspace/kanachan/build && \
    cmake -DPYTHON_VERSION="$(python3 -V | sed -e 's@^Python[[:space:]]\{1,\}\([[:digit:]]\{1,\}\.[[:digit:]]\{1,\}\)\.[[:digit:]]\{1,\}@\1@')" \
          -DCMAKE_BUILD_TYPE=Release \
          .. && \
    VERBOSE=1 make -j make_trie _simulation && \
    mkdir -p /home/ubuntu/.local/share/kanachan && \
    src/xiangting/make_trie && \
    popd && \
    pushd /workspace/kanachan && \
    cp build/src/simulation/lib_simulation.so kanachan/simulation/_simulation.so && \
    pip3 install . && \
    popd
