ARG PYTHON_VERSION=3.10.16
FROM ubuntu:jammy
ARG PYTHON_VERSION

ENV PATH=/usr/local/bin${PATH:+:$PATH}

SHELL ["/bin/bash", "-c"]

RUN set -euxo pipefail; \
    apt-get update; \
    apt-get -y dist-upgrade; \
    DEBIAN_FRONTEND=noninteractive apt-get -y install \
      curl \
      g++ \
      git \
      gnupg \
      libboost-all-dev \
      libbz2-dev \
      libffi-dev \
      libgdbm-compat-dev \
      libgdbm-dev \
      liblzma-dev \
      libncurses-dev \
      libreadline-dev \
      libsqlite3-dev \
      libssl-dev \
      libtool \
      locales \
      locales-all \
      make \
      tk-dev \
      uuid-dev \
      xz-utils \
      zip \
      zlib1g-dev; \
    apt-get clean && rm -rf /var/lib/apt/lists/*; \
    locale-gen en_US.UTF-8 ja_JP.UTF-8

# Install CMake.
RUN set -euxo pipefail; \
    mkdir -p /opt/src; \
    pushd /opt; \
    git clone 'https://github.com/Cryolite/prerequisites.git'; \
    /opt/prerequisites/cmake/install --debug; \
    popd

# Install Python.
RUN set -euxo pipefail; \
    mkdir -p /opt/src; \
    pushd /opt/src; \
    curl -fLsSo Python-${PYTHON_VERSION}.tar.xz "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz"; \
    tar xf Python-${PYTHON_VERSION}.tar.xz; \
    rm /opt/src/Python-${PYTHON_VERSION}.tar.xz; \
    pushd /opt/src/Python-${PYTHON_VERSION}; \
    ./configure --enable-optimizations --with-lto; \
    make -j; \
    make install; \
    popd; \
    rm -rf /opt/src/Python-${PYTHON_VERSION}; \
    popd; \
    python3 -m pip install -U pip

# Install Protocol Buffers.
RUN set -euxo pipefail; \
    mkdir -p /opt/src; \
    pushd /opt/src; \
    git clone 'https://github.com/protocolbuffers/protobuf.git'; \
    pushd /opt/src/protobuf; \
    git submodule update --init --recursive; \
    cmake . -DCMAKE_CXX_STANDARD=17; \
    cmake --build . --parallel $(nproc); \
    cmake --install .; \
    popd; \
    rm -rf /opt/src/protobuf; \
    popd

# Install Nyanten.
RUN set -euxo pipefail; \
    mkdir -p /opt/src; \
    pushd /opt/src; \
    git clone 'https://github.com/Cryolite/nyanten.git'; \
    cmake -S nyanten -B nyanten/build; \
    cmake --install nyanten/build; \
    rm -rf /opt/src/nyanten; \
    popd

COPY . /opt/kanachan

RUN set -euxo pipefail; \
    pushd /opt/kanachan/src/common; \
    protoc -I. --cpp_out=. mahjongsoul.proto; \
    popd

RUN set -euxo pipefail; \
    pushd /opt/kanachan; \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_RPATH=/workspace/.local/lib; \
    VERBOSE=1 cmake --build build --target _nyanten; \
    cp build/src/nyanten/lib_nyanten.so mjai.app/nyanten/_nyanten.so; \
    popd

RUN set -euxo pipefail; \
    mkdir -p /opt/kanachan/mjai.app/build; \
    pushd /opt/kanachan/mjai.app/build; \
    python3 -m pip install -t . mahjong==1.1.11 torch tensordict ../..; \
    cp -r ../nyanten .; \
    cp ../hand_calculator.py .; \
    cp ../config.json.orig config.json; \
    cp ../_kanachan.py ../bot.py .; \
    rm -rf cusparselt; \
    rm -rf nvidia; \
    rm -rf torch; \
    rm -rf triton; \
    sed -i -e 's/@torch.compile//' kanachan/nn/encoder.py; \
    sed -i -e 's/@torch.compile//' kanachan/nn/decoder.py; \
    sed -i -e 's/@torch.compile//' kanachan/nn/qr_decoder.py; \
    sed -i -e 's/weights_only=True/weights_only=False/' kanachan/model_loader.py; \
    popd
