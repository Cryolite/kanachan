ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ARG PYTHON_VERSION=3.12.7
ARG BUILD_TYPE=Release
FROM $BASE_IMAGE
ARG PYTHON_VERSION
ARG BUILD_TYPE

SHELL ["/bin/bash", "-c"]

# Install prerequisite packages.
RUN set -euxo pipefail; \
    apt-get -y update; \
    apt-get -y dist-upgrade; \
    apt-get -y install \
      autoconf \
      automake \
      curl \
      git-core \
      libbz2-dev \
      libffi-dev \
      liblzma-dev \
      libncurses-dev \
      libreadline-dev \
      libsqlite3-dev \
      libssl-dev \
      libtool \
      locales \
      locales-all \
      protobuf-compiler; \
    apt-get clean && rm -rf /var/lib/apt/lists/*; \
    locale-gen en_US.UTF-8 ja_JP.UTF-8; \
    useradd -ms /bin/bash ubuntu; \
    mkdir /workspace; \
    chown ubuntu:ubuntu /workspace

USER ubuntu

# Install `pyenv`.
RUN set -euxo pipefail; \
    curl 'https://pyenv.run' | bash
ENV PYENV_ROOT="/home/ubuntu/.pyenv"
ENV PATH="$PYENV_ROOT/bin${PATH:+:$PATH}"

# Create a Python environment, and install prerequisite Python packages.
RUN set -euxo pipefail; \
    if [[ $PYTHON_VERSION == latest ]]; then \
        PYTHON_LATEST_VERSION=$(pyenv install -l | grep -Eo '^[[:space:]]+[[:digit:]]+\.[[:digit:]]+\.[[:digit:]]+$' | grep -Eo '[[:digit:]]+\.[[:digit:]]+\.[[:digit:]]+' | LANG=C.UTF-8 sort -V | tail -n 1); \
        PYTHON_VERSION=$PYTHON_LATEST_VERSION; \
    fi; \
    PYTHON_VERSION_MAJOR=$(echo $PYTHON_VERSION | grep -Eo '^[[:digit:]]+\.[[:digit:]]+'); \
    pyenv install $PYTHON_VERSION; \
    pyenv global $PYTHON_VERSION; \
    pushd /home/ubuntu/.pyenv/versions/$PYTHON_VERSION/lib; \
    ln -s libpython$PYTHON_VERSION_MAJOR.so libpython.so; \
    popd; \
    echo 'eval "$(pyenv init -)"' >> /home/ubuntu/.bashrc; \
    echo 'eval "$(pyenv init -)"' >> /home/ubuntu/.profile; \
    eval "$(pyenv init -)"; \
    python3 -m pip install -U pip; \
    python3 -m pip install -U \
      build \
      ninja \
      packaging \
      setuptools \
      wheel

# Install PyTorch.
RUN set -euxo pipefail; \
    eval "$(pyenv init -)"; \
    python3 -m pip install -U torch --index-url https://download.pytorch.org/whl/cu121

# Install Apex.
RUN set -euxo pipefail; \
    eval "$(pyenv init -)"; \
    pushd /workspace; \
    git clone 'https://github.com/NVIDIA/apex.git'; \
    pushd apex; \
    python3 -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./; \
    popd; \
    rm -rf apex; \
    popd

# Clone `prerequisites`.
RUN set -euxo pipefail; \
    pushd /workspace; \
    git clone 'https://github.com/Cryolite/prerequisites.git'; \
    popd

# Install the latest GCC.
RUN set -euxo pipefail; \
    eval "$(pyenv init -)"; \
    /workspace/prerequisites/gcc/install --debug --prefix /home/ubuntu/.local

# Set environment variables.
ENV C_INCLUDE_PATH=/home/ubuntu/.local/include
ENV CPLUS_INCLUDE_PATH=/home/ubuntu/.local/include
ENV LIBRARY_PATH="/home/ubuntu/.local/lib64:/home/ubuntu/.local/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="/home/ubuntu/.local/lib64:/home/ubuntu/.local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ENV PATH="/home/ubuntu/.local/bin${PATH:+:$PATH}"
ENV CC=/home/ubuntu/.local/bin/gcc
ENV CXX=/home/ubuntu/.local/bin/g++

# Install CMake.
RUN set -euxo pipefail; \
    /workspace/prerequisites/cmake/install --debug --prefix /home/ubuntu/.local

# Install libbacktrace.
RUN set -euxo pipefail; \
    /workspace/prerequisites/libbacktrace/install --debug --prefix /home/ubuntu/.local

# Install Boost.
RUN set -euxo pipefail; \
    eval "$(pyenv init -)"; \
    PYTHON_PREFIX="$(python3 -c 'import sys; print(sys.prefix);')"; \
    echo "import toolset : using ; using python : : \"$PYTHON_PREFIX/bin/python3\" ;" > /home/ubuntu/user-config.jam; \
    /workspace/prerequisites/boost/download --debug --source-dir /workspace/boost; \
    if [[ $BUILD_TYPE == 'Debug' ]]; then \
      /workspace/prerequisites/boost/build --debug --source-dir /workspace/boost --prefix /home/ubuntu/.local -- \
        -d+2 --with-headers --with-stacktrace --with-python --with-timer --build-type=complete --layout=tagged \
        toolset=gcc variant=debug threading=multi link=shared runtime-link=shared \
        cxxflags=-D_GLIBCXX_DEBUG cxxflags=-D_GLIBCXX_DEBUG_PEDANTIC \
        cflags=-fsanitize=address cxxflags=-fsanitize=address linkflags=-fsanitize=address \
        cflags=-fsanitize=undefined cxxflags=-fsanitize=undefined linkflags=-fsanitize=undefined; \
    else \
      /workspace/prerequisites/boost/build --debug --source-dir /workspace/boost --prefix /home/ubuntu/.local -- \
        -d+2 --with-headers --with-stacktrace --with-python --with-timer --build-type=complete --layout=tagged \
        toolset=gcc variant=release threading=multi link=shared runtime-link=shared; \
    fi; \
    rm -rf /workspace/boost

# Install `marisa-trie`.
RUN set -euxo pipefail; \
    pushd /workspace; \
    git clone 'https://github.com/s-yata/marisa-trie.git'; \
    pushd marisa-trie; \
    autoreconf -i; \
    if [[ $BUILD_TYPE == 'Debug' ]]; then \
      CFLAGS='-D_DEBUG -g' CXXFLAGS='-D_DEBUG -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -g' ./configure --prefix=/home/ubuntu/.local --enable-native-code --disable-static; \
    else \
      CFLAGS='-DNDEBUG -O3 -flto' CXXFLAGS='-DNDEBUG -O3 -flto' ./configure --prefix=/home/ubuntu/.local --enable-native-code --disable-static; \
    fi; \
    make -j; \
    make install; \
    popd; \
    rm -rf marisa-trie; \
    popd

# Clone `shanten-number`.
RUN set -euxo pipefail; \
    pushd /workspace; \
    git clone 'https://github.com/tomohxx/shanten-number.git'; \
    pushd shanten-number; \
    tar xzvf data.tar.gz; \
    popd; \
    popd

COPY --chown=ubuntu . /workspace/kanachan

# Install `kanachan`.
RUN set -euxo pipefail; \
    eval "$(pyenv init -)"; \
    pushd /workspace/kanachan/src/common; \
    protoc -I. --cpp_out=. mahjongsoul.proto; \
    popd; \
    mkdir -p /workspace/kanachan/build; \
    pushd /workspace/kanachan/build; \
    if [[ $BUILD_TYPE == 'Debug' ]]; then \
      cmake \
        -DSHANTEN_NUMBER_SOURCE_PATH=/workspace/shanten-number \
        -DCMAKE_BUILD_TYPE=Debug \
        ..; \
    else \
      cmake \
        -DSHANTEN_NUMBER_SOURCE_PATH=/workspace/shanten-number \
        -DCMAKE_BUILD_TYPE=Release \
        ..; \
    fi; \
    make VERBOSE=1 -j make_trie simulation; \
    mkdir -p /home/ubuntu/.local/share/kanachan; \
    src/xiangting/make_trie /workspace/shanten-number /home/ubuntu/.local/share/kanachan; \
    popd; \
    rm -rf /workspace/shanten-number; \
    pushd /workspace/kanachan; \
    cp build/src/simulation/libsimulation.so kanachan/simulation/_simulation.so; \
    python3 -m build -w; \
    python3 -m pip install -U dist/*.whl; \
    popd; \
    rm -rf /workspace/kanachan/build
