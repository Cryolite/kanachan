ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:22.08-py3
FROM $BASE_IMAGE

SHELL ["/bin/bash", "-c"]

# Install prerequisite packages.
RUN set -euxo pipefail; \
    apt-get -y update; \
    apt-get -y dist-upgrade; \
    apt-get -y install \
      libssl-dev \
      locales \
      locales-all \
      python3-dev \
      python3-venv; \
    apt-get clean && rm -rf /var/lib/apt/lists/*; \
    locale-gen en_US.UTF-8 ja_JP.UTF-8; \
    useradd -ms /bin/bash ubuntu; \
    chown ubuntu:ubuntu /workspace

USER ubuntu

# Install Apex.
RUN set -euxo pipefail; \
    python3 -m venv --system-site-packages "$HOME/.local"; \
    . "$HOME/.local/bin/activate"; \
    python3 -m pip install -U pip; \
    pushd /workspace; \
    git clone 'https://github.com/NVIDIA/apex.git'; \
    pushd apex; \
    MAX_JOBS=$(($(nproc) / 2)) python3 -m pip install -v --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./; \
    popd; \
    rm -rf apex; \
    popd

# Install MTAdam.
RUN set -euxo pipefail; \
    pushd /workspace; \
    git clone 'https://github.com/ItzikMalkiel/MTAdam.git'; \
    pushd MTAdam; \
    mkdir mtadam; \
    cp mtadam.py mtadam; \
    echo 'from .mtadam import MTAdam' > mtadam/__init__.py; \
    echo $'[build-system]\n\
requires = ["setuptools"]\n\
build-backend = "setuptools.build_meta"\n\
\n\
[project]\n\
name = "MTAdam"\n\
authors = [\n\
    {"name" = "Itzik Malkiel"},\n\
]\n\
description = "Automatic Balancing of Multiple Training Loss Terms"\n\
readme = "README.txt"\n\
requires-python = ">=3.8"\n\
dependencies = [\n\
    "tensorflow",\n\
]\n\
version = "0.0.1"\n' > pyproject.toml; \
    python3 -m pip install .; \
    popd; \
    rm -rf MTAdam; \
    popd

# Clone `prerequisites`.
RUN set -euxo pipefail; \
    pushd /workspace; \
    git clone 'https://github.com/Cryolite/prerequisites.git'; \
    popd

# Install the latest GCC.
RUN set -euxo pipefail; \
    export LIBRARY_PATH="${LIBRARY_PATH/%:/}"; \
    /workspace/prerequisites/gcc/install --debug --prefix "$HOME/.local"

# Set environment variables.
ENV C_INCLUDE_PATH "/home/ubuntu/.local/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
ENV CPLUS_INCLUDE_PATH "/home/ubuntu/.local/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
ENV LIBRARY_PATH "/home/ubuntu/.local/lib64:/home/ubuntu/.local/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
ENV LD_LIBRARY_PATH "/home/ubuntu/.local/lib64:/home/ubuntu/.local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ENV PATH "/home/ubuntu/.local/bin${PATH:+:$PATH}"

# Install CMake.
RUN set -euxo pipefail; \
    /workspace/prerequisites/cmake/install --debug --prefix "$HOME/.local"

# Install libbacktrace.
RUN set -euxo pipefail; \
    /workspace/prerequisites/libbacktrace/install --debug --prefix "$HOME/.local"

# Install Boost.
RUN set -euxo pipefail; \
    python_version="$(python3 -V | grep -Eo '^Python [[:digit:]]+\.[[:digit:]]+' | grep -Eo '[[:digit:]]+\.[[:digit:]]+')"; \
    echo "import toolset : using ; using python : : $(which python3) : /usr/include/python$python_version ;" > "$HOME/user-config.jam"; \
    /workspace/prerequisites/boost/download --debug --source-dir /workspace/boost; \
    /workspace/prerequisites/boost/build --debug --source-dir /workspace/boost --prefix "$HOME/.local" -- \
      -d+2 --with-headers --with-stacktrace --with-python --with-timer --build-type=complete --layout=tagged \
      toolset=gcc variant=release threading=multi link=shared runtime-link=shared; \
    rm -rf /workspace/boost

# Install `marisa-trie`.
RUN set -euxo pipefail; \
    pushd /workspace; \
    git clone 'https://github.com/s-yata/marisa-trie.git'; \
    pushd marisa-trie; \
    autoreconf -i; \
    CFLAGS='-DNDEBUG -O3 -flto' CXXFLAGS='-DNDEBUG -O3 -flto' ./configure --prefix="$HOME/.local" --enable-native-code --disable-static; \
    make -j; \
    make install; \
    popd; \
    rm -rf marisa-trie; \
    popd

# Clone `shanten-number`.
RUN set -euxo pipefail; \
    pushd /workspace; \
    git clone 'https://github.com/tomohxx/shanten-number'; \
    pushd shanten-number; \
    tar xzvf index.tar.gz; \
    popd; \
    popd

COPY --chown=ubuntu . /workspace/kanachan

# Install `kanachan`.
RUN set -euxo pipefail; \
    pushd /workspace/kanachan/src/common; \
    protoc -I. --cpp_out=. mahjongsoul.proto; \
    popd; \
    mkdir -p /workspace/kanachan/build; \
    pushd /workspace/kanachan/build; \
    CC=/home/ubuntu/.local/bin/gcc CXX=/home/ubuntu/.local/bin/g++ cmake \
      -DSHANTEN_NUMBER_SOURCE_PATH=/workspace/shanten-number \
      -DCMAKE_BUILD_TYPE=Release \
      ..; \
    make VERBOSE=1 -j make_trie simulation; \
    mkdir -p /home/ubuntu/.local/share/kanachan; \
    src/xiangting/make_trie /workspace/shanten-number /home/ubuntu/.local/share/kanachan; \
    popd; \
    rm -rf /workspace/shanten-number; \
    . "$HOME/.local/bin/activate"; \
    pushd /workspace/kanachan; \
    cp build/src/simulation/libsimulation.so kanachan/simulation/_simulation.so; \
    python3 -m pip install .; \
    popd; \
    rm -rf /workspace/kanachan/build
