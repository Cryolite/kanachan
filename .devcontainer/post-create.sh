#!/usr/bin/env bash

set -euxo pipefail

# Install prerequisite packages.
sudo apt-get -y update
sudo apt-get -y dist-upgrade
sudo apt-get -y install \
  autoconf \
  automake \
  cmake \
  g++ \
  libtool \
  protobuf-compiler \
  python3-dev

# Install CUDA 11.7.1. See https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl.
pushd "$HOME"
wget 'https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin'
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget 'https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda-repo-wsl-ubuntu-11-7-local_11.7.1-1_amd64.deb'
sudo dpkg -i cuda-repo-wsl-ubuntu-11-7-local_11.7.1-1_amd64.deb
rm -f cuda-repo-wsl-ubuntu-11-7-local_11.7.1-1_amd64.deb
popd
sudo cp /var/cuda-repo-wsl-ubuntu-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Install Boost.Stacktrace and Boost.Python.
echo 'import toolset : using ; using python : : /usr/local/python/current/bin/python3 ;' > "$HOME/user-config.jam"
/workspaces/kanachan/prerequisites/boost/download --debug --source-dir "$HOME/.local/src/boost"
/workspaces/kanachan/prerequisites/boost/build --debug --source-dir "$HOME/.local/src/boost" --prefix "$HOME/.local" -- \
  -d+2 --with-headers --with-stacktrace --with-python --build-type=complete --layout=tagged \
  toolset=gcc variant=debug threading=multi link=shared runtime-link=shared \
  cxxflags=-D_GLIBCXX_DEBUG cxxflags=-D_GLIBCXX_DEBUG_PEDANTIC \
  cflags=-fsanitize=address cxxflags=-fsanitize=address linkflags=-fsanitize=address \
  cflags=-fsanitize=undefined cxxflags=-fsanitize=undefined linkflags=-fsanitize=undefined
/workspaces/kanachan/prerequisites/boost/build --debug --source-dir "$HOME/.local/src/boost" --prefix "$HOME/.local" -- \
  -d+2 --with-headers --with-stacktrace --with-python --build-type=complete --layout=tagged \
  toolset=gcc variant=release threading=multi link=shared runtime-link=shared

# Build and install marisa-trie.
mkdir -p "$HOME/.local/src"
pushd "$HOME/.local/src"
git clone 'https://github.com/s-yata/marisa-trie.git'
popd
pushd "$HOME/.local/src/marisa-trie"
autoreconf -i
CFLAGS='-DNDEBUG -O3 -flto' CXXFLAGS='-DNDEBUG -O3 -flto' ./configure --prefix="$HOME/.local" --enable-native-code --disable-static
make -j
make install
popd
rm -rf "$HOME/.local/src/marisa-trie"

# Install PyTorch and other prerequisite Python packages.
python3 -m pip install -U pip
python3 -m pip install -U \
  setuptools \
  wheel \
  packaging \
  pyyaml \
  jsonschema \
  mahjong==1.1.11 \
  torch

# Install Apex.
pushd "$HOME/.local/src"
git clone 'https://github.com/NVIDIA/apex.git'
pushd apex
MAX_JOBS=$(nproc) python3 -m pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
popd
rm -rf apex
popd

# Compile `mahjongsoul.proto`.
pushd /workspaces/kanachan/src/common
protoc -I. --cpp_out=. mahjongsoul.proto
popd
