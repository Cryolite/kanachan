#!/usr/bin/env bash

set -euxo pipefail

PS4='+${BASH_SOURCE[0]}:$LINENO: '
if [[ -t 1 ]] && type -t tput >/dev/null; then
  if (( "$(tput colors)" == 256 )); then
    PS4='$(tput setaf 10)'$PS4'$(tput sgr0)'
  else
    PS4='$(tput setaf 2)'$PS4'$(tput sgr0)'
  fi
fi

# Install prerequisite packages.
sudo apt-get -y update
sudo apt-get -y dist-upgrade
sudo apt-get -y install \
  autoconf \
  automake \
  g++ \
  libtool \
  protobuf-compiler \
  python3-dev

sudo chown vscode:vscode /workspaces

# Install CUDA 11.7.1. See https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl.
pushd /workspaces
wget 'https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin'
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget 'https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda-repo-wsl-ubuntu-11-7-local_11.7.1-1_amd64.deb'
sudo dpkg -i cuda-repo-wsl-ubuntu-11-7-local_11.7.1-1_amd64.deb
rm -f cuda-repo-wsl-ubuntu-11-7-local_11.7.1-1_amd64.deb
popd
sudo cp /var/cuda-repo-wsl-ubuntu-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Install prerequisite Python packages.
python3 -m pip install -U pip
python3 -m pip install -U \
  setuptools \
  torch \
  wheel

# Install Apex.
pushd /workspaces
git clone -b '22.08-dev' 'https://github.com/NVIDIA/apex.git'
pushd apex
MAX_JOBS=4 python3 -m pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
popd
rm -rf apex
popd

# Install MTAdam.
pushd /workspaces
git clone 'https://github.com/ItzikMalkiel/MTAdam.git'
pushd MTAdam
mkdir mtadam
cp mtadam.py mtadam
echo 'from .mtadam import MTAdam' > mtadam/__init__.py
echo '''[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "MTAdam"
authors = [
    {"name" = "Itzik Malkiel"},
]
description = "Automatic Balancing of Multiple Training Loss Terms"
readme = "README.txt"
requires-python = ">=3.8"
dependencies = [
    "tensorflow",
]
version = "0.0.1"''' > pyproject.toml
python3 -m pip install .
popd
rm -rf MTAdam
popd

# Clone `prerequisites`.
pushd /workspaces
git clone 'https://github.com/Cryolite/prerequisites'
popd

# Install GCC.
/workspaces/prerequisites/gcc/install --debug --prefix "$HOME/.local"

# Install CMake.
/workspaces/prerequisites/cmake/install --debug --prefix "$HOME/.local"

# Install libbacktrace.
/workspaces/prerequisites/libbacktrace/install --debug --prefix "$HOME/.local"

# Install Boost.Stacktrace and Boost.Python.
echo 'import toolset : using ; using python : : /usr/local/python/current/bin/python3 ;' > "$HOME/user-config.jam"
/workspaces/prerequisites/boost/download --debug --source-dir /workspaces/boost
/workspaces/prerequisites/boost/build --debug --source-dir /workspaces/boost --prefix "$HOME/.local" -- \
  -d+2 --with-headers --with-stacktrace --with-python --build-type=complete --layout=tagged \
  toolset=gcc variant=debug threading=multi link=shared runtime-link=shared \
  cxxflags=-D_GLIBCXX_DEBUG cxxflags=-D_GLIBCXX_DEBUG_PEDANTIC \
  cflags=-fsanitize=address cxxflags=-fsanitize=address linkflags=-fsanitize=address \
  cflags=-fsanitize=undefined cxxflags=-fsanitize=undefined linkflags=-fsanitize=undefined
/workspaces/prerequisites/boost/build --debug --source-dir /workspaces/boost --prefix "$HOME/.local" -- \
  -d+2 --with-headers --with-stacktrace --with-python --build-type=complete --layout=tagged \
  toolset=gcc variant=release threading=multi link=shared runtime-link=shared
rm -rf /workspaces/boost

# Build and install marisa-trie.
pushd /workspaces
git clone 'https://github.com/s-yata/marisa-trie.git'
pushd marisa-trie
autoreconf -i
CFLAGS='-DNDEBUG -O3 -flto' CXXFLAGS='-DNDEBUG -O3 -flto' ./configure --prefix="$HOME/.local" --enable-native-code --disable-static
make -j
make install
popd
rm -rf marisa-trie
popd

# Clone shanten-number.
pushd /workspaces
git clone 'https://github.com/tomohxx/shanten-number'
pushd shanten-number
tar xzvf index.tar.gz
popd
popd

# Compile `mahjongsoul.proto`.
pushd /workspaces/kanachan/src/common
protoc -I. --cpp_out=. mahjongsoul.proto
popd

# Build kanachan.
mkdir -p /workspaces/kanachan/build
pushd /workspaces/kanachan/build
CC="$HOME/.local/bin/gcc" CXX="$HOME/.local/bin/g++" cmake \
  -DSHANTEN_NUMBER_SOURCE_PATH=/workspaces/shanten-number \
  -DCMAKE_BUILD_TYPE=Release \
  ..
VERBOSE=1 make -j make_trie simulation
mkdir -p "$HOME/.local/share/kanachan"
src/xiangting/make_trie /workspaces/shanten-number "$HOME/.local/share/kanachan"
cp src/simulation/libsimulation.so ../kanachan/simulation/_simulation.so
popd

# Install kanachan.
pushd /workspaces/kanachan
python3 -m pip install .
popd
