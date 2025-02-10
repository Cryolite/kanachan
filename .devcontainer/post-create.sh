#!/usr/bin/env bash

set -euxo pipefail

PYTHON_VERSION=3.12.8

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
  libbz2-dev \
  libffi-dev \
  libgdbm-compat-dev \
  libgdbm-dev \
  liblzma-dev \
  libncurses-dev \
  libreadline-dev \
  libsqlite3-dev \
  libtool \
  protobuf-compiler \
  tk-dev \
  uuid-dev

sudo chown vscode:vscode /workspaces

# Install CUDA 12.4.
pushd /workspaces
wget 'https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb'
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm -f cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
popd

# Install `pyenv`.
curl 'https://pyenv.run' | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$HOME/.bashrc"
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> "$HOME/.profile"
export PYENV_ROOT="$HOME/.pyenv"
echo 'export PATH="$PYENV_ROOT/bin${PATH:+:$PATH}"' >> "$HOME/.bashrc"
echo 'export PATH="$PYENV_ROOT/bin${PATH:+:$PATH}"' >> "$HOME/.profile"
export PATH="$PYENV_ROOT/bin${PATH:+:$PATH}"

# Create a Python environment.
if [[ $PYTHON_VERSION == latest ]]; then
  PYTHON_LATEST_VERSION=$(pyenv install -l | grep -Eo '^[[:space:]]+[[:digit:]]+\.[[:digit:]]+\.[[:digit:]]+$' | grep -Eo '[[:digit:]]+\.[[:digit:]]+\.[[:digit:]]+' | LANG=C.UTF-8 sort -V | tail -n 1)
  PYTHON_VERSION=$PYTHON_LATEST_VERSION
fi
PYTHON_VERSION_MAJOR=$(echo $PYTHON_VERSION | grep -Eo '^[[:digit:]]+\.[[:digit:]]+')
pyenv install $PYTHON_VERSION
pyenv global $PYTHON_VERSION
pushd "$HOME/.pyenv/versions/$PYTHON_VERSION/lib"
ln -s libpython$PYTHON_VERSION_MAJOR.so libpython.so
popd
echo 'eval "$(pyenv init -)"' >> "$HOME/.bashrc"
echo 'eval "$(pyenv init -)"' >> "$HOME/.profile"
eval "$(pyenv init -)"

# Install prerequisite Python packages.
python3 -m pip install -U pip
python3 -m pip install -U \
  build \
  ninja \
  packaging \
  setuptools \
  wheel

# Install PyTorch.
python3 -m pip install -U torch --index-url 'https://download.pytorch.org/whl/cu124'

# Install Apex.
pushd /workspaces
git clone 'https://github.com/NVIDIA/apex.git'
pushd apex
MAX_JOBS=4 python3 -m pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings '--build-option=--cpp_ext' --config-settings '--build-option=--cuda_ext' ./
popd
rm -rf apex
popd

# Clone `prerequisites`.
pushd /workspaces
git clone 'https://github.com/Cryolite/prerequisites.git'
popd

# Install GCC.
/workspaces/prerequisites/gcc/install --debug --prefix "$HOME/.local"
echo 'export C_INCLUDE_PATH="$HOME/.local/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"' >> "$HOME/.bashrc"
echo 'export C_INCLUDE_PATH="$HOME/.local/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"' >> "$HOME/.profile"
export C_INCLUDE_PATH="$HOME/.local/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
echo 'export CPLUS_INCLUDE_PATH="$HOME/.local/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"' >> "$HOME/.bashrc"
echo 'export CPLUS_INCLUDE_PATH="$HOME/.local/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"' >> "$HOME/.profile"
export CPLUS_INCLUDE_PATH="$HOME/.local/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
echo 'export LIBRARY_PATH="$HOME/.local/lib64:$HOME/.local/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"' >> "$HOME/.bashrc"
echo 'export LIBRARY_PATH="$HOME/.local/lib64:$HOME/.local/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"' >> "$HOME/.profile"
export LIBRARY_PATH="$HOME/.local/lib64:$HOME/.local/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
echo 'export LD_LIBRARY_PATH="$HOME/.local/lib64:$HOME/.local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' >> "$HOME/.bashrc"
echo 'export LD_LIBRARY_PATH="$HOME/.local/lib64:$HOME/.local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' >> "$HOME/.profile"
export LD_LIBRARY_PATH="$HOME/.local/lib64:$HOME/.local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo 'export PATH="$HOME/.local/bin${PATH:+:$PATH}"' >> "$HOME/.bashrc"
echo 'export PATH="$HOME/.local/bin${PATH:+:$PATH}"' >> "$HOME/.profile"
export PATH="$HOME/.local/bin${PATH:+:$PATH}"
echo 'export CC="$HOME/.local/bin/gcc"' >> "$HOME/.bashrc"
echo 'export CC="$HOME/.local/bin/gcc"' >> "$HOME/.profile"
export CC="$HOME/.local/bin/gcc"
echo 'export CXX="$HOME/.local/bin/g++"' >> "$HOME/.bashrc"
echo 'export CXX="$HOME/.local/bin/g++"' >> "$HOME/.profile"
export CXX="$HOME/.local/bin/g++"

# Install CMake.
/workspaces/prerequisites/cmake/install --debug --prefix "$HOME/.local"

# Install libbacktrace.
/workspaces/prerequisites/libbacktrace/install --debug --prefix "$HOME/.local"

# Install Boost.Stacktrace, Boost.Python, and Boost.System.
# (Boost.System is required by akochan.)
PYTHON_PREFIX="$(python3 -c 'import sys; print(sys.prefix);')"
echo "import toolset : using ; using python : : \"$PYTHON_PREFIX/bin/python3\" ;" > "$HOME/user-config.jam"
/workspaces/prerequisites/boost/download --debug --source-dir /workspaces/boost
/workspaces/prerequisites/boost/build --debug --source-dir /workspaces/boost --prefix "$HOME/.local" -- \
  -d+2 --with-headers --with-stacktrace --with-python --with-system --build-type=complete --layout=tagged \
  toolset=gcc variant=debug threading=multi link=shared runtime-link=shared \
  cxxflags=-D_GLIBCXX_DEBUG cxxflags=-D_GLIBCXX_DEBUG_PEDANTIC \
  cflags=-fsanitize=address cxxflags=-fsanitize=address linkflags=-fsanitize=address \
  cflags=-fsanitize=undefined cxxflags=-fsanitize=undefined linkflags=-fsanitize=undefined
/workspaces/prerequisites/boost/build --debug --source-dir /workspaces/boost --prefix "$HOME/.local" -- \
  -d+2 --with-headers --with-stacktrace --with-python --with-system --build-type=complete --layout=tagged \
  toolset=gcc variant=release threading=multi link=shared runtime-link=shared
rm -rf /workspaces/boost
pushd "$HOME/.local/lib"
ln -s libboost_system-mt-x64.so libboost_system.so
popd

# Install `Nyanten`.
pushd /workspaces
git clone 'https://github.com/Cryolite/nyanten.git'
cmake -S nyanten -B nyanten/build
cmake --install nyanten/build --prefix /home/vscode/.local
rm -rf nyanten
popd

# Install akochan.
pushd /workspaces
git clone https://github.com/critter-mj/akochan.git
pushd akochan
pushd ai_src
make -f Makefile_Linux
popd
sed -i -e 's/boost::asio::io_service/boost::asio::io_context/' mjai_client.hpp
sed -i -e 's/boost::asio::ip::address::from_string/boost::asio::ip::make_address/' mjai_client.cpp
sed -i -e 's/boost::asio::buffer_cast<const char \*>(buffer\.data())/static_cast<const char *>(buffer.data().data())/' mjai_client.cpp
make -f Makefile_Linux
cp libai.so /home/vscode/.local/lib64
cp system.exe /home/vscode/.local/bin/akochan
popd
popd

# Compile `mahjongsoul.proto`.
pushd /workspaces/kanachan/src/common
protoc -I. --cpp_out=. mahjongsoul.proto
popd

# Build Kanachan.
rm -rf /workspaces/kanachan/build
pushd /workspaces/kanachan
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
VERBOSE=1 cmake --build build --parallel $(nproc) --target simulation
cp build/src/simulation/libsimulation.so kanachan/simulation/_simulation.so
popd

# Install Kanachan.
rm -rf /workspaces/kanachan/dist
pushd /workspaces/kanachan
python3 -m build -w
python3 -m pip install -U dist/*.whl
popd
