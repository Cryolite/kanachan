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

rollback_stack=()

function push_rollback_command ()
{
  rollback_stack+=("$1")
}

function rollback ()
{
  for (( i = ${#rollback_stack[@]} - 1; i >= 0; --i )); do
    eval "${rollback_stack[$i]}"
  done
  rollback_stack=()
}

trap rollback EXIT

function print_error_message ()
{
  if [[ -t 2 ]] && type -t tput >/dev/null; then
    if (( "$(tput colors)" == 256 )); then
      echo "$(tput setaf 9)$1$(tput sgr0)" >&2
    else
      echo "$(tput setaf 1)$1$(tput sgr0)" >&2
    fi
  else
    echo "$1" >&2
  fi
}

if [[ $# != 1 ]]; then
  set +x
  print_error_message "Usage: $0 /PATH/TO/VIRTUAL_ENVIRONMENT_DIR"
  exit 1
fi

if [[ $1 == --help ]]; then
  set +x
  echo "Usage: $0 /PATH/TO/VIRTUAL_ENVIRONMENT_DIR"
  exit 0
fi

KANACHAN_ROOT="$(pwd)"

PREFIX="$(readlink -f "$1")"
if [[ -e $PREFIX ]]; then
  set +x
  print_error_message "$PREFIX: Already exists. If it is truly the location you intend, please remove it first."
  exit 1
fi

WORKDIR="$(mktemp -d)"
push_rollback_command "rm -rf '$WORKDIR'"
pushd "$WORKDIR"

# Create Python virtual environment.
python3 -m venv "$PREFIX"
cat > "$PREFIX/bin/activate" <<EOF
# This file must be used with "source bin/activate" *from bash*
# you cannot run it directly

deactivate () {
    # reset old environment variables
    if [ -n "\${_OLD_VIRTUAL_PATH:-}" ] ; then
        PATH="\${_OLD_VIRTUAL_PATH:-}"
        export PATH
        unset _OLD_VIRTUAL_PATH
    fi
    if [ -n "\${_OLD_VIRTUAL_PYTHONHOME:-}" ] ; then
        PYTHONHOME="\${_OLD_VIRTUAL_PYTHONHOME:-}"
        export PYTHONHOME
        unset _OLD_VIRTUAL_PYTHONHOME
    fi
    if [[ -v _OLD_VIRTUAL_LD_LIBRARY_PATH ]]; then
        export LD_LIBRARY_PATH="\$_OLD_VIRTUAL_LD_LIBRARY_PATH"
        unset _OLD_VIRTUAL_LD_LIBRARY_PATH
    elif [[ "\$1" != nondestructive ]]; then
        unset LD_LIBRARY_PATH
    fi
    if [[ -v _OLD_VIRTUAL_LIBRARY_PATH ]]; then
        export LIBRARY_PATH="\$_OLD_VIRTUAL_LIBRARY_PATH"
        unset _OLD_VIRTUAL_LIBRARY_PATH
    elif [[ "\$1" != nondestructive ]]; then
        unset LIBRARY_PATH
    fi
    if [[ -v _OLD_VIRTUAL_CPLUS_INCLUDE_PATH ]]; then
        export CPLUS_INCLUDE_PATH="\$_OLD_VIRTUAL_CPLUS_INCLUDE_PATH"
        unset _OLD_VIRTUAL_CPLUS_INCLUDE_PATH
    elif [[ "\$1" != nondestructive ]]; then
        unset CPLUS_INCLUDE_PATH
    fi
    if [[ -v _OLD_VIRTUAL_C_INCLUDE_PATH ]]; then
        export C_INCLUDE_PATH="\$_OLD_VIRTUAL_C_INCLUDE_PATH"
        unset _OLD_VIRTUAL_C_INCLUDE_PATH
    elif [[ "\$1" != nondestructive ]]; then
        unset C_INCLUDE_PATH
    fi

    # This should detect bash and zsh, which have a hash command that must
    # be called to get it to forget past commands.  Without forgetting
    # past commands the \$PATH changes we made may not be respected
    if [ -n "\${BASH:-}" -o -n "\${ZSH_VERSION:-}" ] ; then
        hash -r 2> /dev/null
    fi

    if [ -n "\${_OLD_VIRTUAL_PS1:-}" ] ; then
        PS1="\${_OLD_VIRTUAL_PS1:-}"
        export PS1
        unset _OLD_VIRTUAL_PS1
    fi

    unset VIRTUAL_ENV
    unset VIRTUAL_ENV_PROMPT
    if [ ! "\${1:-}" = "nondestructive" ] ; then
    # Self destruct!
        unset -f deactivate
    fi
}

# unset irrelevant variables
deactivate nondestructive

VIRTUAL_ENV="$PREFIX"
export VIRTUAL_ENV

_PYTHON_INCLUDE_PATH="\$("\$VIRTUAL_ENV/bin/python3" -m sysconfig | grep -E '^[[:space:]]*platinclude = ' | sed -e 's@^[[:space:]]*platinclude = "\\(.*\\)"@\\1@')"

if [[ -v C_INCLUDE_PATH ]]; then
    _OLD_VIRTUAL_C_INCLUDE_PATH="\$C_INCLUDE_PATH"
fi
export C_INCLUDE_PATH="\$VIRTUAL_ENV/include:\$_PYTHON_INCLUDE_PATH\${C_INCLUDE_PATH:+:\$C_INCLUDE_PATH}"

if [[ -v CPLUS_INCLUDE_PATH ]]; then
    _OLD_VIRTUAL_CPLUS_INCLUDE_PATH="\$CPLUS_INCLUDE_PATH"
fi
export CPLUS_INCLUDE_PATH="\$VIRTUAL_ENV/include:\$_PYTHON_INCLUDE_PATH\${CPLUS_INCLUDE_PATH:+:\$CPLUS_INCLUDE_PATH}"

unset _PYTHON_INCLUDE_PATH

if [[ -v LIBRARY_PATH ]]; then
    _OLD_VIRTUAL_LIBRARY_PATH="\$LIBRARY_PATH"
fi
export LIBRARY_PATH="\$VIRTUAL_ENV/lib64:\$VIRTUAL_ENV/lib\${LIBRARY_PATH:+:\$LIBRARY_PATH}"

if [[ -v LD_LIBRARY_PATH ]]; then
    _OLD_VIRTUAL_LD_LIBRARY_PATH="\$LD_LIBRARY_PATH"
fi
export LD_LIBRARY_PATH="\$VIRTUAL_ENV/lib64:\$VIRTUAL_ENV/lib:/usr/local/cuda-12.1/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"

_OLD_VIRTUAL_PATH="\$PATH"
PATH="\$VIRTUAL_ENV/bin:/usr/local/cuda-12.1/bin\${PATH:+:\$PATH}"
export PATH

# unset PYTHONHOME if set
# this will fail if PYTHONHOME is set to the empty string (which is bad anyway)
# could use `if (set -u; : \$PYTHONHOME) ;` in bash
if [ -n "\${PYTHONHOME:-}" ] ; then
    _OLD_VIRTUAL_PYTHONHOME="\${PYTHONHOME:-}"
    unset PYTHONHOME
fi

if [ -z "\${VIRTUAL_ENV_DISABLE_PROMPT:-}" ] ; then
    _OLD_VIRTUAL_PS1="\${PS1:-}"
    PS1="(.local) \${PS1:-}"
    export PS1
    VIRTUAL_ENV_PROMPT="(.local) "
    export VIRTUAL_ENV_PROMPT
fi

# This should detect bash and zsh, which have a hash command that must
# be called to get it to forget past commands.  Without forgetting
# past commands the \$PATH changes we made may not be respected
if [ -n "\${BASH:-}" -o -n "\${ZSH_VERSION:-}" ] ; then
    hash -r 2> /dev/null
fi
EOF
. "$PREFIX/bin/activate"

# Install prerequisite Python packages.
python3 -m pip install -U pip
python3 -m pip install -U packaging setuptools wheel
python3 -m pip install -U torch torchvision torchaudio --index-url 'https://download.pytorch.org/whl/cu121'

# Install Apex.
git clone 'https://github.com/NVIDIA/apex.git'
pushd apex
MAX_JOBS=4 python3 -m pip install -v --disable-pip-version-check --no-cache-dir \
  --no-build-isolation --config-settings '--build-option=--cpp_ext' \
  --config-settings '--build-option=--cuda_ext' .
popd
rm -rf apex

# Install MTAdam.
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

# Clone `prerequisites`.
git clone 'https://github.com/Cryolite/prerequisites'

# Install GCC.
prerequisites/gcc/install --debug --prefix "$PREFIX"

# Install CMake.
prerequisites/cmake/install --debug --prefix "$PREFIX"

# Install libbacktrace.
prerequisites/libbacktrace/install --debug --prefix "$PREFIX"

# Install Boost.Stacktrace and Boost.Python.
if [[ -e "$HOME/user-config.jam" ]]; then
  mv "$HOME/user-config.jam" "$HOME/user-config.jam.old"
  push_rollback_command 'mv -f "$HOME/user-config.jam.old" "$HOME/user-config.jam"'
fi
echo "import toolset : using ; using python : : $PREFIX/bin/python3 ;" > "$HOME/user-config.jam"
prerequisites/boost/download --debug --source-dir "$WORKDIR/boost"
prerequisites/boost/build --debug --source-dir "$WORKDIR/boost" --prefix "$PREFIX" -- \
  -d+2 --with-headers --with-stacktrace --with-python --build-type=complete --layout=tagged \
  toolset=gcc variant=debug threading=multi link=shared runtime-link=shared \
  cxxflags=-D_GLIBCXX_DEBUG cxxflags=-D_GLIBCXX_DEBUG_PEDANTIC \
  cflags=-fsanitize=address cxxflags=-fsanitize=address linkflags=-fsanitize=address \
  cflags=-fsanitize=undefined cxxflags=-fsanitize=undefined linkflags=-fsanitize=undefined
prerequisites/boost/build --debug --source-dir "$WORKDIR/boost" --prefix "$PREFIX" -- \
  -d+2 --with-headers --with-stacktrace --with-python --build-type=complete --layout=tagged \
  toolset=gcc variant=release threading=multi link=shared runtime-link=shared
rm -rf "$WORKDIR/boost"

# Build and install marisa-trie.
git clone 'https://github.com/s-yata/marisa-trie.git'
pushd marisa-trie
autoreconf -i
CFLAGS='-DNDEBUG -O3 -flto' CXXFLAGS='-DNDEBUG -O3 -flto' ./configure --prefix="$PREFIX" --enable-native-code --disable-static
make -j
make install
popd
rm -rf marisa-trie

# Clone shanten-number.
git clone 'https://github.com/tomohxx/shanten-number'
pushd shanten-number
tar xzvf index.tar.gz
popd

# Compile `mahjongsoul.proto`.
pushd "$KANACHAN_ROOT/src/common"
protoc -I. --cpp_out=. mahjongsoul.proto
popd

# Build kanachan.
mkdir build
pushd build
CC="$PREFIX/bin/gcc" CXX="$PREFIX/bin/g++" \
  cmake -DSHANTEN_NUMBER_SOURCE_PATH="$WORKDIR/shanten-number" -DCMAKE_BUILD_TYPE=Release \
  "$KANACHAN_ROOT"
VERBOSE=1 make -j make_trie simulation
mkdir -p "$PREFIX/share/kanachan"
src/xiangting/make_trie "$WORKDIR/shanten-number" "$PREFIX/share/kanachan"
cp src/simulation/libsimulation.so ../kanachan/simulation/_simulation.so
popd

# Install kanachan.
pushd "$KANACHAN_ROOT"
python3 -m pip install .
popd

popd
