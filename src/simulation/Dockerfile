FROM cryolite/kanachan.prerequisites

ARG CMAKE_BUILD_TYPE=Release

RUN apt-get update && \
    apt-get -y install \
      python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip3 install -U pip

USER ubuntu

ENV PATH="/home/ubuntu/.local/bin:${PATH}"

RUN pip3 install -U \
      mahjong \
      setuptools \
      wheel

RUN pip3 install torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# `marisa-trie` only supports for in-tree build.
RUN pushd /home/ubuntu/.local/src/marisa-trie && \
    autoreconf -i && \
    if [[ $CMAKE_BUILD_TYPE == Debug ]]; then \
      CFLAGS='-g -fsanitize=address -fsanitize=undefined' CXXFLAGS='-D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -g -fsanitize=address -fsanitize=undefined' ./configure --prefix=/home/ubuntu/.local --enable-native-code --disable-static; \
    else \
      CFLAGS='-DNDEBUG -O3 -flto' CXXFLAGS='-DNDEBUG -O3 -flto' ./configure --prefix=/home/ubuntu/.local --enable-native-code --disable-static; \
    fi && \
    make -j && \
    make install && \
    popd

COPY --chown=ubuntu . /opt/kanachan

WORKDIR /opt/kanachan

RUN pushd src/common && \
    protoc -I. --cpp_out=. mahjongsoul.proto && \
    popd && \
    mkdir -p build && \
    pushd build && \
    cmake -DPYTHON_VERSION="$(python3 -V | sed -e 's@^Python[[:space:]]\{1,\}\([[:digit:]]\{1,\}\.[[:digit:]]\{1,\}\)\.[[:digit:]]\{1,\}@\1@')" \
          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
          -DCMAKE_C_COMPILER=/usr/local/bin/gcc \
          -DCMAKE_CXX_COMPILER=/usr/local/bin/g++ \
          .. && \
    VERBOSE=1 make -j make_trie _simulation && \
    mkdir -p /home/ubuntu/.local/share/kanachan && \
    ASAN_OPTIONS=handle_abort=1:allow_addr2line=1 src/xiangting/make_trie && \
    popd && \
    cp build/src/simulation/lib_simulation.so kanachan/simulation/_simulation.so && \
    pip3 install .
