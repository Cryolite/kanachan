FROM ubuntu:20.04

ENV LD_LIBRARY_PATH=/usr/local/lib64:/usr/local/lib
ENV PATH=/usr/local/bin${PATH:+:$PATH}

RUN apt-get update && \
    apt-get -y dist-upgrade && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install \
      autoconf \
      automake \
      curl \
      g++ \
      git \
      gnupg \
      libssl-dev \
      libtool \
      locales \
      locales-all \
      m4 \
      make \
      xz-utils \
      zip \
      zlib1g-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    locale-gen en_US.UTF-8 ja_JP.UTF-8 && \
    mkdir -p /opt/src && \
    (cd /opt && git clone 'https://github.com/Cryolite/prerequisites.git') && \
    /opt/prerequisites/gcc/install --debug && \
    /opt/prerequisites/libbacktrace/install --debug && \
    /opt/prerequisites/cmake/install --debug && \
    (cd /opt/src && curl -fLsSo Python-3.10.7.tar.xz 'https://www.python.org/ftp/python/3.10.7/Python-3.10.7.tar.xz') && \
    (cd /opt/src && tar xf Python-3.10.7.tar.xz) && \
    rm /opt/src/Python-3.10.7.tar.xz && \
    (cd /opt/src/Python-3.10.7 && ./configure --enable-optimizations --with-lto) && \
    (cd /opt/src/Python-3.10.7 && make -j) && \
    (cd /opt/src/Python-3.10.7 && make install) && \
    rm -rf /opt/src/Python-3.10.7 && \
    echo 'using python : : /usr/local/bin/python3 ;' >/root/user-config.jam && \
    /opt/prerequisites/boost/download --debug --source-dir /opt/src/boost && \
    /opt/prerequisites/boost/build --debug \
      --source-dir /opt/src/boost -- \
      -d+2 --with-headers --with-stacktrace --with-python --build-type=complete --layout=tagged \
      toolset=gcc variant=release threading=multi link=shared runtime-link=shared && \
    (mkdir -p /opt/src && cd /opt/src && curl -fLsSo protobuf-cpp-3.21.7.tar.gz 'https://github.com/protocolbuffers/protobuf/releases/download/v21.7/protobuf-cpp-3.21.7.tar.gz') && \
    (cd /opt/src && tar xf protobuf-cpp-3.21.7.tar.gz) && \
    rm /opt/src/protobuf-cpp-3.21.7.tar.gz && \
    (cd /opt/src/protobuf-3.21.7 && ./configure) && \
    (cd /opt/src/protobuf-3.21.7 && make -j) && \
    (cd /opt/src/protobuf-3.21.7 && make install) && \
    rm -rf /opt/src/protobuf-3.21.7 && \
    (mkdir -p /opt/src && cd /opt/src && git clone 'https://github.com/s-yata/marisa-trie.git') && \
    (cd /opt/src/marisa-trie && autoreconf -i) && \
    (cd /opt/src/marisa-trie && CFLAGS='-DNDEBUG -O3 -flto' CXXFLAGS='-DNDEBUG -O3 -flto' ./configure --enable-native-code --disable-static) && \
    (cd /opt/src/marisa-trie && make -j) && \
    (cd /opt/src/marisa-trie && make install) && \
    rm -rf /opt/src/marisa-trie && \
    (mkdir -p /opt/src && cd /opt/src && git clone 'https://github.com/tomohxx/shanten-number') && \
    (cd /opt/src/shanten-number && tar xf index.tar.gz) && \
    (mkdir -p /opt/src && cd /opt/src && git clone 'https://github.com/MahjongRepository/mahjong' -b v1.1.11)
