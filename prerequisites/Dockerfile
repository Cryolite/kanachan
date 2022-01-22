FROM ubuntu:latest

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get -y dist-upgrade && \
    apt-get -y install \
      autoconf \
      curl \
      g++ \
      git \
      gpg \
      libgmp-dev \
      libtool \
      libssl-dev \
      make \
      protobuf-compiler \
      python3-dev \
      xz-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    useradd -ms /bin/bash ubuntu && \
    mkdir /opt/kanachan-prerequisites && \
    chown ubuntu:ubuntu /opt/kanachan-prerequisites

COPY --chown=ubuntu . /opt/kanachan-prerequisites

RUN /opt/kanachan-prerequisites/gcc/install --debug

ENV LD_LIBRARY_PATH=/usr/local/lib64:/usr/local/lib

RUN /opt/kanachan-prerequisites/libbacktrace/install --debug

RUN /opt/kanachan-prerequisites/cmake/install --debug

RUN /opt/kanachan-prerequisites/googletest/install --debug

RUN echo 'import toolset : using ; using python : : /usr/bin/python3 ;' >/root/user-config.jam
RUN /opt/kanachan-prerequisites/boost/download --debug --source-dir /usr/local/src/boost
RUN /opt/kanachan-prerequisites/boost/build --debug --source-dir /usr/local/src/boost -- \
  -d+2 --with-headers --with-stacktrace --with-python --build-type=complete --layout=tagged \
  toolset=gcc variant=debug threading=multi link=shared runtime-link=shared \
  cxxflags=-D_GLIBCXX_DEBUG cxxflags=-D_GLIBCXX_DEBUG_PEDANTIC \
  cflags=-fsanitize=address cxxflags=-fsanitize=address linkflags=-fsanitize=address \
  cflags=-fsanitize=undefined cxxflags=-fsanitize=undefined linkflags=-fsanitize=undefined
RUN /opt/kanachan-prerequisites/boost/build --debug --source-dir /usr/local/src/boost -- \
  -d+2 --with-headers --with-stacktrace --with-python --build-type=complete --layout=tagged \
  toolset=gcc variant=release threading=multi link=shared runtime-link=shared
RUN find /usr -name 'libboost_stacktrace_*.so.*' | xargs -I '{}' cp '{}' /usr/local/lib
RUN cd /usr/local/lib && find -maxdepth 1 -name 'libboost_stacktrace_*' | xargs -I '{}' bash -c 'ln -s "$(basename {})" "$(basename {} | sed -e "s@\(\.[[:digit:]]\{1,\}\)\{3\}\$@@")"'

USER ubuntu

RUN mkdir -p /home/ubuntu/.local/src && \
    pushd /home/ubuntu/.local/src && \
    git clone 'https://github.com/s-yata/marisa-trie.git' && \
    popd

RUN mkdir -p /home/ubuntu/.local/src && \
    pushd /home/ubuntu/.local/src && \
    git clone 'https://github.com/tomohxx/shanten-number' && \
    pushd shanten-number && \
    tar xzvf index.tar.gz && \
    popd && \
    popd

USER root
