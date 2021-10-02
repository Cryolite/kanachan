FROM ubuntu:latest

SHELL ["/bin/bash", "-c"]

RUN apt-get update && \
    apt-get -y dist-upgrade && \
    apt-get -y install \
      curl \
      g++ \
      git \
      gpg \
      libgmp-dev \
      libssl-dev \
      make \
      protobuf-compiler \
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

RUN /opt/kanachan-prerequisites/boost/download --debug --source-dir /usr/local/src/boost
RUN /opt/kanachan-prerequisites/boost/build --debug --source-dir /usr/local/src/boost -- \
  -d+2 --with-stacktrace --build-type=complete --layout=tagged \
  toolset=gcc variant=debug threading=multi link=shared runtime-link=shared \
  cxxflags=-D_GLIBCXX_DEBUG cxxflags=-D_GLIBCXX_DEBUG_PEDANTIC \
  cflags=-fsanitize=address cxxflags=-fsanitize=address linkflags=-fsanitize=address \
  cflags=-fsanitize=undefined cxxflags=-fsanitize=undefined linkflags=-fsanitize=undefined
RUN /opt/kanachan-prerequisites/boost/build --debug --source-dir /usr/local/src/boost -- \
  -d+2 --with-stacktrace --build-type=complete --layout=tagged \
  toolset=gcc variant=release threading=multi link=shared runtime-link=shared
