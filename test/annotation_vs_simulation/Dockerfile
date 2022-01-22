FROM cryolite/kanachan.prerequisites

ARG CMAKE_BUILD_TYPE=Release

COPY --chown=ubuntu . /opt/kanachan

USER ubuntu

WORKDIR /opt/kanachan

RUN pushd src/common && \
    protoc -I. --cpp_out=. mahjongsoul.proto && \
    popd && \
    mkdir build && \
    pushd build && \
    cmake -DPYTHON_VERSION="$(python3 -V | sed -e 's@^Python[[:space:]]\{1,\}\([[:digit:]]\{1,\}\.[[:digit:]]\{1,\}\)\.[[:digit:]]\{1,\}@\1@')" \
          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
          -DCMAKE_C_COMPILER=/usr/local/bin/gcc \
          -DCMAKE_CXX_COMPILER=/usr/local/bin/g++ \
          .. && \
    VERBOSE=1 make -j generate && \
    popd

ENTRYPOINT ["test/annotation_vs_simulation/generate.sh"]
