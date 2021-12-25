FROM cryolite/kanachan.prerequisites

ARG CMAKE_BUILD_TYPE=Release

COPY --chown=ubuntu . /opt/kanachan

USER ubuntu

WORKDIR /opt/kanachan

RUN cp /opt/kanachan-prerequisites/mahjongsoul.pb.{h,cc} src/common && \
    mkdir build && \
    pushd build && \
    cmake -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
          -DCMAKE_C_COMPILER=/usr/local/bin/gcc \
          -DCMAKE_CXX_COMPILER=/usr/local/bin/g++ \
          .. && \
    VERBOSE=1 make -j annotate && \
    popd

ENTRYPOINT ["build/src/annotation/annotate", "/data"]