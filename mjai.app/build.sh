#!/usr/bin/env bash

set -euxo pipefail
. prerequisites/prologue.sh

docker build --pull -f mjai.app/Dockerfile -t cryolite/kanachan.mjai-app .

container_id=$(docker run -d --rm cryolite/kanachan.mjai-app sleep infinity)
push_rollback_command "docker stop $container_id"

docker exec $container_id mkdir -p /opt/kanachan
docker cp . ${container_id}:/opt/kanachan

docker exec $container_id /opt/kanachan/prerequisites/boost/download --debug --source-dir /opt/src/boost
docker exec $container_id /opt/kanachan/prerequisites/boost/build --debug \
  --source-dir /opt/src/boost -- \
  -d+2 --with-headers --with-stacktrace --with-python --build-type=complete --layout=tagged \
  toolset=gcc variant=release threading=multi link=shared runtime-link=shared

docker exec $container_id bash -c 'cd /opt/kanachan/src/common && protoc -I. --cpp_out=. mahjongsoul.proto'

docker exec $container_id mkdir -p /opt/kanachan/build
docker exec $container_id bash -c \
  'cd /opt/kanachan/build && \
   cmake \
     -DPYTHON_VERSION="$(python3 -V | sed -e '\''s@^Python[[:space:]]\{1,\}\([[:digit:]]\{1,\}\.[[:digit:]]\{1,\}\)\.[[:digit:]]\{1,\}@\1@'\'')" \
     -DPYTHON_INCLUDE_PATH=/usr/local/include/python"$(python3 -V | sed -e '\''s@^Python[[:space:]]\{1,\}\([[:digit:]]\{1,\}\.[[:digit:]]\{1,\}\)\.[[:digit:]]\{1,\}@\1@'\'')" \
     -DMARISA_TRIE_ROOT=/usr/local \
     -DSHANTEN_NUMBER_SOURCE_PATH=/opt/src/shanten-number \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_BUILD_RPATH=/workspace/.local/lib \
     ..'
docker exec $container_id bash -c 'cd /opt/kanachan/build && VERBOSE=1 make -j make_trie _simulation'
docker exec $container_id /opt/kanachan/build/src/xiangting/make_trie /opt/src/shanten-number /opt/kanachan/mjai.app

docker exec $container_id mkdir -p /opt/kanachan/mjai.app/.local/lib
docker exec $container_id bash -c 'cp -f /usr/local/lib/libboost_stacktrace_backtrace* /opt/kanachan/mjai.app/.local/lib'
docker exec $container_id bash -c 'cp -f /usr/local/lib/libboost_python* /opt/kanachan/mjai.app/.local/lib'
docker exec $container_id bash -c 'cp -f /usr/local/lib/libmarisa.so* /opt/kanachan/mjai.app/.local/lib'
docker exec $container_id cp -rf /opt/src/mahjong/mahjong /opt/kanachan/mjai.app
docker exec $container_id bash -c 'cd /opt/kanachan && cp -f build/src/simulation/lib_simulation.so mjai.app/xiangting_calculator/_xiangting_calculator.so'
docker exec $container_id cp -rf /opt/kanachan/kanachan /opt/kanachan/mjai.app

docker cp "$2" ${container_id}:/opt/kanachan/mjai.app/encoder.pth
docker cp "$3" ${container_id}:/opt/kanachan/mjai.app/decoder.pth

docker exec $container_id bash -c 'cd /opt/kanachan/mjai.app && cp -f game.json.orig game.json'

docker exec $container_id bash -c 'cd /opt/kanachan/mjai.app && \
  zip -r mjai-app.zip \
         .local \
         _kanachan.py \
         bot.py \
         decoder.json \
         decoder.pth \
         encoder.json \
         encoder.pth \
         game.json \
         hand_calculator.py \
         kanachan \
         mahjong \
         shupai.trie \
         shupai.xiangting \
         xiangting_calculator \
         zipai.trie \
         zipai.xiangting'

docker cp ${container_id}:/opt/kanachan/mjai.app/mjai-app.zip .
