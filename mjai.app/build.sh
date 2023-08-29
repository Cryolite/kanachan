#!/usr/bin/env bash

set -euxo pipefail
. prerequisites/prologue.sh

docker build --pull -f mjai.app/Dockerfile -t cryolite/kanachan.mjai-app .

container_id=$(docker run -d --rm cryolite/kanachan.mjai-app sleep infinity)
push_rollback_command "docker stop $container_id"

docker exec -it $container_id mkdir -p /opt/kanachan
docker cp . ${container_id}:/opt/kanachan
docker exec -it $container_id rm -rf /opt/kanachan/build

docker exec -it $container_id bash -c 'cd /opt/kanachan/src/common && protoc -I. --cpp_out=. mahjongsoul.proto'

docker exec -it $container_id mkdir -p /opt/kanachan/build
docker exec -it $container_id bash -c \
  'cd /opt/kanachan/build && \
   cmake \
     -DPYTHON_VERSION="$(python3 -V | sed -e '\''s@^Python[[:space:]]\{1,\}\([[:digit:]]\{1,\}\.[[:digit:]]\{1,\}\)\.[[:digit:]]\{1,\}@\1@'\'')" \
     -DPYTHON_INCLUDE_PATH=/usr/local/include/python"$(python3 -V | sed -e '\''s@^Python[[:space:]]\{1,\}\([[:digit:]]\{1,\}\.[[:digit:]]\{1,\}\)\.[[:digit:]]\{1,\}@\1@'\'')" \
     -DMARISA_TRIE_ROOT=/usr/local \
     -DSHANTEN_NUMBER_SOURCE_PATH=/opt/src/shanten-number \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_BUILD_RPATH=/workspace/.local/lib \
     ..'
docker exec -it $container_id bash -c 'cd /opt/kanachan/build && VERBOSE=1 make -j make_trie xiangting_calculator'
docker exec -it $container_id /opt/kanachan/build/src/xiangting/make_trie /opt/src/shanten-number /opt/kanachan/mjai.app

docker exec -it $container_id mkdir -p /opt/kanachan/mjai.app/.local/lib
docker exec -it $container_id bash -c 'cp -f /usr/local/lib/libboost_stacktrace_backtrace* /opt/kanachan/mjai.app/.local/lib'
docker exec -it $container_id bash -c 'cp -f /usr/local/lib/libboost_python* /opt/kanachan/mjai.app/.local/lib'
docker exec -it $container_id bash -c 'cp -f /usr/local/lib/libmarisa.so* /opt/kanachan/mjai.app/.local/lib'
docker exec -it $container_id cp -rf /opt/src/mahjong/mahjong /opt/kanachan/mjai.app
docker exec -it $container_id bash -c 'cd /opt/kanachan && cp -f build/src/simulation/libxiangting_calculator.so mjai.app/xiangting_calculator/_xiangting_calculator.so'
docker exec -it $container_id cp -rf /opt/kanachan/kanachan /opt/kanachan/mjai.app

docker cp "$1" ${container_id}:/opt/kanachan/mjai.app/model.kanachan

docker exec -it $container_id bash -c 'cd /opt/kanachan/mjai.app && cp -f config.json.orig config.json'

docker exec -it $container_id bash -c 'cd /opt/kanachan/mjai.app && \
  zip -r mjai-app.zip \
         .local \
         _kanachan.py \
         bot.py \
         config.json \
         hand_calculator.py \
         kanachan \
         mahjong \
         model.kanachan \
         shupai.trie \
         shupai.xiangting \
         xiangting_calculator \
         zipai.trie \
         zipai.xiangting'

docker cp ${container_id}:/opt/kanachan/mjai.app/mjai-app.zip .
