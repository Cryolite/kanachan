#!/usr/bin/env bash

set -euxo pipefail
. prerequisites/prologue.sh

docker build --pull -f mjai.app/Dockerfile -t cryolite/kanachan.mjai-app .

container_id=$(docker run -d --rm cryolite/kanachan.mjai-app sleep infinity)
push_rollback_command "docker stop $container_id"

docker cp "$1" ${container_id}:/opt/kanachan/mjai.app/build/model.kanachan

docker exec -it $container_id bash -c 'cd /opt/kanachan/mjai.app/build; zip -r mjai-app.zip *'

docker cp ${container_id}:/opt/kanachan/mjai.app/build/mjai-app.zip .
