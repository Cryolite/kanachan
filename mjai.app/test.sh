#/usr/bin/env bash

set -euxo pipefail
. prerequisites/prologue.sh

archive_path="${1:-mjai-app.zip}"

tempdir="$(mktemp -d)"
push_rollback_command "rm -rf '$tempdir'"

pushd "$tempdir"

python3 -m venv .local
. .local/bin/activate
push_rollback_command 'deactivate'
python3 -m pip install -U pip

git clone 'https://github.com/smly/mjai.app.git'
pushd mjai.app
python3 -m pip install -U .
popd

popd

while true; do
  logs_dir="./logs.$(date +%Y-%m-%d-%H-%M-%S)"

  pushd "$tempdir"

  cat > test.py <<EOF
#/usr/bin/env python3

import random
import sys
from mjai import Simulator

submissions = [
    "$archive_path",
    "$archive_path",
    "$archive_path",
    "$archive_path",
]
Simulator(submissions, logs_dir="$logs_dir", seed=(random.randint(0, sys.maxsize), random.randint(0, sys.maxsize)), timeout=10).run()
EOF
  chmod 755 test.py

  popd

  python3 "$tempdir/test.py" || true
  grep -Fqr '"error"' "$logs_dir" || rm -rf "$logs_dir"
done
