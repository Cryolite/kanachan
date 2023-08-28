#/usr/bin/env bash

set -euxo pipefail
. prerequisites/prologue.sh

tempdir="$(mktemp -d)"
push_rollback_command "rm -rf '$tempdir'"

pushd $tempdir

python3 -m venv .local
. .local/bin/activate
push_rollback_command 'deactivate'
python3 -m pip install -U pip

git clone 'https://github.com/smly/mjai.app.git'
pushd mjai.app
python3 -m pip install -U .
popd

cat > test.py <<'EOF'
#/usr/bin/env python3

import random
import sys
from mjai import Simulator

submissions = [
    "mjai-app.zip",
    "mjai-app.zip",
    "mjai-app.zip",
    "mjai-app.zip",
]
Simulator(submissions, logs_dir="./logs", seed=(random.randint(0, sys.maxsize), random.randint(0, sys.maxsize)), timeout=10).run()
EOF
chmod 755 "$tempdir/test.py"

popd

python3 "$tempdir/test.py"
