#!/usr/bin/env bash

set -euo pipefail

PROGRAM_NAME=generate.sh

function print_usage ()
{
  cat >&2 <<'EOF'
Usage: generate.sh PAISHAN_FILE ANNOTATION_FILE
Output the tarball binary consisting of test case files to the standard output.
EOF
}

if (( $# < 2 )); then
  echo 'Too few arguments' >&2
  print_usage
  exit 1
fi
if (( $# > 2 )); then
  echo 'Too many arguments' >&2
  print_usage
  exit 1
fi

temp_dir="$(mktemp -d)"
trap "rm -rf '$temp_dir'" EXIT

build/test/annotation_vs_simulation/generate "$1" "$2" "$temp_dir" 1>&2

cd "$temp_dir"
find . -name '*.json' -type f > list.txt
tar -cT list.txt
