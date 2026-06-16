#!/bin/bash
#set -e

PROJECT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/..";pwd)
PROJECT_CODE_PATH="${PROJECT_PATH}"

FORMAT_PATH="
include
src
test
exec
"

function format_all() {
  for i in `echo "$FORMAT_PATH"`
  do
    format_single $i
  done
}

function format_single() {
    echo "*******format $1*********"
    [ -d $1 ] || {
      echo "cannot access '$1': No such directory"
      exit 1
    }
    set +e
    find $1 -name "*.h" -o -name "*.cc" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" | xargs clang-format -i 2>&1
    set -e
}

if [ $# -ge 1 ]; then
  format_single "${PROJECT_CODE_PATH}/$1"
else
  echo "*********start format all*********"
  format_all
fi

[ $? == 0 ] && echo "*******SUCCESS*********"
