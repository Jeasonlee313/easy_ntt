#!/bin/bash
#set -e

PROJECT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/..";pwd)
PROJECT_CODE_PATH="${PROJECT_PATH}"

LINT_PATH="
include
src
test
exec
"

EXCLUDE_PATH="
${PROJECT_CODE_PATH}/include/driver/cuda_macros.h
"
EXCLUDE_ARGS=""
for ep in `echo "$EXCLUDE_PATH"`; do
  EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude=$ep"
done
echo "EXCLUDE_ARGS: $EXCLUDE_ARGS"

function lint_all() {
  for i in `echo "$LINT_PATH"`
  do
#    echo "*******lint $i*********"
    lint_single $i
  done
}

function lint_single() {
    echo "*******lint $1*********"
    [ -d $1 ] || {
      echo "cannot access '$1': No such directory"
      exit 1
    }
    result=`find $1 -name "*.h" -o -name "*.cc" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" | xargs cpplint $EXCLUDE_ARGS`
    set +e
    echo "$result"|grep -v "Done processing"
    set -e
}

if [ $# -ge 1 ]; then
  lint_single "${PROJECT_CODE_PATH}/$1"
else
  echo "*********start lint all*********"
  lint_all
fi

[ $? == 0 ] && echo "*******SUCCESS*********"
