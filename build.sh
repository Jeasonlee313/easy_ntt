#!/bin/bash

set -e

rm -rf build install
mkdir -p build && cd build
cmake .. --install-prefix $(pwd)/../install/easy_ntt 2>&1 | tee compile.log
make 2>&1 | tee -a compile.log
ctest --output-on-failure 2>&1 | tee -a compile.log
make install 2>&1 | tee -a compile.log
