#!/bin/bash
rm lane-seg
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .

mv lane-seg ..