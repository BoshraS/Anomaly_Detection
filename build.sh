#!/bin/bash

cmakeExe=cmake
buildType=Release # Release or Debug
buildDir=build
buildShared=OFF
enableMatplotlib=ON

# Path to CUDA and libtorch
Torch_DIR="/opt/libtorch/share/cmake/Torch"

# Path to nibrary
inc_path="../nibrary/build-static/install/include/nibrary_v0.4.2"
lib_path="../nibrary/build-static/install/lib/nibrary_v0.4.2"

# Path to compiler
c_compiler=/bin/gcc
cxx_compiler=/bin/g++


rm -rf ${buildDir}
mkdir -p ${buildDir}
cd ${buildDir}

${cmakeExe} \
-DCMAKE_C_COMPILER=${c_compiler} \
-DCMAKE_CXX_COMPILER=${cxx_compiler} \
-DCMAKE_BUILD_TYPE=${buildType} \
-DTorch_DIR=${Torch_DIR} \
-DCMAKE_INCLUDE_PATH=${inc_path} \
-DCMAKE_LIBRARY_PATH=${lib_path} \
-DBUILD_SHARED_LIBS=${buildShared} \
-DENABLE_MATPLOTLIB=${enableMatplotlib} \
..

${cmakeExe} --build . --config ${buildType} --target install --parallel 16

cd ..

