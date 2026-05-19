# Standalone C++ example: measure ONNX saving time (with external data)
# using the onnx_light C++ API.
#
# Prerequisites
# -------------
# First install onnx_light as a C++ library.  From the onnx-light repository
# root run:
#
#   cmake -S . -B build-install \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DONNX_LIGHT_BUILD_PYTHON=OFF \
#         -DCMAKE_INSTALL_PREFIX=/usr/local   # or any prefix you prefer
#   cmake --build  build-install
#   cmake --install build-install
#
# Build this example
# ------------------
# Point CMAKE_PREFIX_PATH at the install prefix chosen above:
#
#   cmake -S examples/save_onnx_light_time -B build-save-onnx-light-time \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_PREFIX_PATH=/usr/local
#   cmake --build build-save-onnx-light-time
#
# Run
# ---
#   ./build-save-onnx-light-time/save_onnx_light_time path/to/model.onnx /tmp/out 10 4

cmake_minimum_required(VERSION 3.15)
project(save_onnx_light_time LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnx_light REQUIRED)

add_executable(save_onnx_light_time main.cpp)
target_link_libraries(save_onnx_light_time PRIVATE onnx_light::lib_onnx_proto)
