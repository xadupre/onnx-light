# Standalone C++ example: measure ONNX loading time with the onnx_light C++ API.
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
#   cmake -S examples/load_onnx_time -B build-load-onnx-time \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_PREFIX_PATH=/usr/local
#   cmake --build build-load-onnx-time
#
# Run
# ---
#   ./build-load-onnx-time/load_onnx_time path/to/model.onnx 10

cmake_minimum_required(VERSION 3.15)
project(load_onnx_time LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnx REQUIRED)

add_executable(load_onnx_time main.cpp)
target_link_libraries(load_onnx_time PRIVATE onnx::onnx)
