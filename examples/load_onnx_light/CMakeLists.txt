# Standalone C++ example: load an ONNX file with the onnx_light C++ API.
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
#   cmake -S examples/load_onnx_light -B build-load-onnx-light \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_PREFIX_PATH=/usr/local
#   cmake --build build-load-onnx-light
#
# Run
# ---
#   ./build-load-onnx-light/load_onnx_light path/to/model.onnx

cmake_minimum_required(VERSION 3.15)
project(load_onnx_light LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnx_light REQUIRED)

add_executable(load_onnx_light main.cpp)
target_link_libraries(load_onnx_light PRIVATE onnx_light::onnx_light)
