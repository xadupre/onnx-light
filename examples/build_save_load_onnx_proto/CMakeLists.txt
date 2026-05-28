# Standalone C++ example: build, save and load an ONNX model using only
# the proto layer of onnx_light (no schemas, no shape inference, no checker,
# no backend kernels). The executable links against
# ``onnx_light::lib_onnx_proto`` only.
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
#   cmake -S examples/build_save_load_onnx_proto -B build-build-save-load \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_PREFIX_PATH=/usr/local
#   cmake --build build-build-save-load
#
# Run
# ---
#   ./build-build-save-load/build_save_load_onnx_proto /tmp/example_add.onnx

cmake_minimum_required(VERSION 3.15)
project(build_save_load_onnx_proto LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnx_light REQUIRED)

add_executable(build_save_load_onnx_proto main.cc)
target_link_libraries(build_save_load_onnx_proto PRIVATE onnx_light::lib_onnx_proto)
