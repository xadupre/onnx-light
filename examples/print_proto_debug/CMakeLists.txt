# Standalone C++ example: print any ONNX proto as readable text for debugging.
# The executable links against ``onnx_light::lib_onnx_proto`` only, which is
# the minimum dependency set needed to format generated protobuf messages.
#
# Prerequisites
# -------------
# First install onnx_light as a C++ library. From the onnx-light repository
# root run:
#
#   cmake -S . -B build-install \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DONNX_LIGHT_BUILD_PYTHON=OFF \
#         -DCMAKE_INSTALL_PREFIX=/usr/local   # or any prefix you prefer
#   cmake --build build-install
#   cmake --install build-install
#
# Build this example
# ------------------
# Point CMAKE_PREFIX_PATH at the install prefix chosen above:
#
#   cmake -S examples/print_proto_debug -B build-print-proto-debug \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_PREFIX_PATH=/usr/local
#   cmake --build build-print-proto-debug
#
# Run
# ---
#   ./build-print-proto-debug/print_proto_debug

cmake_minimum_required(VERSION 3.15)
project(print_proto_debug LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnx_light REQUIRED)

add_executable(print_proto_debug main.cc)
target_link_libraries(print_proto_debug PRIVATE onnx_light::lib_onnx_proto)
