# Standalone C++ example: build the Add backend test node case via the
# onnx_light::onnx_backend_test library, run a tiny runtime Add against it,
# and check that the runtime outputs match the expected outputs.
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
#   cmake -S examples/run_add_node_test -B build-run-add-node-test \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_PREFIX_PATH=/usr/local
#   cmake --build build-run-add-node-test
#
# Run
# ---
#   ./build-run-add-node-test/run_add_node_test

cmake_minimum_required(VERSION 3.15)
project(run_add_node_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnx_light REQUIRED)

add_executable(run_add_node_test main.cc)
# onnx_light::onnx_backend_test is the exported CMake name of the
# lib_onnx_backend_test target. It transitively pulls in onnx_proto.
target_link_libraries(run_add_node_test PRIVATE onnx_light::onnx_backend_test)
