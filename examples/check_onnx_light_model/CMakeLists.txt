# Standalone C++ example: validate an ONNX model with the onnx_light checker API.
#
# Prerequisites
# -------------
# First install onnx_light as a C++ library. From the onnx-light repository
# root run:
#
#   cmake -S . -B build-install \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DONNX_LIGHT_BUILD_PYTHON=OFF \
#         -DONNX_LIGHT_BUILD_KERNELS=OFF \
#         -DCMAKE_INSTALL_PREFIX=/usr/local   # or any prefix you prefer
#   cmake --build build-install
#   cmake --install build-install
#
# ``-DONNX_LIGHT_BUILD_KERNELS=OFF`` skips building ``lib_onnx_kernels`` and
# ``lib_onnx_backend_test`` (the operator-kernel runtime and the backend-test
# case registry). They are not needed by this example, which only links the
# checker / shape-inference layer exposed by ``onnx_light::onnx_light``.
#
# Build this example
# ------------------
# Point CMAKE_PREFIX_PATH at the install prefix chosen above:
#
#   cmake -S examples/check_onnx_light_model -B build-check-onnx-light-model \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_PREFIX_PATH=/usr/local
#   cmake --build build-check-onnx-light-model
#
# Run
# ---
#   ./build-check-onnx-light-model/check_onnx_light_model path/to/model.onnx 1

cmake_minimum_required(VERSION 3.15)
project(check_onnx_light_model LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnx_light REQUIRED)

add_executable(check_onnx_light_model main.cc)
# onnx_light::onnx_light is the exported CMake name of the lib_onnx_lib target.
target_link_libraries(check_onnx_light_model PRIVATE onnx_light::onnx_light)
