# Standalone C++ example: implement a brand-new kernel class for an existing
# ONNX operator (``Abs``), register it into onnx_light's shared kernel dispatch
# table, run a model that uses that operator and verify the new kernel is the
# one actually executed. This mirrors the scenario implemented by the companion
# onnx-light-cpu project.
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
#   cmake -S examples/register_custom_kernel -B build-register-custom-kernel \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_PREFIX_PATH=/usr/local
#   cmake --build build-register-custom-kernel
#
# Run
# ---
#   ./build-register-custom-kernel/register_custom_kernel

cmake_minimum_required(VERSION 3.15)
project(register_custom_kernel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnx_light REQUIRED)

add_executable(register_custom_kernel main.cc)
# onnx_light::lib_onnx_kernels is the exported CMake name of the
# lib_onnx_kernels target (the operator-kernel runtime, RunNode /
# RuntimeSession and the built-in KernelDispatchTable). It transitively pulls
# in lib_onnx_core and lib_onnx_proto.
target_link_libraries(register_custom_kernel PRIVATE onnx_light::lib_onnx_kernels)
