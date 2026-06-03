# Standalone C++ example: export an ONNX model to the Khronos NNEF v1.0
# format using the onnx_light C++ API.  The executable links against
# ``onnx_light::onnx_nnef`` (which transitively pulls ``lib_onnx_proto``).
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
#   cmake -S examples/export_nnef -B build-export-nnef \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_PREFIX_PATH=/usr/local
#   cmake --build build-export-nnef
#
# Run
# ---
#   ./build-export-nnef/export_nnef path/to/model.onnx path/to/out_nnef

cmake_minimum_required(VERSION 3.15)
project(export_nnef LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnx_light REQUIRED)

add_executable(export_nnef main.cc)
# onnx_light::onnx_nnef is the exported CMake name of the lib_onnx_nnef target.
target_link_libraries(export_nnef PRIVATE onnx_light::onnx_nnef)
