# Standalone C++ example: export an ONNX model to the Khronos NNEF v1.0
# format. The NNEF exporter itself is compiled as part of this example
# (sources under ``nnef/``) so the onnx_light package does not ship the
# functionality — only the proto layer of onnx_light is required.
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

add_executable(export_nnef
    main.cc
    nnef/exporter.cc
    nnef/tensor_io.cc
)
# The NNEF sources include "nnef/exporter.h" / "nnef/tensor_io.h" relative to
# the example root, so expose this directory as a private include path.
target_include_directories(export_nnef PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(export_nnef PRIVATE onnx_light::lib_onnx_proto)
