# Standalone C++ example: measure ONNX loading time with the standard onnx
# C++ library (protobuf-based).
#
# Prerequisites
# -------------
# Install the standard onnx C++ development library.  On Ubuntu/Debian:
#
#   sudo apt-get install -y libonnx-dev libprotobuf-dev
#
# Build this example
# ------------------
#
#   cmake -S examples/load_onnx_time -B build-load-onnx-time \
#         -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-load-onnx-time
#
# Or use the helper script:
#
#   bash examples/load_onnx_time/build.sh
#
# Run
# ---
#   ./build-load-onnx-time/load_onnx_time path/to/model.onnx 10

cmake_minimum_required(VERSION 3.15)
project(load_onnx_time LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Protobuf REQUIRED)
find_package(ONNX REQUIRED)

add_executable(load_onnx_time main.cpp)
target_compile_definitions(load_onnx_time PRIVATE ONNX_ML=1)
target_link_libraries(load_onnx_time PRIVATE onnx onnx_proto)
