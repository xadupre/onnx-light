# Standalone C++ example: measure ONNX loading time with the standard onnx
# C++ library (protobuf-based).
#
# Prerequisites
# -------------
# Option A – use the system-installed onnx C++ library.  On Ubuntu/Debian:
#
#   sudo apt-get install -y libonnx-dev libprotobuf-dev
#
# Option B – build onnx (and optionally protobuf) from source and pass the
# install prefix via CMAKE_PREFIX_PATH (the build.sh helper handles this
# automatically, including auto-detection when no system packages are found):
#
#   cmake -S examples/load_onnx_time -B build-load-onnx-time \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_PREFIX_PATH=/path/to/onnx-install
#   cmake --build build-load-onnx-time
#
# Build this example (system onnx, fully automatic)
# --------------------------------------------------
#
#   bash examples/load_onnx_time/build.sh
#
# The helper script probes for the onnx CMake package; if not found it
# automatically clones and builds onnx (and protobuf when also absent).
# Explicit version overrides:
#
#   ONNX_GIT_TAG=v1.17.0 bash examples/load_onnx_time/build.sh
#   ONNX_GIT_TAG=v1.17.0 PROTOBUF_GIT_TAG=v3.21.12 bash examples/load_onnx_time/build.sh
#
# Run
# ---
#   ./build-load-onnx-time/load_onnx_time path/to/model.onnx 10

cmake_minimum_required(VERSION 3.15)
project(load_onnx_time LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Try config mode first (lowercase, used by from-source protobuf installs and
# by ONNX's cmake config internally).  A successful config-mode find sets the
# protobuf_DIR cache entry so that ONNX's own find_package(protobuf) call
# becomes a no-op, preventing the "Some (but not all) targets already defined"
# double-include error that occurs when both our CMakeLists.txt and ONNX's
# config file call find_package for protobuf on the same install prefix.
find_package(protobuf CONFIG QUIET)
if(NOT TARGET protobuf::libprotobuf)
    # Config file not available (e.g. system libprotobuf-dev on Ubuntu/Debian
    # only ships FindProtobuf.cmake).  Fall back to module mode.
    find_package(Protobuf REQUIRED)
endif()
find_package(ONNX REQUIRED)

add_executable(load_onnx_time main.cpp)
target_compile_definitions(load_onnx_time PRIVATE ONNX_ML=1)
target_link_libraries(load_onnx_time PRIVATE onnx onnx_proto)
