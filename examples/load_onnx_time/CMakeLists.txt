# Standalone C++ example: measure ONNX loading time with the standard onnx
# C++ library (protobuf-based).
#
# Prerequisites
# -------------
# Option A – use the system-installed onnx C++ library.  On Ubuntu/Debian:
#
#   sudo apt-get install -y libonnx-dev libprotobuf-dev
#
# Option B – let build.sh handle everything automatically:
#
#   bash examples/load_onnx_time/build.sh
#
# The helper script probes for the onnx CMake package.  When not found it
# pre-clones onnx from git and passes FETCHCONTENT_SOURCE_DIR_ONNX so that
# cmake builds onnx (and all its transitive dependencies such as protobuf,
# abseil and utf8_range) inline – no manual target_include_directories /
# target_link_libraries needed.  Explicit git tag overrides:
#
#   ONNX_GIT_TAG=v1.17.0 bash examples/load_onnx_time/build.sh
#
# Run
# ---
#   ./build-load-onnx-time/load_onnx_time path/to/model.onnx 10

cmake_minimum_required(VERSION 3.15)
project(load_onnx_time LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------
# Attempt 1 – use a system-installed or pre-installed onnx package.
# ------------------------------------------------------------------
# Try protobuf config mode first to prevent the "some but not all targets
# already defined" double-include error that occurs when both this file and
# a from-source ONNX cmake config call find_package(protobuf CONFIG).
find_package(protobuf CONFIG QUIET)
if(NOT TARGET protobuf::libprotobuf)
    # Config file not available (e.g. system libprotobuf-dev on Ubuntu/Debian
    # only ships FindProtobuf.cmake).  Fall back to module mode.
    find_package(Protobuf QUIET)
endif()
find_package(ONNX QUIET)

if(NOT TARGET onnx)
    # ------------------------------------------------------------------
    # Attempt 2 – build onnx from source via CMake FetchContent.
    #
    # All transitive dependencies (protobuf, abseil, utf8_range, …) are
    # resolved automatically by onnx's own CMakeLists – no manual
    # target_include_directories / target_link_libraries needed.
    #
    # build.sh pre-clones the onnx repository and passes
    #   -DFETCHCONTENT_SOURCE_DIR_ONNX=<path>    (skips network download)
    #   -DONNX_GIT_TAG=<tag>                      (used when not pre-cloned)
    # When invoked directly with cmake (without build.sh), cmake downloads
    # onnx from ONNX_GIT_URL at configure time.
    # ------------------------------------------------------------------
    if(NOT DEFINED ONNX_GIT_TAG)
        set(ONNX_GIT_TAG "v1.17.0")
    endif()
    if(NOT DEFINED ONNX_GIT_URL)
        set(ONNX_GIT_URL "https://github.com/onnx/onnx.git")
    endif()
    message(STATUS "load_onnx_time: ONNX not found via find_package; "
                   "building from source (${ONNX_GIT_TAG})")
    set(ONNX_ML ON CACHE BOOL "" FORCE)
    set(ONNX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(ONNX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    include(FetchContent)
    FetchContent_Declare(onnx
        GIT_REPOSITORY "${ONNX_GIT_URL}"
        GIT_TAG        "${ONNX_GIT_TAG}"
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(onnx)
endif()

add_executable(load_onnx_time main.cpp)
target_compile_definitions(load_onnx_time PRIVATE ONNX_ML=1)
target_link_libraries(load_onnx_time PRIVATE onnx onnx_proto)
