# Standalone C++ example: run every C++-generated backend test case through
# onnxruntime (downloaded release build) and verify the runtime outputs match
# the expected outputs.
#
# Why a downloaded release build of onnxruntime?
# ----------------------------------------------
# Building onnxruntime from source is heavy (gigabytes, long compile times,
# Python dependencies) and not the point of this example. The official
# pre-built CPU releases shipped at
#
#   https://github.com/microsoft/onnxruntime/releases
#
# already contain the headers (``include/``) and the dynamic library
# (``lib/libonnxruntime.so[.X.Y.Z]`` on Linux, ``onnxruntime.dll`` /
# ``onnxruntime.lib`` on Windows, ``libonnxruntime.dylib`` on macOS), so this
# example just consumes one of those archives. The companion ``build.sh`` /
# ``build.bat`` scripts automate the download.
#
# Prerequisites
# -------------
# 1. Install onnx_light as a C++ library. From the onnx-light repository
#    root run:
#
#      cmake -S . -B build-install \
#            -DCMAKE_BUILD_TYPE=Release \
#            -DONNX_LIGHT_BUILD_PYTHON=OFF \
#            -DCMAKE_INSTALL_PREFIX=/usr/local   # or any prefix you prefer
#      cmake --build build-install
#      cmake --install build-install
#
# 2. Either download a pre-built onnxruntime CPU release and extract it
#    somewhere (e.g. ``/opt/onnxruntime-linux-x64-1.19.2``, with ``include/``
#    and ``lib/`` inside), or let CMake fetch one for you via the bundled
#    ``cmake/FindOrt.cmake`` module (a trimmed copy of onnx-extended's
#    ``_cmake/externals/FindOrt.cmake``).
#
# Build this example
# ------------------
# Option A -- point at an existing extracted release:
#
#   cmake -S examples/run_backend_test_ort -B build-run-backend-test-ort \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_PREFIX_PATH=/usr/local \
#         -DONNXRUNTIME_ROOT_DIR=/opt/onnxruntime-linux-x64-1.19.2
#   cmake --build build-run-backend-test-ort
#
# Option B -- let CMake download the release (pick the version via
# ``-DORT_VERSION=<x.y.z>``, default is 1.19.2):
#
#   cmake -S examples/run_backend_test_ort -B build-run-backend-test-ort \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_PREFIX_PATH=/usr/local
#   cmake --build build-run-backend-test-ort
#
# Run
# ---
# On Linux you typically need to make the onnxruntime shared library
# discoverable at runtime:
#
#   LD_LIBRARY_PATH=/opt/onnxruntime-linux-x64-1.19.2/lib \
#     ./build-run-backend-test-ort/run_backend_test_ort

cmake_minimum_required(VERSION 3.15)
project(run_backend_test_ort LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnx_light REQUIRED)

# Locate the downloaded onnxruntime release. ONNXRUNTIME_ROOT_DIR can be
# passed on the cmake command line or as an environment variable; when it is
# not set, fall back to the bundled ``FindOrt.cmake`` module (a trimmed copy
# of onnx-extended's _cmake/externals/FindOrt.cmake) which downloads the
# prebuilt release archive via FetchContent.
if(NOT DEFINED ONNXRUNTIME_ROOT_DIR AND DEFINED ENV{ONNXRUNTIME_ROOT_DIR})
  set(ONNXRUNTIME_ROOT_DIR "$ENV{ONNXRUNTIME_ROOT_DIR}")
endif()

if(ONNXRUNTIME_ROOT_DIR)
  find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT_DIR}/include"
          "${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime"
          "${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime/core/session"
    NO_DEFAULT_PATH)

  find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS "${ONNXRUNTIME_ROOT_DIR}/lib"
    NO_DEFAULT_PATH)

  if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
      "Could not find onnxruntime under '${ONNXRUNTIME_ROOT_DIR}'. "
      "Expected 'include/onnxruntime_cxx_api.h' and 'lib/libonnxruntime.*' "
      "(or 'onnxruntime.lib' on Windows). Unset ONNXRUNTIME_ROOT_DIR to let "
      "the bundled FindOrt.cmake download a release automatically.")
  endif()
else()
  message(STATUS
    "ONNXRUNTIME_ROOT_DIR not set; falling back to FindOrt to download a "
    "prebuilt onnxruntime release. Override the version with -DORT_VERSION="
    "<x.y.z>.")
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  find_package(Ort REQUIRED)
endif()

add_executable(run_backend_test_ort main.cc)
target_include_directories(run_backend_test_ort PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(run_backend_test_ort
  PRIVATE
    onnx_light::onnx_backend_test
    "${ONNXRUNTIME_LIBRARY}")

if(UNIX AND NOT APPLE)
  # Make the example find libonnxruntime.so at runtime when it sits next to
  # the downloaded release directory the user pointed at.
  set_target_properties(run_backend_test_ort PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
    BUILD_RPATH   "${ONNXRUNTIME_ROOT_DIR}/lib")
endif()
