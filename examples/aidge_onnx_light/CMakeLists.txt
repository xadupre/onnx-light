# Standalone C++ example: combine onnx-light and the Eclipse Aidge
# (https://projects.eclipse.org/projects/technology.aidge) deep learning
# framework. onnx-light is used to load and re-serialize an ONNX model
# (handling models larger than the protobuf 2 GB limit, parallel and zero-copy
# parsing, optional encrypted storage, ...), then Aidge -- when available --
# is used to actually import the resulting model and instantiate its
# computation graph.
#
# Why this example?
# -----------------
# Aidge's ONNX importer relies on the official ONNX protobuf bindings and
# inherits the 2 GB message-size limit and the duplicate-of-everything memory
# pattern. onnx-light bypasses protobuf entirely, so a common workflow is:
#
#   1. load the (potentially huge / encrypted / split) ONNX file with
#      onnx-light;
#   2. inspect / patch / decrypt the in-memory ModelProto;
#   3. re-serialize a clean, protobuf-compatible model to a temporary file;
#   4. hand that file to Aidge's importONNX().
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
#      cmake --build  build-install
#      cmake --install build-install
#
# 2. (Optional) Install Eclipse Aidge -- at least ``aidge_core`` and
#    ``aidge_onnx`` -- following the project's instructions
#    (https://eclipse.dev/aidge/source/GetStarted/install.html). When the
#    Aidge CMake packages are available, this example also runs the imported
#    model through Aidge. If Aidge is not found the example still builds and
#    only performs the onnx-light steps.
#
# Build this example
# ------------------
# Point ``CMAKE_PREFIX_PATH`` at the install prefix(es) chosen above. The
# Aidge prefix can be appended with ``;`` (semicolon) on all platforms:
#
#   cmake -S examples/aidge_onnx_light -B build-aidge-onnx-light \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_PREFIX_PATH="/usr/local;/opt/aidge"
#   cmake --build build-aidge-onnx-light
#
# Pass ``-DAIDGE_ONNX_LIGHT_REQUIRE_AIDGE=ON`` to make Aidge mandatory and
# fail the configure step if it is missing.
#
# Run
# ---
#   ./build-aidge-onnx-light/aidge_onnx_light path/to/model.onnx \
#       [output.onnx]

cmake_minimum_required(VERSION 3.15)
project(aidge_onnx_light LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(AIDGE_ONNX_LIGHT_REQUIRE_AIDGE
       "Fail the configure step if the Eclipse Aidge packages cannot be found"
       OFF)

find_package(onnx_light REQUIRED)

# Aidge ships a few independent CMake packages. We try to find the two that
# this example needs (``aidge_core`` for the graph data structures and
# ``aidge_onnx`` for the ONNX importer). Aidge is treated as optional by
# default so the example also builds in environments that only have
# onnx-light installed.
if(AIDGE_ONNX_LIGHT_REQUIRE_AIDGE)
  find_package(aidge_core CONFIG REQUIRED)
  find_package(aidge_onnx CONFIG REQUIRED)
else()
  find_package(aidge_core CONFIG QUIET)
  find_package(aidge_onnx CONFIG QUIET)
endif()

add_executable(aidge_onnx_light main.cc)
target_link_libraries(aidge_onnx_light PRIVATE onnx_light::lib_onnx_proto)

if(aidge_core_FOUND AND aidge_onnx_FOUND)
  message(STATUS
          "aidge_onnx_light: Aidge found, enabling Aidge integration "
          "(aidge_core ${aidge_core_VERSION}, aidge_onnx ${aidge_onnx_VERSION})")
  target_compile_definitions(aidge_onnx_light PRIVATE AIDGE_ONNX_LIGHT_HAS_AIDGE=1)
  target_link_libraries(aidge_onnx_light PRIVATE _aidge_core _aidge_onnx)
else()
  message(STATUS
          "aidge_onnx_light: Aidge not found, building the onnx-light-only "
          "code path. Install Eclipse Aidge and re-run CMake to enable the "
          "Aidge integration.")
endif()
