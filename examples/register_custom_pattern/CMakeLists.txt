# Standalone C++ example: define and use a custom graph-rewriting pattern
# without linking the standard ONNX pattern extension.
#
# Build after installing onnx_light:
#
#   cmake -S examples/register_custom_pattern -B build-register-custom-pattern \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_PREFIX_PATH=/path/to/onnx-light/install
#   cmake --build build-register-custom-pattern

cmake_minimum_required(VERSION 3.15)
project(register_custom_pattern LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnx_light REQUIRED)

add_executable(register_custom_pattern main.cc)
target_link_libraries(register_custom_pattern PRIVATE onnx_light::lib_onnx_core)
