cmake_minimum_required(VERSION 3.15)
project(contiguous_kv_decode LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnx_light REQUIRED)

add_executable(contiguous_kv_decode main.cc)
target_link_libraries(contiguous_kv_decode PRIVATE onnx_light::lib_onnx_kernels)
