cmake_minimum_required(VERSION 3.15)
project(parallel_for_profiling LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnx_light REQUIRED)

add_executable(parallel_for_profiling main.cc)
target_link_libraries(parallel_for_profiling PRIVATE onnx_light::lib_onnx_kernels)
