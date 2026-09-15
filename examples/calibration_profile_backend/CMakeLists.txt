cmake_minimum_required(VERSION 3.15)
project(calibration_profile_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnx_light REQUIRED)

add_executable(calibration_profile_backend main.cc)
target_link_libraries(calibration_profile_backend PRIVATE onnx_light::lib_onnx_core)
