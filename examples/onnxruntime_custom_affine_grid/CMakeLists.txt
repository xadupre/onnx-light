cmake_minimum_required(VERSION 3.15)
project(onnxruntime_custom_affine_grid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
else()
  list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/../run_backend_test_ort/cmake")
  find_package(Ort REQUIRED)
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR
    "ONNX Runtime was not found. Set ONNXRUNTIME_ROOT_DIR to an extracted "
    "ONNX Runtime release containing include/ and lib/.")
endif()

add_library(ort_affine_grid_custom SHARED affine_grid_custom_op.cc)
target_include_directories(ort_affine_grid_custom PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")

add_executable(run_ort_affine_grid main.cc)
target_include_directories(run_ort_affine_grid PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(run_ort_affine_grid PRIVATE "${ONNXRUNTIME_LIBRARY}")

if(UNIX AND NOT APPLE)
  set_target_properties(run_ort_affine_grid PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib")
endif()
