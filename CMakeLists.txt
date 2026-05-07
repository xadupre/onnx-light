cmake_minimum_required(VERSION 3.15)
project(onnx_light VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(ONNX_LIGHT_BUILD_TESTS "Build C++ unit tests." OFF)
option(ONNX_LIGHT_BUILD_PYTHON "Build Python extension." ON)

if(ONNX_LIGHT_BUILD_PYTHON)
  # Find Python
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

  # Find nanobind
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NB_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
  find_package(nanobind CONFIG REQUIRED)
endif()

# Find OpenMP (optional)
find_package(OpenMP QUIET)

# Set include directories
set(ROOT_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/cpp/include")
set(ONNX_PROTO_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx/onnx_proto")

# Collect C++ source files (excluding the pybind binding file and test files)
file(GLOB ONNX_LIGHT_SOURCES
    "${ONNX_PROTO_PATH}/onnx*.cpp"
    "${ONNX_PROTO_PATH}/s*.cpp"
    "${ONNX_PROTO_PATH}/t*.cpp"
)

# Build the static library
add_library(lib_onnx_cpp STATIC
    ${ONNX_LIGHT_SOURCES}
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/cpp/onnx_extended_helpers.cpp"
)
target_include_directories(lib_onnx_cpp PUBLIC
    "${ROOT_INCLUDE_PATH}"
    "${ONNX_PROTO_PATH}"
)
set_property(TARGET lib_onnx_cpp PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(lib_onnx_cpp PUBLIC cxx_std_17)

if(OpenMP_CXX_FOUND)
  target_link_libraries(lib_onnx_cpp PUBLIC OpenMP::OpenMP_CXX)
endif()

if(ONNX_LIGHT_BUILD_PYTHON)
  # Build the nanobind extension
  nanobind_add_module(
      _onnxpy
      "${ONNX_PROTO_PATH}/_onnxpy.cpp"
  )
  target_include_directories(_onnxpy PRIVATE
      "${ROOT_INCLUDE_PATH}"
      "${ONNX_PROTO_PATH}"
  )
  target_link_libraries(_onnxpy PRIVATE lib_onnx_cpp)

  if(OpenMP_CXX_FOUND)
    target_link_libraries(_onnxpy PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

if(ONNX_LIGHT_BUILD_TESTS)
  enable_testing()
  add_executable(
    test_onnx_extended_helpers
    "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp/test_onnx_extended_helpers.cpp"
  )
  target_include_directories(test_onnx_extended_helpers PRIVATE
    "${ROOT_INCLUDE_PATH}"
  )
  target_link_libraries(test_onnx_extended_helpers PRIVATE lib_onnx_cpp)
  add_test(NAME test_onnx_extended_helpers COMMAND test_onnx_extended_helpers)
endif()

if(ONNX_LIGHT_BUILD_PYTHON)
  # Install the extension into the package directory
  install(TARGETS _onnxpy LIBRARY DESTINATION onnx_light/onnx/onnx_proto)
endif()
