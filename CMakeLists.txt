cmake_minimum_required(VERSION 3.15)
project(onnx_light VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(ONNX_LIGHT_BUILD_TESTS "Build C++ unit tests." OFF)
option(ONNX_LIGHT_BUILD_BENCHMARKS "Build C++ benchmarks." OFF)
option(ONNX_LIGHT_BENCH_GPROF "Compile benchmarks with -pg for gprof profiling." OFF)
option(ONNX_LIGHT_BUILD_PYTHON "Build Python extension." ON)
option(ONNX_LIGHT_INSTALL "Install the C++ library and headers." ON)

include(GNUInstallDirs)

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

# Find OpenSSL (optional – enables encrypted model save/load)
find_package(OpenSSL QUIET)

# Set include directories
set(ROOT_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/cpp/include")
set(ONNX_PROTO_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx/onnx_proto")
set(ONNX_COMMON_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx/common")
set(ONNX_DEFS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx/defs")

# Collect header files so IDE generators such as Visual Studio show them in the project tree.
file(GLOB_RECURSE ONNX_LIGHT_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/*.hpp"
)

# Collect C++ source files (excluding the pybind binding file and test files)
file(GLOB ONNX_LIGHT_SOURCES CONFIGURE_DEPENDS
    "${ONNX_PROTO_PATH}/onnx*.cpp"
    "${ONNX_PROTO_PATH}/s*.cpp"
    "${ONNX_PROTO_PATH}/t*.cpp"
)

# Additional ONNX compatibility sources from folder onnx/common
file(GLOB ONNX_COMMON_SOURCES CONFIGURE_DEPENDS
    "${ONNX_COMMON_PATH}/*.cc"
)

# ONNX defs sources from folder onnx/defs
file(GLOB_RECURSE ONNX_DEFS_SOURCES CONFIGURE_DEPENDS
    "${ONNX_DEFS_PATH}/*.cc"
)

# Inliner source (depends on visitor.h and attribute_binder.h, both adapted for onnx-light)
set(ONNX_INLINER_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx/inliner/inliner.cc"
)

# Version converter sources
file(GLOB ONNX_SHAPE_INFERENCE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx/shape_inference/*.cc"
)

# Version converter sources
file(GLOB ONNX_VERSION_CONVERTER_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx/version_converter/*.cc"
)

# checker_io.cc: filesystem-security functions from the ONNX checker (no shape
# inference dependency), needed to link any target that calls
# checker::resolve_external_data_location or checker::open_external_data.
# implementation.cc: FunctionNodeInferenceCtx, InferFunctionOutputTypes, and
# InferFunctionOutputTypesFromBytes (bytes-level Python-binding wrapper).
set(ONNX_CHECKER_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx/checker.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx/checker_io.cc"
)

file(GLOB_RECURSE ONNX_CPP_TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp/test_*.cpp"
)
list(REMOVE_ITEM ONNX_CPP_TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp/test_onnx_light_helpers.cpp"
)

file(GLOB_RECURSE ONNX_LIGHT_BENCHMARK_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/bench_*.cpp"
)

set(ONNX_LIGHT_CPP_ALL_SOURCES
    ${ONNX_LIGHT_SOURCES}
    ${ONNX_COMMON_SOURCES}
    ${ONNX_DEFS_SOURCES}
    ${ONNX_CHECKER_SOURCES}
    ${ONNX_INLINER_SOURCES}
    ${ONNX_SHAPE_INFERENCE_SOURCES}
    ${ONNX_VERSION_CONVERTER_SOURCES}
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/cpp/onnx_light_helpers.cpp"
)

# Build the static library
add_library(lib_onnx_cpp STATIC
    ${ONNX_LIGHT_HEADERS}
    ${ONNX_LIGHT_CPP_ALL_SOURCES}
)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light" PREFIX "Files" FILES ${ONNX_LIGHT_HEADERS})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light" PREFIX "Files" FILES ${ONNX_LIGHT_CPP_ALL_SOURCES})
target_include_directories(lib_onnx_cpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/onnx_light>
    $<BUILD_INTERFACE:${ROOT_INCLUDE_PATH}>
    $<BUILD_INTERFACE:${ONNX_PROTO_PATH}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/onnx_light>
)
set_property(TARGET lib_onnx_cpp PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(lib_onnx_cpp PROPERTIES EXPORT_NAME onnx_light)
target_compile_features(lib_onnx_cpp PUBLIC cxx_std_17)

if(OpenMP_CXX_FOUND)
  target_link_libraries(lib_onnx_cpp PUBLIC OpenMP::OpenMP_CXX)
endif()

if(OpenSSL_FOUND)
  target_compile_definitions(lib_onnx_cpp PUBLIC ONNX_LIGHT_HAS_OPENSSL)
  target_link_libraries(lib_onnx_cpp PUBLIC OpenSSL::Crypto)
endif()

if(ONNX_LIGHT_BUILD_PYTHON)
  # Build the nanobind extension
  nanobind_add_module(
      _onnxpy
      "${ONNX_PROTO_PATH}/_onnxpy.cpp"
      "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx/bindings/_onnxpy_submodules.cpp"
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

  # Find or fetch GTest
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  include(GoogleTest)

  add_executable(
    test_onnx_light_helpers
    "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp/test_onnx_light_helpers.cpp"
  )
  target_include_directories(test_onnx_light_helpers PRIVATE
    "${ROOT_INCLUDE_PATH}"
  )
  target_link_libraries(test_onnx_light_helpers PRIVATE lib_onnx_cpp)
  add_test(NAME test_onnx_light_helpers COMMAND test_onnx_light_helpers)

  add_executable(
    test_onnx_light
    ${ONNX_CPP_TEST_SOURCES}
  )
  target_include_directories(test_onnx_light PRIVATE
    "${ROOT_INCLUDE_PATH}"
    "${ONNX_PROTO_PATH}"
    "${ONNX_COMMON_PATH}"
  )
  target_link_libraries(test_onnx_light PRIVATE lib_onnx_cpp GTest::gtest_main)
  gtest_discover_tests(test_onnx_light DISCOVERY_TIMEOUT 120)
endif()

if(ONNX_LIGHT_BUILD_BENCHMARKS)
  foreach(benchmark_source IN LISTS ONNX_LIGHT_BENCHMARK_SOURCES)
    get_filename_component(benchmark_target "${benchmark_source}" NAME_WE)

    add_executable(${benchmark_target}
        "${benchmark_source}"
    )
    target_include_directories(${benchmark_target} PRIVATE
        "${ROOT_INCLUDE_PATH}"
        "${ONNX_PROTO_PATH}"
    )
    target_link_libraries(${benchmark_target} PRIVATE lib_onnx_cpp)

    if(OpenMP_CXX_FOUND)
      target_link_libraries(${benchmark_target} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if(ONNX_LIGHT_BENCH_GPROF)
      target_compile_options(${benchmark_target} PRIVATE -pg)
      target_link_options(${benchmark_target} PRIVATE -pg)
    endif()
  endforeach()
endif()

if(ONNX_LIGHT_BUILD_PYTHON)
  # Install the extension into the package directory
  install(TARGETS _onnxpy LIBRARY DESTINATION onnx_light/onnx/onnx_proto)
endif()

if(ONNX_LIGHT_INSTALL)
  include(CMakePackageConfigHelpers)

  # Install the static library and export its target
  install(TARGETS lib_onnx_cpp
    EXPORT onnx_lightTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  )

  # Install public headers from onnx_light/cpp/include
  install(DIRECTORY "${ROOT_INCLUDE_PATH}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )

  # Install public headers from onnx_light/onnx/onnx_proto
  install(DIRECTORY "${ONNX_PROTO_PATH}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    PATTERN "_onnxpy.cpp" EXCLUDE
  )

  # Install public headers from onnx_light/onnx/common
  install(DIRECTORY "${ONNX_COMMON_PATH}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/onnx/common"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )

  # Install public headers from onnx_light/onnx/defs
  install(DIRECTORY "${ONNX_DEFS_PATH}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/onnx/defs"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )

  # Export the targets file
  install(EXPORT onnx_lightTargets
    FILE onnx_lightTargets.cmake
    NAMESPACE onnx_light::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnx_light"
  )

  # Generate and install the package config files
  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/onnx_lightConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/onnx_lightConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnx_light"
  )

  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/onnxConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/onnxConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnx"
  )

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/onnx_lightConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
  )

  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/onnxConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
  )

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/onnx_lightConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/onnx_lightConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnx_light"
  )

  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/onnxConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/onnxConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnx"
  )
endif()
