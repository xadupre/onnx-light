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

# Collect C++ source files (excluding the pybind binding file and test files)
file(GLOB ONNX_LIGHT_SOURCES
    "${ONNX_PROTO_PATH}/onnx*.cpp"
    "${ONNX_PROTO_PATH}/s*.cpp"
    "${ONNX_PROTO_PATH}/t*.cpp"
)

# Additional ONNX compatibility sources from onnx/common
set(ONNX_COMMON_SOURCES
    "${ONNX_COMMON_PATH}/assertions.cc"
    "${ONNX_COMMON_PATH}/status.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx/version_converter/internal_symbol.cc"
    "${ONNX_COMMON_PATH}/ir_pb_converter.cc"
)

# ONNX defs sources from onnx/defs
set(ONNX_DEFS_SOURCES
    "${ONNX_DEFS_PATH}/attr_proto_util.cc"
    "${ONNX_DEFS_PATH}/data_type_utils.cc"
    "${ONNX_DEFS_PATH}/doc_strings.cc"
    "${ONNX_DEFS_PATH}/function.cc"
    "${ONNX_DEFS_PATH}/parser.cc"
    "${ONNX_DEFS_PATH}/schema.cc"
    "${ONNX_DEFS_PATH}/tensor_util.cc"
)

# Inliner source (depends on visitor.h and attribute_binder.h, both adapted for onnx-light)
set(ONNX_INLINER_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx/inliner/inliner.cc"
)

# Version converter sources
set(ONNX_VERSION_CONVERTER_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx/version_converter/convert.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx/version_converter/helper.cc"
)

# checker_io.cc: filesystem-security functions from the ONNX checker (no shape
# inference dependency), needed to link any target that calls
# checker::resolve_external_data_location or checker::open_external_data.
set(ONNX_CHECKER_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx/checker_io.cc"
)

# Build the static library
add_library(lib_onnx_cpp STATIC
    ${ONNX_LIGHT_SOURCES}
    ${ONNX_COMMON_SOURCES}
    ${ONNX_DEFS_SOURCES}
    ${ONNX_CHECKER_SOURCES}
    ${ONNX_INLINER_SOURCES}
    ${ONNX_VERSION_CONVERTER_SOURCES}
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/cpp/onnx_light_helpers.cpp"
)
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
    "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp/test_onnx_protos.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp/test_shape_inference.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp/test_onnx_common.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp/test_onnx_defs.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp/test_onnx_parser.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp/test_onnx_helper.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp/test_onnx_alignment_options.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp/test_onnx_threads.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp/test_onnx_checker.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp/test_schema_registration.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp/test_onnx_ir.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp/test_function_get.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp/test_data_propagation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp/test_function_verify.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp/test_function_context.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp/test_op_reg.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp/test_onnx_crypt.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cpp/test_inliner.cpp"
  )
  target_include_directories(test_onnx_light PRIVATE
    "${ROOT_INCLUDE_PATH}"
    "${ONNX_PROTO_PATH}"
    "${ONNX_COMMON_PATH}"
  )
  target_link_libraries(test_onnx_light PRIVATE lib_onnx_cpp GTest::gtest_main)
  gtest_discover_tests(test_onnx_light DISCOVERY_TIMEOUT 30)
endif()

if(ONNX_LIGHT_BUILD_BENCHMARKS)
  add_executable(bench_parse_serialize
      "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/bench_parse_serialize.cpp"
  )
  target_include_directories(bench_parse_serialize PRIVATE
      "${ROOT_INCLUDE_PATH}"
      "${ONNX_PROTO_PATH}"
  )
  target_link_libraries(bench_parse_serialize PRIVATE lib_onnx_cpp)

  if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_parse_serialize PRIVATE OpenMP::OpenMP_CXX)
  endif()

  if(ONNX_LIGHT_BENCH_GPROF)
    target_compile_options(bench_parse_serialize PRIVATE -pg)
    target_link_options(bench_parse_serialize PRIVATE -pg)
  endif()

  add_executable(bench_load_file
      "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/bench_load_file.cpp"
  )
  target_include_directories(bench_load_file PRIVATE
      "${ROOT_INCLUDE_PATH}"
      "${ONNX_PROTO_PATH}"
  )
  target_link_libraries(bench_load_file PRIVATE lib_onnx_cpp)

  if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_load_file PRIVATE OpenMP::OpenMP_CXX)
  endif()

  if(ONNX_LIGHT_BENCH_GPROF)
    target_compile_options(bench_load_file PRIVATE -pg)
    target_link_options(bench_load_file PRIVATE -pg)
  endif()

  add_executable(bench_save_file
      "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/bench_save_file.cpp"
  )
  target_include_directories(bench_save_file PRIVATE
      "${ROOT_INCLUDE_PATH}"
      "${ONNX_PROTO_PATH}"
  )
  target_link_libraries(bench_save_file PRIVATE lib_onnx_cpp)

  if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_save_file PRIVATE OpenMP::OpenMP_CXX)
  endif()

  if(ONNX_LIGHT_BENCH_GPROF)
    target_compile_options(bench_save_file PRIVATE -pg)
    target_link_options(bench_save_file PRIVATE -pg)
  endif()

  add_executable(bench_load_external_nocopy
      "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/bench_load_external_nocopy.cpp"
  )
  target_include_directories(bench_load_external_nocopy PRIVATE
      "${ROOT_INCLUDE_PATH}"
      "${ONNX_PROTO_PATH}"
  )
  target_link_libraries(bench_load_external_nocopy PRIVATE lib_onnx_cpp)

  if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_load_external_nocopy PRIVATE OpenMP::OpenMP_CXX)
  endif()

  if(ONNX_LIGHT_BENCH_GPROF)
    target_compile_options(bench_load_external_nocopy PRIVATE -pg)
    target_link_options(bench_load_external_nocopy PRIVATE -pg)
  endif()
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
