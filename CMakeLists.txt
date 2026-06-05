cmake_minimum_required(VERSION 3.15)

# Default to a Release build when no build type is specified. Without this,
# single-config generators (Makefiles, Ninja) fall back to an empty build type
# (effectively unoptimized), which can make the C++ unit tests roughly an
# order of magnitude slower (e.g. ~0.08s/test vs ~0.01s/test for typical
# BackendTestCase entries that exercise reference kernels at registration
# time). Setting this *before* project() ensures the chosen flags apply to all
# downstream targets.
get_property(_is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT _is_multi_config AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
      "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

project(onnx_light VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(ONNX_LIGHT_BUILD_TESTS "Build C++ unit tests." OFF)
option(ONNX_LIGHT_BUILD_BENCHMARKS "Build C++ benchmarks." OFF)
option(ONNX_LIGHT_BENCH_GPROF "Compile benchmarks with -pg for gprof profiling." OFF)
option(ONNX_LIGHT_BENCH_WITH_UPSTREAM_ONNX
    "Fetch and build upstream onnx (protobuf-based) to enable the side-by-side BENCH_HAS_UPSTREAM_ONNX comparison block in bench_load_file."
    OFF)
option(ONNX_LIGHT_BUILD_PYTHON "Build Python extension." ON)
option(ONNX_LIGHT_INSTALL "Install the C++ library and headers." ON)
option(ONNX_ML "Enable ai.onnx.ml support." ON)

# Enable parallel compilation (file-level) on MSVC. Without /MP, cl.exe
# compiles source files within a single project sequentially even when
# `cmake --build --parallel` is used (the --parallel flag only parallelizes
# at the project/target level on MSBuild).
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/MP>)
endif()

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
set(ROOT_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_helpers/include")
set(ONNX_PROTO_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_proto")
set(ONNX_COMMON_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_lib/common")
set(ONNX_DEFS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_lib/defs")
set(ONNX_OP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_op")
set(ONNX_OPTIM_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_optim")
set(ONNX_BACKEND_TEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_backend_test")

# List all files to be built.
file(GLOB_RECURSE ONNX_LIGHT_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/*.hpp"
)
file(GLOB ONNX_LIGHT_SOURCES CONFIGURE_DEPENDS
    "${ONNX_PROTO_PATH}/onnx*.cc"
    "${ONNX_PROTO_PATH}/s*.cc"
    "${ONNX_PROTO_PATH}/t*.cc"
)
file(GLOB ONNX_COMMON_SOURCES CONFIGURE_DEPENDS "${ONNX_COMMON_PATH}/*.cc")
file(GLOB_RECURSE ONNX_DEFS_SOURCES CONFIGURE_DEPENDS "${ONNX_DEFS_PATH}/*.cc")
set(ONNX_PY_PROTOOP_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_py/_onnxpyprotoop.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_py/_onnxpy_proto.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_py/_onnxpy_op.cc"
)
set(ONNX_PY_PROTOLIB_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_py/_onnxpyprotolib.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_py/_onnxpy_lib.cc"
)
set(ONNX_PY_OPTIM_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_py/_onnxpyoptim.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_py/_onnxpy_optim.cc"
)
set(ONNX_PY_BACKEND_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_py/_onnxbackend.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_py/_onnxpy_backend_test.cc"
)
file(GLOB_RECURSE ONNX_OP_SOURCES CONFIGURE_DEPENDS "${ONNX_OP_PATH}/*.cc")
file(GLOB_RECURSE ONNX_OPTIM_SOURCES CONFIGURE_DEPENDS "${ONNX_OPTIM_PATH}/*.cc")
file(GLOB_RECURSE ONNX_BACKEND_TEST_SOURCES CONFIGURE_DEPENDS "${ONNX_BACKEND_TEST_PATH}/*.cc")
set(ONNX_INLINER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_lib/inliner/inliner.cc")
file(GLOB ONNX_SHAPE_INFERENCE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_lib/shape_inference/*.cc")
file(GLOB ONNX_VERSION_CONVERTER_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_lib/version_converter/*.cc")
set(ONNX_CHECKER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_lib/checker.cc" )
file(GLOB_RECURSE ONNX_CPP_PROTO_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc_onnx_proto/test_*.cc")
file(GLOB_RECURSE ONNX_CPP_LIB_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc_onnx_lib/test_*.cc")
file(GLOB_RECURSE ONNX_CPP_OP_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc_onnx_op/test_*.cc")
file(GLOB_RECURSE ONNX_CPP_OPTIM_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc_onnx_optim/test_*.cc")
file(GLOB_RECURSE ONNX_CPP_BACKEND_TEST_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc_onnx_backend_test/test_*.cc")
file(GLOB_RECURSE ONNX_CPP_CC_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unittests/proto/test_*.cc")
set(ONNX_CPP_TEST_SOURCES
    ${ONNX_CPP_PROTO_TEST_SOURCES}
    ${ONNX_CPP_LIB_TEST_SOURCES}
    ${ONNX_CPP_OP_TEST_SOURCES}
    ${ONNX_CPP_OPTIM_TEST_SOURCES}
    ${ONNX_CPP_BACKEND_TEST_TEST_SOURCES}
    ${ONNX_CPP_CC_TEST_SOURCES}
)
set(ONNX_CPP_ALL_TEST_SOURCES ${ONNX_CPP_TEST_SOURCES})
file(GLOB_RECURSE ONNX_LIGHT_BENCHMARK_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/bench_*.cc")
set(ONNX_LIGHT_PROTO_ALL_SOURCES ${ONNX_LIGHT_SOURCES} "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_helpers/onnx_light_helpers.cc")
set(ONNX_LIGHT_LIB_ALL_SOURCES
    ${ONNX_COMMON_SOURCES}
    ${ONNX_DEFS_SOURCES}
    ${ONNX_CHECKER_SOURCES}
    ${ONNX_INLINER_SOURCES}
    ${ONNX_SHAPE_INFERENCE_SOURCES}
    ${ONNX_VERSION_CONVERTER_SOURCES}
)

# Builds onnx_proto library.
# Build as SHARED when ONNX_LIGHT_BUILD_PYTHON=ON so that both
# _onnxpyprotoop and _onnxbackend share a single copy of the proto class
# definitions (vtables, typeid, out-of-line member functions) at runtime.
# This lets the two extensions exchange proto objects (e.g. ModelProto)
# without the serialise/reparse round-trip and avoids any duplicate
# definition concern between the two .so files.
# Pure C++ consumers (ONNX_LIGHT_BUILD_PYTHON=OFF) keep the lighter static
# variant they used to ship.
if(ONNX_LIGHT_BUILD_PYTHON)
  set(_lib_onnx_proto_type SHARED)
else()
  set(_lib_onnx_proto_type STATIC)
endif()
add_library(lib_onnx_proto ${_lib_onnx_proto_type} ${ONNX_LIGHT_HEADERS} ${ONNX_LIGHT_PROTO_ALL_SOURCES})
if(_lib_onnx_proto_type STREQUAL "SHARED" AND WIN32)
  # Export every symbol so downstream extensions can use the proto classes
  # without sprinkling __declspec(dllexport) annotations in headers.
  set_target_properties(lib_onnx_proto PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light" PREFIX "Files" FILES ${ONNX_LIGHT_HEADERS})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light" PREFIX "Files" FILES ${ONNX_LIGHT_PROTO_ALL_SOURCES})
target_include_directories(lib_onnx_proto PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/onnx_light>
    $<BUILD_INTERFACE:${ROOT_INCLUDE_PATH}>
    $<BUILD_INTERFACE:${ONNX_PROTO_PATH}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/onnx_light>
)
set_property(TARGET lib_onnx_proto PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(lib_onnx_proto PUBLIC cxx_std_20)
if(OpenMP_CXX_FOUND)
  target_link_libraries(lib_onnx_proto PUBLIC OpenMP::OpenMP_CXX)
endif()
if(OpenSSL_FOUND)
  target_compile_definitions(lib_onnx_proto PUBLIC ONNX_LIGHT_HAS_OPENSSL)
  target_link_libraries(lib_onnx_proto PUBLIC OpenSSL::Crypto)
endif()

# Builds onnx_lib library.
add_library(lib_onnx_lib STATIC ${ONNX_LIGHT_HEADERS} ${ONNX_LIGHT_LIB_ALL_SOURCES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light" PREFIX "Files" FILES ${ONNX_LIGHT_LIB_ALL_SOURCES})
target_include_directories(lib_onnx_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/onnx_light>
    $<BUILD_INTERFACE:${ROOT_INCLUDE_PATH}>
    $<BUILD_INTERFACE:${ONNX_PROTO_PATH}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/onnx_light>
)
set_property(TARGET lib_onnx_lib PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(lib_onnx_lib PROPERTIES EXPORT_NAME onnx_light)
target_compile_features(lib_onnx_lib PUBLIC cxx_std_20)
target_link_libraries(lib_onnx_lib PUBLIC lib_onnx_proto)
if(ONNX_ML)
  target_compile_definitions(lib_onnx_lib PUBLIC ONNX_ML=1)
endif()
# Precompiled headers: compile the heaviest shared headers once for all translation units.
target_precompile_headers(lib_onnx_lib PRIVATE
    <algorithm>
    <functional>
    <memory>
    <string>
    <unordered_map>
    <vector>
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_lib/defs/schema.h"
)

# Builds onnx_op library.
add_library(lib_onnx_op STATIC ${ONNX_LIGHT_HEADERS} ${ONNX_OP_SOURCES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light" PREFIX "Files" FILES ${ONNX_OP_SOURCES})
target_include_directories(lib_onnx_op PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/onnx_light>
    $<BUILD_INTERFACE:${ROOT_INCLUDE_PATH}>
    $<BUILD_INTERFACE:${ONNX_PROTO_PATH}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/onnx_light>
)
set_property(TARGET lib_onnx_op PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(lib_onnx_op PUBLIC cxx_std_20)
target_link_libraries(lib_onnx_op PUBLIC lib_onnx_proto)

# Builds onnx_optim library.
add_library(lib_onnx_optim STATIC ${ONNX_LIGHT_HEADERS} ${ONNX_OPTIM_SOURCES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light" PREFIX "Files" FILES ${ONNX_OPTIM_SOURCES})
target_include_directories(lib_onnx_optim PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/onnx_light>
    $<BUILD_INTERFACE:${ROOT_INCLUDE_PATH}>
    $<BUILD_INTERFACE:${ONNX_PROTO_PATH}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/onnx_light>
)
set_property(TARGET lib_onnx_optim PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(lib_onnx_optim PUBLIC cxx_std_20)
target_link_libraries(lib_onnx_optim PUBLIC lib_onnx_op)

# Builds onnx_backend_test library.
# This library is intentionally independent from lib_onnx_lib/lib_onnx_op
# but publicly depends on lib_onnx_proto because it uses proto types
# (e.g. ModelProto) in its headers and sources.
# It exposes a small infrastructure (struct Tensor, struct TestCase,
# expect()) and a registry of C++-implemented backend test node cases.
# It also bundles deterministic pseudo-random helpers (SplitMix64) used
# to back ``onnx_light.backend``.
add_library(lib_onnx_backend_test STATIC ${ONNX_LIGHT_HEADERS} ${ONNX_BACKEND_TEST_SOURCES})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light" PREFIX "Files" FILES ${ONNX_BACKEND_TEST_SOURCES})
target_include_directories(lib_onnx_backend_test PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/onnx_light>
    $<BUILD_INTERFACE:${ROOT_INCLUDE_PATH}>
    $<BUILD_INTERFACE:${ONNX_PROTO_PATH}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/onnx_light>
)
set_property(TARGET lib_onnx_backend_test PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(lib_onnx_backend_test PROPERTIES EXPORT_NAME onnx_backend_test)
target_compile_features(lib_onnx_backend_test PUBLIC cxx_std_20)
target_link_libraries(lib_onnx_backend_test PUBLIC lib_onnx_proto)

# Speed up the build of lib_onnx_backend_test by precompiling the small set
# of headers that virtually every one of the 400+ source files in
# onnx_light/onnx_backend_test/{cases,kernels,...} ends up parsing. The
# common base is ``onnx_backend_test/test_case.h`` (which transitively
# brings in ``onnx.h`` and ``simple_tensor.h``) and a handful of widely
# used C++ standard headers (``<vector>``, ``<string>``, ``<cstdint>``,
# ``<stdexcept>``). Without a PCH each translation unit re-parses tens of
# thousands of lines; with a PCH that work is done once for the target.
#
# This is a transparent optimization (no source changes required) and
# only affects targets that opt in. Other targets that link against
# ``lib_onnx_backend_test`` still see the regular public headers.
target_precompile_headers(lib_onnx_backend_test PRIVATE
    "${ONNX_BACKEND_TEST_PATH}/test_case.h"
    <algorithm>
    <cmath>
    <cstdint>
    <cstring>
    <stdexcept>
    <string>
    <utility>
    <vector>
)

# Builds _onnxpyprotoop, _onnxpyprotolib, _onnxpyoptim and _onnxbackend libraries.
if(ONNX_LIGHT_BUILD_PYTHON)
  # _onnxpyprotoop: proto bindings + onnx_op schema bindings.
  nanobind_add_module(_onnxpyprotoop ${ONNX_PY_PROTOOP_SOURCES})
  target_include_directories(_onnxpyprotoop PRIVATE "${ROOT_INCLUDE_PATH}" "${ONNX_PROTO_PATH}")
  target_link_libraries(_onnxpyprotoop PRIVATE lib_onnx_op)

  # _onnxpyprotolib: onnx_lib bindings (defs/parser/checker/inliner/
  # shape/version_converter). This module does not register proto classes;
  # those are registered by _onnxpyprotoop.
  nanobind_add_module(_onnxpyprotolib ${ONNX_PY_PROTOLIB_SOURCES})
  target_include_directories(_onnxpyprotolib PRIVATE "${ROOT_INCLUDE_PATH}" "${ONNX_PROTO_PATH}")
  target_link_libraries(_onnxpyprotolib PRIVATE lib_onnx_lib)

  # _onnxpyoptim: optim bindings (symbolic expressions and shape inference).
  # Split from _onnxpyproto* so the schema bindings and the shape-inference
  # bindings live in distinct extensions.
  nanobind_add_module(_onnxpyoptim ${ONNX_PY_OPTIM_SOURCES})
  target_include_directories(_onnxpyoptim PRIVATE "${ROOT_INCLUDE_PATH}" "${ONNX_PROTO_PATH}")
  target_link_libraries(_onnxpyoptim PRIVATE lib_onnx_optim)

  # _onnxbackend: backend helpers (random) and backend-test case utilities. Links
  # lib_onnx_backend_test (which publicly depends on lib_onnx_proto so that
  # ModelProto can be returned by reference to Python and resolved against the
  # single nb::class_<ModelProto> registered by _onnxpyprotoop).
  nanobind_add_module(_onnxbackend ${ONNX_PY_BACKEND_SOURCES})
  target_include_directories(_onnxbackend PRIVATE "${ROOT_INCLUDE_PATH}" "${ONNX_PROTO_PATH}")
  target_link_libraries(_onnxbackend PRIVATE lib_onnx_backend_test)

  # All extensions and the shared lib_onnx_proto.so are installed in the same
  # directory (onnx_light/onnx_py/). Set an $ORIGIN rpath so the dynamic loader
  # picks the colocated proto library at import time without LD_LIBRARY_PATH.
  if(_lib_onnx_proto_type STREQUAL "SHARED" AND (UNIX AND NOT APPLE))
    set_target_properties(_onnxpyprotoop _onnxpyprotolib _onnxpyoptim _onnxbackend PROPERTIES
        BUILD_RPATH "\$ORIGIN"
        INSTALL_RPATH "\$ORIGIN")
  elseif(_lib_onnx_proto_type STREQUAL "SHARED" AND APPLE)
    set_target_properties(_onnxpyprotoop _onnxpyprotolib _onnxpyoptim _onnxbackend PROPERTIES
        BUILD_RPATH "@loader_path"
        INSTALL_RPATH "@loader_path")
  endif()
endif()

# Builds test
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

  # test_onnx_light_helpers
  add_executable(test_onnx_light_helpers "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc_common/test_onnx_light_helpers.cc")
  target_include_directories(test_onnx_light_helpers PRIVATE "${ROOT_INCLUDE_PATH}")
  target_link_libraries(test_onnx_light_helpers PRIVATE lib_onnx_lib)
  add_test(NAME test_onnx_light_helpers COMMAND test_onnx_light_helpers)

  # gtest
  add_executable(test_onnx_light ${ONNX_CPP_TEST_SOURCES})
  target_include_directories(test_onnx_light PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light"
    "${ROOT_INCLUDE_PATH}"
    "${ONNX_PROTO_PATH}"
    "${ONNX_COMMON_PATH}"
  )
  target_link_libraries(test_onnx_light PRIVATE lib_onnx_lib lib_onnx_op lib_onnx_optim lib_onnx_backend_test GTest::gtest_main)
  gtest_discover_tests(test_onnx_light DISCOVERY_TIMEOUT 120)

  add_custom_target(onnx_cpp_tests SOURCES ${ONNX_CPP_ALL_TEST_SOURCES})
  source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc_onnx_proto" PREFIX "Source Files/cc_onnx_proto"
               FILES ${ONNX_CPP_PROTO_TEST_SOURCES})
  source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc_onnx_lib" PREFIX "Source Files/cc_onnx_lib"
               FILES ${ONNX_CPP_LIB_TEST_SOURCES})
  source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc_onnx_op" PREFIX "Source Files/cc_onnx_op"
               FILES ${ONNX_CPP_OP_TEST_SOURCES})
  source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc_onnx_optim" PREFIX "Source Files/cc_onnx_optim"
               FILES ${ONNX_CPP_OPTIM_TEST_SOURCES})
  source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc_onnx_backend_test" PREFIX "Source Files/cc_onnx_backend_test"
               FILES ${ONNX_CPP_BACKEND_TEST_TEST_SOURCES})
  source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/unittests/proto" PREFIX "Source Files/proto"
               FILES ${ONNX_CPP_CC_TEST_SOURCES})
endif()

# Each benchmark is an individual project placed in the "Benchmarks" VS solution folder.
# Created when ONNX_LIGHT_BUILD_TESTS=ON (for browsing / profiling) or
# ONNX_LIGHT_BUILD_BENCHMARKS=ON (to build and run them).
if(ONNX_LIGHT_BUILD_TESTS OR ONNX_LIGHT_BUILD_BENCHMARKS)
  # Try to locate the upstream onnx (protobuf-based) C++ library so that
  # benchmarks (e.g. bench_load_file) can profile both onnx and onnx_light
  # in the same run. Optional: skipped silently when not available.
  find_package(protobuf CONFIG QUIET)
  if(NOT TARGET protobuf::libprotobuf)
    find_package(Protobuf QUIET)
  endif()
  find_package(ONNX QUIET)

  # When ONNX_LIGHT_BENCH_WITH_UPSTREAM_ONNX=ON and find_package(ONNX) did not
  # already create the `onnx` target (which is the common case, because the
  # pip-installed onnx wheel does not ship ONNXConfig.cmake nor a precompiled
  # C++ library), fetch and build the upstream onnx repository so that the
  # `onnx` and `onnx_proto` targets become available for bench_load_file.
  if(ONNX_LIGHT_BENCH_WITH_UPSTREAM_ONNX AND NOT TARGET onnx)
    message(STATUS "ONNX_LIGHT_BENCH_WITH_UPSTREAM_ONNX=ON: fetching upstream onnx via FetchContent")
    include(FetchContent)
    set(ONNX_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    set(ONNX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(ONNX_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(ONNX_USE_LITE_PROTO OFF CACHE BOOL "" FORCE)
    set(ONNX_GEN_PB_TYPE_STUBS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      onnx_upstream
      URL https://codeload.github.com/onnx/onnx/tar.gz/refs/tags/v1.18.0
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(onnx_upstream)
  endif()

  foreach(benchmark_source IN LISTS ONNX_LIGHT_BENCHMARK_SOURCES)
    get_filename_component(benchmark_target "${benchmark_source}" NAME_WE)

    add_executable(${benchmark_target} "${benchmark_source}")
    target_include_directories(${benchmark_target} PRIVATE
        "${ROOT_INCLUDE_PATH}"
        "${ONNX_PROTO_PATH}"
    )
    target_link_libraries(${benchmark_target} PRIVATE lib_onnx_lib)
    set_property(TARGET ${benchmark_target} PROPERTY FOLDER "Benchmarks")

    if(OpenMP_CXX_FOUND)
      target_link_libraries(${benchmark_target} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if(ONNX_LIGHT_BENCH_GPROF)
      target_compile_options(${benchmark_target} PRIVATE -pg)
      target_link_options(${benchmark_target} PRIVATE -pg)
    endif()

    # When the upstream onnx library is available, enable the
    # side-by-side onnx vs onnx_light comparison block in bench_load_file
    # by defining BENCH_HAS_UPSTREAM_ONNX and linking against onnx/onnx_proto.
    #
    # Easiest path: run
    # `bash benchmarks/profile.sh perf bench_load_file --with-upstream-onnx ...`
    # which sets -DONNX_LIGHT_BENCH_WITH_UPSTREAM_ONNX=ON. CMake then uses
    # FetchContent to download and build upstream onnx, producing the `onnx`
    # and `onnx_proto` targets used below.
    # When neither find_package(ONNX) nor ONNX_LIGHT_BENCH_WITH_UPSTREAM_ONNX
    # provides the `onnx` target, this block is silently skipped and
    # bench_load_file still builds — but only the onnx_light timing line is
    # reported at runtime.
    if(TARGET onnx AND benchmark_target STREQUAL "bench_load_file")
      target_compile_definitions(${benchmark_target} PRIVATE BENCH_HAS_UPSTREAM_ONNX ONNX_ML=1)
      target_link_libraries(${benchmark_target} PRIVATE onnx onnx_proto)
    elseif(ONNX_LIGHT_BENCH_WITH_UPSTREAM_ONNX AND benchmark_target STREQUAL "bench_load_file")
      message(WARNING
        "ONNX_LIGHT_BENCH_WITH_UPSTREAM_ONNX=ON but no `onnx` target was created; "
        "BENCH_HAS_UPSTREAM_ONNX will NOT be defined for bench_load_file.")
    endif()
  endforeach()
endif()

if(ONNX_LIGHT_BUILD_PYTHON)
  # Install the extensions into the package directory. When lib_onnx_proto is
  # built SHARED (the default in Python builds), install it alongside so the
  # $ORIGIN rpath above resolves it without LD_LIBRARY_PATH.
  install(TARGETS _onnxpyprotoop _onnxpyprotolib _onnxpyoptim _onnxbackend LIBRARY DESTINATION onnx_light/onnx_py)
  if(_lib_onnx_proto_type STREQUAL "SHARED")
    install(TARGETS lib_onnx_proto
        LIBRARY DESTINATION onnx_light/onnx_py
        RUNTIME DESTINATION onnx_light/onnx_py)
  endif()
endif()

if(ONNX_LIGHT_INSTALL)
  include(CMakePackageConfigHelpers)

  # Install the static library and export its target
  install(TARGETS lib_onnx_proto lib_onnx_lib lib_onnx_op lib_onnx_optim lib_onnx_backend_test
    EXPORT onnx_lightTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  )

  # Install public headers from onnx_light/onnx_helpers/include
  install(DIRECTORY "${ROOT_INCLUDE_PATH}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )

  # Install public headers from onnx_light/onnx_proto
  install(DIRECTORY "${ONNX_PROTO_PATH}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    PATTERN "_onnxpy.cc" EXCLUDE
  )
  # Also install under onnx_proto/ for headers that include "onnx_proto/...".
  install(DIRECTORY "${ONNX_PROTO_PATH}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/onnx_proto"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    PATTERN "_onnxpy.cc" EXCLUDE
  )

  # Install public headers from onnx_light/onnx_lib/common
  install(DIRECTORY "${ONNX_COMMON_PATH}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/onnx_lib/common"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )

  # Install public root headers from onnx_light/onnx_lib
  install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_lib/checker.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_lib/onnx-data.pb.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_lib/string_utils.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/onnx_lib"
  )

  # Install public headers from onnx_light/onnx_lib/defs
  install(DIRECTORY "${ONNX_DEFS_PATH}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/onnx_lib/defs"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )

  # Install public headers from onnx_light/onnx_op
  install(DIRECTORY "${ONNX_OP_PATH}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/onnx_op"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )

  # Install public headers from onnx_light/onnx_optim
  install(DIRECTORY "${ONNX_OPTIM_PATH}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/onnx_optim"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )

  # Install public headers from onnx_light/onnx_backend_test
  install(DIRECTORY "${ONNX_BACKEND_TEST_PATH}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/onnx_backend_test"
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
