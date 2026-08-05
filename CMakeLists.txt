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

project(onnx_light VERSION 0.1.12 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(ONNX_LIGHT_BUILD_TESTS "Build C++ unit tests." OFF)
option(ONNX_LIGHT_BUILD_BENCHMARKS "Build C++ benchmarks." OFF)
option(ONNX_LIGHT_BUILD_FUZZERS
    "Build libFuzzer-instrumented C++ fuzz harnesses (fuzz/fuzz_*.cc). Requires Clang and adds -fsanitize=fuzzer,address."
    OFF)
option(ONNX_LIGHT_BENCH_GPROF "Compile benchmarks with -pg for gprof profiling." OFF)
option(ONNX_LIGHT_BENCH_WITH_UPSTREAM_ONNX
    "Fetch and build upstream onnx (protobuf-based) to enable the side-by-side BENCH_HAS_UPSTREAM_ONNX comparison block in bench_load_file."
    OFF)
option(ONNX_LIGHT_BUILD_PYTHON "Build Python extension." ON)
option(ONNX_LIGHT_INSTALL "Install the C++ library and headers." ON)
option(ONNX_LIGHT_BUILD_KERNELS
    "Build the operator-kernel runtime (lib_onnx_kernels) and the backend-test registry (lib_onnx_backend_test). Turn OFF to install only the schema / checker / shape-inference / version-converter libraries (useful for downstream projects that only need onnx_light::lib_onnx_lib or onnx_light::lib_onnx_proto)."
    ON)
option(ONNX_LIGHT_PROVIDE_ONNX_TARGETS
    "Expose drop-in `onnx` and `onnx_proto` CMake targets (aliasing the onnx-light libraries) so onnx-light can be consumed as a replacement for upstream onnx, e.g. via FetchContent from onnxruntime. When ON the aggregate `onnx` target bundles the schema / checker / shape-inference libraries and `onnx_proto` aliases lib_onnx_proto."
    OFF)
option(ONNX_ML "Enable ai.onnx.ml support." ON)
option(ONNX_HARDENING
    "Enable OpenSSF Compiler Hardening Guide flags for C/C++ targets. \
See cmake/Hardening.cmake and \
https://best.openssf.org/Compiler-Hardening-Guides/Compiler-Options-Hardening-Guide-for-C-and-C++.html"
    OFF)

# When ONNX_LIGHT_BUILD_KERNELS=OFF and ONNX_LIGHT_BUILD_TESTS=ON the test
# executable is built in "reduced" mode: only the proto / schema / shape-
# inference / optimisation tests are compiled (unittests/cc/onnx_proto,
# onnx_lib, onnx_op, onnx_shapes).  Kernel, backend-test, and gradient test
# sources are excluded because the libraries they depend on are not built.
# The Python extensions are tolerant of ONNX_LIGHT_BUILD_KERNELS=OFF:
# in that "reduced" build the _onnxpykernels and _onnxpybackend extensions are
# simply not produced (the surrounding Python package raises an explicit error
# when the reference runtime or backend-test helpers are requested).

# Enable parallel compilation (file-level) on MSVC when using MSBuild. Without
# /MP, cl.exe compiles source files within a single MSBuild project sequentially
# even when `cmake --build --parallel` is used (that flag only parallelizes at
# the project/target level on MSBuild). With Ninja, /MP is counter-productive:
# Ninja already spawns one cl.exe per TU in parallel, and adding /MP causes each
# cl.exe to also fork child compiler processes, over-subscribing the available
# CPU cores and slowing the build significantly. scikit-build-core (used in CI)
# prefers Ninja when it is in PATH (as it is on GitHub-hosted Windows runners),
# so guarding /MP on the generator prevents that over-subscription.
if(MSVC)
  if(NOT CMAKE_GENERATOR MATCHES "Ninja")
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/MP>)
  endif()
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/WX>)
endif()

include(GNUInstallDirs)

# Precompiled headers vs. compiler-cache launchers (sccache/ccache).
#
# A PCH turns the heaviest shared headers into a one-shot parse cost and speeds
# up a single cold build noticeably. However, PCH is fundamentally incompatible
# with sccache on MSVC: sccache marks every ``/Fp`` (use-PCH) and ``/Yc``
# (create-PCH) compilation as *non-cacheable*. In CI the whole point of the
# launcher is to reuse compiled objects across runs, so with PCH enabled the
# cache hit rate collapses (observed ~9% on the Windows core build, with 659
# ``/Fp`` non-cacheable compilations) and every run recompiles the bulk of the
# tree from scratch, making the build dramatically slower than it should be.
#
# So: when a compiler-cache launcher is configured (as it is in CI via
# ``CMAKE_CXX_COMPILER_LAUNCHER=sccache``), disable PCH by default so the
# launcher can cache 100% of the translation units and warm runs are near
# instant. Plain local builds (no launcher) keep PCH on to speed up the cold
# build. ``ONNX_LIGHT_USE_PCH`` lets users force either behaviour explicitly.
if(DEFINED ONNX_LIGHT_USE_PCH)
  set(_onnx_light_use_pch ${ONNX_LIGHT_USE_PCH})
elseif(CMAKE_CXX_COMPILER_LAUNCHER MATCHES "sccache|ccache"
    OR CMAKE_C_COMPILER_LAUNCHER MATCHES "sccache|ccache")
  set(_onnx_light_use_pch OFF)
else()
  set(_onnx_light_use_pch ON)
endif()
message(STATUS "onnx-light: precompiled headers (ONNX_LIGHT_USE_PCH): ${_onnx_light_use_pch}")

# Wrapper around ``target_precompile_headers`` that honours ``_onnx_light_use_pch``.
function(onnx_light_target_precompile_headers target)
  if(_onnx_light_use_pch)
    target_precompile_headers(${target} ${ARGN})
  endif()
endfunction()

# Optional OpenSSF compiler hardening flags. Resolves the list of supported
# flags up front so each target definition can call onnx_light_apply_hardening
# unconditionally.
if(ONNX_HARDENING)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  include(Hardening)
else()
  function(onnx_light_apply_hardening target)
  endfunction()
endif()

if(ONNX_LIGHT_BUILD_PYTHON)
  # Find Python. Development.SABIModule (CMake >= 3.26) is needed to enable
  # nanobind's STABLE_ABI mode, which produces a single abi3-tagged extension
  # that is forward-compatible across all CPython 3.12+ versions. It is
  # requested as an optional component so that older CMake installations (and
  # free-threaded interpreters, which do not expose this target) continue to
  # work without error.
  if(CMAKE_CROSSCOMPILING)
    # When cross-compiling, the interpreter and target dev libraries must be
    # located independently (host ``Python`` interpreter + target ``Python3``
    # development package).
    find_package(Python3 3.10 REQUIRED COMPONENTS Development.Module
      OPTIONAL_COMPONENTS Development.SABIModule)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter)

    # nanobind >= 2.13 only auto-runs its own find_package(Python ...) when
    # neither Python::Interpreter nor Python::Module exist yet. Since
    # Python::Interpreter is already provided above (host interpreter, while
    # Python3::Module above is the cross-compilation target's dev libs),
    # nanobind would otherwise skip that step and fail to find Python::Module.
    if(NOT TARGET Python::Module)
      add_library(Python::Module ALIAS Python3::Module)
    endif()
    if(TARGET Python3::SABIModule AND NOT TARGET Python::SABIModule)
      add_library(Python::SABIModule ALIAS Python3::SABIModule)
    endif()
  else()
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module
      OPTIONAL_COMPONENTS Development.SABIModule)
  endif()

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

# BLAKE3 (vendored, onnx_light/onnx_proto/blake3) is written in C, so enable the
# C language for the few translation units that make up the hasher.
enable_language(C)

# Set include directories
set(ROOT_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_helpers/include")
set(ONNX_PROTO_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_proto")
set(ONNX_COMMON_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_lib/common")
set(ONNX_DEFS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_lib/defs")
set(ONNX_CORE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_core")
set(ONNX_MANIPULATIONS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_manipulations")
set(ONNX_OP_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_op")
set(ONNX_SHAPE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_extensions/shapes")
set(ONNX_BACKEND_TEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_extensions/backend_test")
set(ONNX_KERNELS_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_extensions/kernels")
set(ONNX_GRADIENT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_extensions/gradient")

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
file(GLOB ONNX_MANIPULATIONS_SOURCES CONFIGURE_DEPENDS "${ONNX_MANIPULATIONS_PATH}/*.cc")
set(ONNX_PY_PROTOOP_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_py/_onnxpy_proto.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_py/_onnxpy_op.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_py/_onnxpyprotoop.cc"
)
set(ONNX_PY_PROTOLIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_py/_onnxpy_lib.cc")
set(ONNX_PY_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_py/_onnxpy_core.cc")
set(ONNX_PY_KERNELS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_py/_onnxpy_kernels.cc")
set(ONNX_PY_BACKEND_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_py/_onnxpy_backend_test.cc")
set(ONNX_PY_GRADIENT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_py/_onnxpy_gradient.cc")

file(GLOB_RECURSE ONNX_CORE_SOURCES CONFIGURE_DEPENDS "${ONNX_CORE_PATH}/*.cc")
file(GLOB_RECURSE ONNX_OP_SOURCES CONFIGURE_DEPENDS "${ONNX_OP_PATH}/*.cc")
file(GLOB_RECURSE ONNX_SHAPE_SOURCES CONFIGURE_DEPENDS "${ONNX_SHAPE_PATH}/*.cc")
# Kernel sources (operator implementations) and the small runtime
# infrastructure they rely on (``TestCase``, ``Tensor``, ``RunNodes``,
# pseudo-random helpers, ...) live in their own ``onnx_light/onnx_extensions/kernels``
# tree and are compiled into the dedicated ``lib_onnx_kernels`` library so
# that consumers can take a dependency on the kernels alone, independent
# of the surrounding backend-test cases.
file(GLOB_RECURSE ONNX_KERNELS_SOURCES CONFIGURE_DEPENDS "${ONNX_KERNELS_PATH}/*.cc")
file(GLOB_RECURSE ONNX_GRADIENT_SOURCES CONFIGURE_DEPENDS "${ONNX_GRADIENT_PATH}/*.cc")
# ``lib_onnx_backend_test`` contains the test-case registries
# (``cases/``, ``cases_for_shapes/``, ``cases_numerical/``) and
# ``collect_test_cases.cc`` which implements CollectTestCases /
# CollectTestCasesByName by delegating to the per-category Collect* helpers.
# It depends publicly on ``lib_onnx_kernels`` for ``TestCase`` / ``Tensor`` /
# kernel headers, and on ``lib_onnx_core`` for the TestCase definition itself.
file(GLOB_RECURSE ONNX_BACKEND_TEST_SOURCES CONFIGURE_DEPENDS "${ONNX_BACKEND_TEST_PATH}/*.cc")
set(ONNX_INLINER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_lib/inliner/inliner.cc")
file(GLOB ONNX_SHAPE_INFERENCE_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_lib/shape_inference/*.cc")
file(GLOB ONNX_VERSION_CONVERTER_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_lib/version_converter/*.cc")
set(ONNX_CHECKER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_lib/checker.cc" )
file(GLOB_RECURSE ONNX_LIGHT_BENCHMARK_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/bench_*.cc")

# Vendored BLAKE3 hasher (onnx_light/onnx_proto/blake3). Compiled portable-only
# (no SIMD/assembly object files) and with BLAKE3_USE_TBB so the std::async based
# join in blake3_join.cc parallelizes large hashes without a oneTBB dependency.
set(ONNX_BLAKE3_PATH "${ONNX_PROTO_PATH}/blake3")
set(ONNX_BLAKE3_SOURCES
    "${ONNX_BLAKE3_PATH}/blake3.c"
    "${ONNX_BLAKE3_PATH}/blake3_dispatch.c"
    "${ONNX_BLAKE3_PATH}/blake3_portable.c"
    "${ONNX_BLAKE3_PATH}/blake3_join.cc"
    "${ONNX_BLAKE3_PATH}/blake3_hash.cc")
set_source_files_properties(${ONNX_BLAKE3_SOURCES} PROPERTIES COMPILE_DEFINITIONS
    "BLAKE3_USE_TBB;BLAKE3_NO_SSE2;BLAKE3_NO_SSE41;BLAKE3_NO_AVX2;BLAKE3_NO_AVX512;BLAKE3_USE_NEON=0")

set(ONNX_LIGHT_PROTO_ALL_SOURCES
    ${ONNX_LIGHT_SOURCES}
    ${ONNX_BLAKE3_SOURCES}
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_helpers/onnx_light_helpers.cc")
set(ONNX_LIGHT_LIB_ALL_SOURCES
    "${ONNX_COMMON_PATH}/ir_pb_converter.cc"
    ${ONNX_DEFS_SOURCES}
    ${ONNX_CHECKER_SOURCES}
    ${ONNX_INLINER_SOURCES}
    ${ONNX_SHAPE_INFERENCE_SOURCES}
    ${ONNX_VERSION_CONVERTER_SOURCES}
)

# Build the core C++ libraries (lib_onnx_proto / lib_onnx_lib / lib_onnx_op /
# lib_onnx_shape / lib_onnx_kernels / lib_onnx_backend_test) as SHARED when
# ONNX_LIGHT_BUILD_PYTHON=ON so that all Python extensions (_onnxpyprotoop,
# _onnxpyprotolib, _onnxpycore, _onnxpykernels and _onnxpybackend) share a
# single copy of every library's compiled code (vtables, typeid, out-of-line
# member functions, kernel registries, ...) at runtime. This avoids
# duplicating tens of megabytes of compiled code across the five Python
# extension .so/.pyd files and makes it safe for the extensions to exchange
# objects (e.g. ModelProto, TestCase) by pointer.
#
# Pure C++ consumers (ONNX_LIGHT_BUILD_PYTHON=OFF) keep the lighter static
# variant they used to ship.
#
# Windows and macOS keep the five non-proto libs STATIC even when
# ONNX_LIGHT_BUILD_PYTHON=ON: on Windows ``WINDOWS_EXPORT_ALL_SYMBOLS`` does
# not export data symbols (e.g. the ``kDoc_*`` arrays), which would break
# the C++ test executable and any cross-DLL data references, and on macOS
# nanobind extensions are compiled with hidden visibility so RTTI for
# exception types defined in those libs would not match across .dylib
# boundaries. The original ``lib_onnx_proto`` SHARED variant predates this
# change and is preserved on all platforms.
if(ONNX_LIGHT_BUILD_PYTHON)
  set(_lib_onnx_proto_type SHARED)
else()
  set(_lib_onnx_proto_type STATIC)
endif()
if(ONNX_LIGHT_BUILD_PYTHON AND UNIX AND NOT APPLE)
  set(_onnx_light_lib_type SHARED)
else()
  set(_onnx_light_lib_type STATIC)
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

# Reduce the binary size of lib_onnx_proto. The proto message classes are
# macro-generated (SERIALIZATION_METHOD / FIELD in stream_class.h), which emits
# a large number of small out-of-line forwarders and inline accessors per class
# and per field. Many of these are never referenced by a given consumer.
# Compiling each function/data symbol into its own section and letting the
# linker garbage-collect unreferenced sections (and fold identical ones) drops
# that dead weight from the final binary without changing behavior.
if(MSVC)
  # cl.exe: /Gy (function-level linking) + /Gw (data COMDAT) enable the linker
  # to discard unused functions/data. link.exe: /OPT:REF removes unreferenced
  # symbols, /OPT:ICF folds identical COMDATs.
  target_compile_options(lib_onnx_proto PRIVATE
      $<$<COMPILE_LANGUAGE:C,CXX>:/Gy>
      $<$<COMPILE_LANGUAGE:C,CXX>:/Gw>)
  if(_lib_onnx_proto_type STREQUAL "SHARED")
    target_link_options(lib_onnx_proto PRIVATE /OPT:REF /OPT:ICF)
  endif()
else()
  target_compile_options(lib_onnx_proto PRIVATE
      $<$<COMPILE_LANGUAGE:C,CXX>:-ffunction-sections>
      $<$<COMPILE_LANGUAGE:C,CXX>:-fdata-sections>)
  # Only the SHARED variant is linked here; the STATIC archive is garbage
  # collected when the final consumer links it. --gc-sections (GNU/LLVM ld) and
  # -dead_strip (Apple ld) drop the unreferenced sections at link time.
  if(_lib_onnx_proto_type STREQUAL "SHARED")
    if(APPLE)
      target_link_options(lib_onnx_proto PRIVATE -Wl,-dead_strip)
    else()
      target_link_options(lib_onnx_proto PRIVATE -Wl,--gc-sections)
    endif()
  endif()
endif()

# Builds onnx_core library.
# This intermediate library sits between lib_onnx_proto and the higher-level
# manipulation / operator / kernel libraries. It owns the TensorType
# enumeration and the ToTypeString converter so that lib_onnx_op, lib_onnx_shape,
# and lib_onnx_manipulations can use them without depending on each other. It
# also provides the graph-node helpers (CollectExternalInputs, CollectNodeInputs,
# CollectRemainingInputs) so that lib_onnx_kernels depends on lib_onnx_core
# directly instead of lib_onnx_manipulations.
add_library(lib_onnx_core ${_onnx_light_lib_type} ${ONNX_LIGHT_HEADERS} ${ONNX_CORE_SOURCES})
if(_onnx_light_lib_type STREQUAL "SHARED" AND WIN32)
  set_target_properties(lib_onnx_core PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
target_include_directories(lib_onnx_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/onnx_light>
    $<BUILD_INTERFACE:${ROOT_INCLUDE_PATH}>
    $<BUILD_INTERFACE:${ONNX_PROTO_PATH}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/onnx_light>
)
set_property(TARGET lib_onnx_core PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(lib_onnx_core PUBLIC cxx_std_20)
target_link_libraries(lib_onnx_core PUBLIC lib_onnx_proto)

# Builds onnx_manipulations library.
# This library bundles the lightweight ModelProto/GraphProto manipulation
# helpers that are independent from the ONNX operator schemas: the textual
# parser / printer, the attribute / tensor proto helpers, the data-type name
# utilities, and the compose helpers. ``lib_onnx_lib`` (schemas, checker, shape
# inference, version converter) depends publicly on it. It depends on
# ``lib_onnx_core`` (and transitively on ``lib_onnx_proto``).
add_library(lib_onnx_manipulations ${_onnx_light_lib_type} ${ONNX_LIGHT_HEADERS} "${ONNX_COMMON_PATH}/status.cc" "${ONNX_COMMON_PATH}/assertions.cc" ${ONNX_MANIPULATIONS_SOURCES})
if(_onnx_light_lib_type STREQUAL "SHARED" AND WIN32)
  set_target_properties(lib_onnx_manipulations PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light" PREFIX "Files" FILES "${ONNX_COMMON_PATH}/status.cc" "${ONNX_COMMON_PATH}/assertions.cc" ${ONNX_MANIPULATIONS_SOURCES})
target_include_directories(lib_onnx_manipulations PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/onnx_light>
    $<BUILD_INTERFACE:${ROOT_INCLUDE_PATH}>
    $<BUILD_INTERFACE:${ONNX_PROTO_PATH}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/onnx_light>
)
set_property(TARGET lib_onnx_manipulations PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(lib_onnx_manipulations PUBLIC cxx_std_20)
target_link_libraries(lib_onnx_manipulations PUBLIC lib_onnx_core)

# Builds onnx_lib library.
add_library(lib_onnx_lib ${_onnx_light_lib_type} ${ONNX_LIGHT_HEADERS} ${ONNX_LIGHT_LIB_ALL_SOURCES})
if(_onnx_light_lib_type STREQUAL "SHARED" AND WIN32)
  set_target_properties(lib_onnx_lib PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light" PREFIX "Files" FILES ${ONNX_LIGHT_LIB_ALL_SOURCES})
target_include_directories(lib_onnx_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/onnx_light>
    $<BUILD_INTERFACE:${ROOT_INCLUDE_PATH}>
    $<BUILD_INTERFACE:${ONNX_PROTO_PATH}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/onnx_light>
)
set_property(TARGET lib_onnx_lib PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(lib_onnx_lib PUBLIC cxx_std_20)
target_link_libraries(lib_onnx_lib PUBLIC lib_onnx_proto lib_onnx_manipulations)
if(ONNX_ML)
  target_compile_definitions(lib_onnx_lib PUBLIC ONNX_ML=1)
endif()
# Precompiled headers: compile the heaviest shared headers once for all translation units.
onnx_light_target_precompile_headers(lib_onnx_lib PRIVATE
    <algorithm>
    <functional>
    <memory>
    <string>
    <unordered_map>
    <vector>
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_lib/defs/schema.h"
)

# Builds onnx_op library.
add_library(lib_onnx_op ${_onnx_light_lib_type} ${ONNX_LIGHT_HEADERS} ${ONNX_OP_SOURCES})
if(_onnx_light_lib_type STREQUAL "SHARED" AND WIN32)
  set_target_properties(lib_onnx_op PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light" PREFIX "Files" FILES ${ONNX_OP_SOURCES})
target_include_directories(lib_onnx_op PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/onnx_light>
    $<BUILD_INTERFACE:${ROOT_INCLUDE_PATH}>
    $<BUILD_INTERFACE:${ONNX_PROTO_PATH}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/onnx_light>
)
set_property(TARGET lib_onnx_op PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(lib_onnx_op PUBLIC cxx_std_20)
target_link_libraries(lib_onnx_op PUBLIC lib_onnx_core)

# Builds onnx_shapes library.
add_library(lib_onnx_shape ${_onnx_light_lib_type} ${ONNX_LIGHT_HEADERS} ${ONNX_SHAPE_SOURCES})
if(_onnx_light_lib_type STREQUAL "SHARED" AND WIN32)
  set_target_properties(lib_onnx_shape PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light" PREFIX "Files" FILES ${ONNX_SHAPE_SOURCES})
target_include_directories(lib_onnx_shape PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/onnx_light>
    $<BUILD_INTERFACE:${ROOT_INCLUDE_PATH}>
    $<BUILD_INTERFACE:${ONNX_PROTO_PATH}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/onnx_light>
)
set_property(TARGET lib_onnx_shape PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(lib_onnx_shape PUBLIC cxx_std_20)
target_link_libraries(lib_onnx_shape PUBLIC lib_onnx_core)

# Builds onnx_kernels library.
# This library is the runtime/operator-kernel layer extracted from the
# former monolithic ``lib_onnx_backend_test``. It bundles every ONNX
# operator kernel together with the minimal infrastructure they need
# (``TestCase``, ``OpsetId``, ``Tensor``, ``RunNodes``,
# ``RuntimeContext`` and deterministic pseudo-random helpers) and is
# exposed to Python through ``_onnxpykernels`` (and re-exported via
# ``_onnxpybackend`` transitively through ``lib_onnx_backend_test``). It
# depends on ``lib_onnx_core`` (which brings in ``lib_onnx_proto``) for the
# graph-manipulation helpers (``CollectExternalInputs``, ``CollectNodeInputs``)
# and proto types.
#
# Guarded by ``ONNX_LIGHT_BUILD_KERNELS`` so consumers that only need the
# schema/checker/shape-inference/version-converter libraries (for example
# the standalone ``examples/check_onnx_light_model`` checker example) can
# skip building this very large translation-unit set.
if(ONNX_LIGHT_BUILD_KERNELS)
add_library(lib_onnx_kernels ${_onnx_light_lib_type} ${ONNX_LIGHT_HEADERS} ${ONNX_KERNELS_SOURCES})
if(_onnx_light_lib_type STREQUAL "SHARED" AND WIN32)
  set_target_properties(lib_onnx_kernels PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light" PREFIX "Files" FILES ${ONNX_KERNELS_SOURCES})
target_include_directories(lib_onnx_kernels PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/onnx_light>
    $<BUILD_INTERFACE:${ROOT_INCLUDE_PATH}>
    $<BUILD_INTERFACE:${ONNX_PROTO_PATH}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/onnx_light>
)
set_property(TARGET lib_onnx_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(lib_onnx_kernels PUBLIC cxx_std_20)
target_link_libraries(lib_onnx_kernels PUBLIC lib_onnx_core)
# Same PCH rationale as ``lib_onnx_backend_test``: kernel translation
# units all pull in ``test_case.h`` (via ``kernel_context.h``) and the
# usual standard-library headers.
onnx_light_target_precompile_headers(lib_onnx_kernels PRIVATE
    "${ONNX_CORE_PATH}/backend_test/test_case.h"
    <algorithm>
    <cmath>
    <cstdint>
    <cstring>
    <limits>
    <stdexcept>
    <string>
    <utility>
    <vector>
)

# Builds onnx_backend_test library.
# This library is intentionally independent from lib_onnx_lib/lib_onnx_op
# but publicly depends on lib_onnx_proto because it uses proto types
# (e.g. ModelProto) in its headers and sources.
# It exposes a small infrastructure (struct Tensor, struct TestCase,
# expect()) and a registry of C++-implemented backend test node cases.
#
# The operator kernel implementations themselves live in the separate
# ``lib_onnx_kernels`` target (built from ``onnx_extensions/kernels/kernels``)
# which ``lib_onnx_backend_test`` PUBLIC-links so existing consumers
# transparently keep getting both the test infrastructure and the
# kernels.
add_library(lib_onnx_backend_test ${_onnx_light_lib_type} ${ONNX_LIGHT_HEADERS} ${ONNX_BACKEND_TEST_SOURCES})
if(_onnx_light_lib_type STREQUAL "SHARED" AND WIN32)
  set_target_properties(lib_onnx_backend_test PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light" PREFIX "Files" FILES ${ONNX_BACKEND_TEST_SOURCES})
target_include_directories(lib_onnx_backend_test PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/onnx_light>
    $<BUILD_INTERFACE:${ROOT_INCLUDE_PATH}>
    $<BUILD_INTERFACE:${ONNX_PROTO_PATH}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/onnx_light>
)
set_property(TARGET lib_onnx_backend_test PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(lib_onnx_backend_test PUBLIC cxx_std_20)
target_link_libraries(lib_onnx_backend_test PUBLIC lib_onnx_proto lib_onnx_kernels)

# Speed up the build of lib_onnx_backend_test by precompiling the small set
# of headers that virtually every one of the case source files in
# onnx_light/onnx_extensions/backend_test/cases/ ends up parsing. The common base is
# ``onnx_core/backend_test/test_case.h`` (which transitively brings in ``onnx.h``
# and ``simple_tensor.h``) and a handful of widely used C++ standard headers
# (``<vector>``, ``<string>``, ``<cstdint>``, ``<stdexcept>``). Without a
# PCH each translation unit re-parses tens of thousands of lines; with a
# PCH that work is done once for the target.
#
# This is a transparent optimization (no source changes required) and
# only affects targets that opt in. Other targets that link against
# ``lib_onnx_backend_test`` still see the regular public headers.
onnx_light_target_precompile_headers(lib_onnx_backend_test PRIVATE
    "${ONNX_CORE_PATH}/backend_test/expect.h"
    "${ONNX_CORE_PATH}/backend_test/test_case_registry.h"
    <algorithm>
    <cmath>
    <cstdint>
    <cstring>
    <stdexcept>
    <string>
    <utility>
    <vector>
)
endif() # ONNX_LIGHT_BUILD_KERNELS

# Builds onnx_gradient library.
# This library provides reverse-mode automatic differentiation for ONNX graphs.
# It is guarded by ONNX_LIGHT_BUILD_KERNELS (the "extended" build variant) so
# that the minimal schema/checker-only build remains lean.
if(ONNX_LIGHT_BUILD_KERNELS)
add_library(lib_onnx_gradient ${_onnx_light_lib_type} ${ONNX_LIGHT_HEADERS}
    ${ONNX_GRADIENT_SOURCES})
if(_onnx_light_lib_type STREQUAL "SHARED" AND WIN32)
  set_target_properties(lib_onnx_gradient PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light" PREFIX "Files"
    FILES ${ONNX_GRADIENT_SOURCES})
target_include_directories(lib_onnx_gradient PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/onnx_light>
    $<BUILD_INTERFACE:${ROOT_INCLUDE_PATH}>
    $<BUILD_INTERFACE:${ONNX_PROTO_PATH}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/onnx_light>
)
set_property(TARGET lib_onnx_gradient PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_features(lib_onnx_gradient PUBLIC cxx_std_20)
target_link_libraries(lib_onnx_gradient PUBLIC lib_onnx_core)
endif() # ONNX_LIGHT_BUILD_KERNELS (gradient)
# When ONNX_LIGHT_PROVIDE_ONNX_TARGETS=ON, expose CMake targets named exactly
# like the ones upstream onnx defines (`onnx`, `onnx_proto`) so a downstream
# project (for example onnxruntime configured with -Donnxruntime_USE_ONNX_LIGHT=ON)
# can add_subdirectory(onnx-light) via FetchContent and keep linking against
# `onnx` and `onnx_proto` unchanged.
#
# - `onnx_proto` aliases lib_onnx_proto (the protobuf-compatible message types
#   plus parser / serializer).
# - `onnx` is an INTERFACE aggregate that pulls in the ONNX C++ API
#   (lib_onnx_lib), the operator schemas (lib_onnx_op), shape inference
#   (lib_onnx_shape) and the graph manipulation helpers (lib_onnx_manipulations).
#   When ONNX_LIGHT_BUILD_KERNELS=ON the reference kernels and backend-test
#   registry are added as well so downstream backend tests can be implemented
#   with onnx-light.
# - `onnx_light`, `onnx::onnx` and `onnx::onnx_proto` are provided as additional
#   aliases matching the namespaced names used by find_package(onnx_light) and
#   find_package(onnx) consumers.
if(ONNX_LIGHT_PROVIDE_ONNX_TARGETS)
  if(NOT TARGET onnx_proto)
    add_library(onnx_proto ALIAS lib_onnx_proto)
  endif()

  if(NOT TARGET onnx)
    add_library(onnx_compat INTERFACE)
    target_link_libraries(onnx_compat INTERFACE
        lib_onnx_lib
        lib_onnx_op
        lib_onnx_shape
        lib_onnx_manipulations
        lib_onnx_proto)
    if(ONNX_LIGHT_BUILD_KERNELS)
      target_link_libraries(onnx_compat INTERFACE
          lib_onnx_kernels
          lib_onnx_backend_test)
    endif()
    # Expose an `onnx/`-rooted include tree of forwarding headers so downstream
    # code written against the standard onnx C++ package (which includes
    # `onnx/onnx_pb.h`, `onnx/defs/schema.h`, ...) compiles unchanged against
    # onnx-light without pulling in protobuf. The forwarding headers include the
    # real onnx-light headers (`onnx_lib/...`, `onnx_manipulations/...`), which
    # resolve via the PUBLIC include dirs propagated from the linked libraries.
    target_include_directories(onnx_compat INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_compat_include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_proto>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/onnx_compat_include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/onnx_proto>)
    add_library(onnx ALIAS onnx_compat)
  endif()

  # Namespaced aliases so consumers can reference the same names whether they
  # link the in-build targets or the installed find_package() targets.
  if(NOT TARGET onnx_light)
    add_library(onnx_light ALIAS onnx_compat)
  endif()
  if(NOT TARGET onnx::onnx)
    add_library(onnx::onnx ALIAS onnx_compat)
  endif()
  if(NOT TARGET onnx::onnx_proto)
    add_library(onnx::onnx_proto ALIAS lib_onnx_proto)
  endif()

  # Expose protobuf-compatible CMake targets so a downstream project that links
  # `protobuf::libprotobuf` / `protobuf::libprotobuf-lite` (for example
  # onnxruntime configured with -Donnxruntime_USE_ONNX_LIGHT=ON) can build
  # without fetching or linking the real protobuf. These INTERFACE targets pull
  # in lib_onnx_proto (which implements the RepeatedField / stream helpers used
  # by google_protobuf_compat.h) and add the `google/protobuf/...` forwarding
  # headers plus the compat shim to the include path.
  if(NOT TARGET protobuf::libprotobuf)
    add_library(onnx_light_protobuf_compat INTERFACE)
    target_link_libraries(onnx_light_protobuf_compat INTERFACE lib_onnx_proto)
    target_include_directories(onnx_light_protobuf_compat INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_compat_include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_proto>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/onnx_compat_include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/onnx_proto>)
    add_library(protobuf::libprotobuf ALIAS onnx_light_protobuf_compat)
  endif()
  if(NOT TARGET protobuf::libprotobuf-lite)
    add_library(protobuf::libprotobuf-lite ALIAS onnx_light_protobuf_compat)
  endif()
endif()


if(ONNX_LIGHT_BUILD_PYTHON)
  # _onnxpyprotoop: proto bindings + onnx_op schema bindings.
  # STABLE_ABI: emit an abi3-tagged extension on CPython >= 3.12 so a single
  # wheel covers all future minor versions. nanobind silently drops the flag on
  # older interpreters and on free-threaded builds (which use their own ABI).
  nanobind_add_module(_onnxpyprotoop STABLE_ABI ${ONNX_PY_PROTOOP_SOURCES})
  target_include_directories(_onnxpyprotoop PRIVATE "${ROOT_INCLUDE_PATH}" "${ONNX_PROTO_PATH}")
  target_link_libraries(_onnxpyprotoop PRIVATE lib_onnx_op lib_onnx_manipulations)

  # _onnxpyprotolib: onnx_lib bindings (defs/parser/checker/inliner/
  # shape/version_converter). This module does not register proto classes;
  # those are registered by _onnxpyprotoop.
  nanobind_add_module(_onnxpyprotolib STABLE_ABI ${ONNX_PY_PROTOLIB_SOURCES})
  target_include_directories(_onnxpyprotolib PRIVATE "${ROOT_INCLUDE_PATH}" "${ONNX_PROTO_PATH}")
  target_link_libraries(_onnxpyprotolib PRIVATE lib_onnx_lib)

  # _onnxpycore: core bindings (symbolic expressions and shape inference).
  # Split from _onnxpyproto* so the schema bindings and the shape-inference
  # bindings live in distinct extensions.
  nanobind_add_module(_onnxpycore STABLE_ABI ${ONNX_PY_CORE_SOURCES})
  target_include_directories(_onnxpycore PRIVATE "${ROOT_INCLUDE_PATH}" "${ONNX_PROTO_PATH}")
  target_link_libraries(_onnxpycore PRIVATE lib_onnx_shape)

  # _onnxpykernels: kernel helpers (deterministic pseudo-random helpers backing
  # ``onnx_backend_test``). Links lib_onnx_kernels only and does not depend on
  # the backend-test case registries.
  #
  # _onnxpykernels and _onnxpybackend are only built when the kernel runtime is
  # available (ONNX_LIGHT_BUILD_KERNELS=ON). In the reduced build they are
  # omitted and the Python package raises an explicit error when the reference
  # runtime or backend-test helpers are requested.
  if(ONNX_LIGHT_BUILD_KERNELS)
  nanobind_add_module(_onnxpykernels STABLE_ABI ${ONNX_PY_KERNELS_SOURCES})
  target_include_directories(_onnxpykernels PRIVATE "${ROOT_INCLUDE_PATH}" "${ONNX_PROTO_PATH}")
  target_link_libraries(_onnxpykernels PRIVATE lib_onnx_kernels)

  # _onnxpybackend: backend-test case utilities. Links lib_onnx_backend_test
  # (which publicly depends on lib_onnx_kernels and lib_onnx_proto so that
  # ModelProto can be returned by reference to Python and resolved against the
  # single nb::class_<ModelProto> registered by _onnxpyprotoop).
  # When the libraries are built SHARED (the default for Python builds), the
  # cross-library references between ``lib_onnx_backend_test`` and
  # ``lib_onnx_kernels`` are resolved through the dynamic linker so no
  # special ordering is required; in the pure C++ STATIC variant we repeat
  # ``lib_onnx_backend_test`` after ``lib_onnx_kernels`` to break the
  # circular reference between the two static archives.
  nanobind_add_module(_onnxpybackend STABLE_ABI ${ONNX_PY_BACKEND_SOURCES})
  target_include_directories(_onnxpybackend PRIVATE "${ROOT_INCLUDE_PATH}" "${ONNX_PROTO_PATH}")
  if(_onnx_light_lib_type STREQUAL "SHARED")
    target_link_libraries(_onnxpybackend PRIVATE lib_onnx_backend_test lib_onnx_kernels)
  else()
    target_link_libraries(_onnxpybackend PRIVATE lib_onnx_backend_test lib_onnx_kernels lib_onnx_backend_test)
  endif()
  endif() # ONNX_LIGHT_BUILD_KERNELS

  # _onnxpygradient: gradient (automatic differentiation) bindings.
  # Only built in the extended variant (ONNX_LIGHT_BUILD_KERNELS=ON) because the
  # gradient library links lib_onnx_gradient which depends on lib_onnx_proto.
  if(ONNX_LIGHT_BUILD_KERNELS)
  nanobind_add_module(_onnxpygradient STABLE_ABI ${ONNX_PY_GRADIENT_SOURCES})
  target_include_directories(_onnxpygradient PRIVATE "${ROOT_INCLUDE_PATH}" "${ONNX_PROTO_PATH}")
  target_link_libraries(_onnxpygradient PRIVATE lib_onnx_gradient)
  endif() # ONNX_LIGHT_BUILD_KERNELS (gradient)
  # directory (onnx_light/onnx_py/). Set an $ORIGIN rpath so the dynamic loader
  # picks the colocated libraries at import time without LD_LIBRARY_PATH. The
  # rpath must also be set on the lib_onnx_*.so libraries themselves: when
  # auditwheel repairs the wheel it statically walks the DT_NEEDED entries of
  # every .so and refuses to repair if a referenced library (e.g.
  # ``liblib_onnx_proto.so`` referenced by ``liblib_onnx_kernels.so``) cannot
  # be located via the consuming library's own rpath.
  set(_onnxpy_rpath_targets _onnxpyprotoop _onnxpyprotolib _onnxpycore)
  if(ONNX_LIGHT_BUILD_KERNELS)
    list(APPEND _onnxpy_rpath_targets _onnxpykernels _onnxpybackend _onnxpygradient)
  endif()
  if(_lib_onnx_proto_type STREQUAL "SHARED")
    list(APPEND _onnxpy_rpath_targets lib_onnx_proto)
  endif()
  if(_onnx_light_lib_type STREQUAL "SHARED")
    list(APPEND _onnxpy_rpath_targets lib_onnx_lib lib_onnx_manipulations lib_onnx_op lib_onnx_shape)
    if(ONNX_LIGHT_BUILD_KERNELS)
      list(APPEND _onnxpy_rpath_targets lib_onnx_kernels lib_onnx_backend_test lib_onnx_gradient)
    endif()
  endif()
  if((_onnx_light_lib_type STREQUAL "SHARED" OR _lib_onnx_proto_type STREQUAL "SHARED") AND (UNIX AND NOT APPLE))
    set_target_properties(${_onnxpy_rpath_targets} PROPERTIES
        BUILD_RPATH "\$ORIGIN"
        INSTALL_RPATH "\$ORIGIN")
  elseif((_onnx_light_lib_type STREQUAL "SHARED" OR _lib_onnx_proto_type STREQUAL "SHARED") AND APPLE)
    set_target_properties(${_onnxpy_rpath_targets} PROPERTIES
        BUILD_RPATH "@loader_path"
        INSTALL_RPATH "@loader_path")
  endif()
endif()

# Builds test
if(ONNX_LIGHT_BUILD_TESTS)

  file(GLOB_RECURSE ONNX_CPP_PROTO_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/onnx_proto/test_*.cc")
  file(GLOB_RECURSE ONNX_CPP_CORE_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/onnx_core/test_*.cc")
  file(GLOB_RECURSE ONNX_CPP_LIB_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/onnx_lib/test_*.cc")
  file(GLOB_RECURSE ONNX_CPP_OP_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/onnx_op/test_*.cc")
  file(GLOB_RECURSE ONNX_CPP_SHAPE_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/onnx_extensions/shapes/test_*.cc")
  file(GLOB_RECURSE ONNX_CPP_KERNELS_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/onnx_extensions/kernels/test_*.cc")
  file(GLOB_RECURSE ONNX_CPP_BACKEND_TEST_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/onnx_extensions/backend_test/test_*.cc")
  file(GLOB_RECURSE ONNX_CPP_GRADIENT_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/onnx_extensions/gradient/test_*.cc")
  set(ONNX_CPP_PROTO_STREAM_CLASS_PRINT_TEST
      "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/onnx_proto/test_stream_class_print.cc")
  list(REMOVE_ITEM ONNX_CPP_PROTO_TEST_SOURCES "${ONNX_CPP_PROTO_STREAM_CLASS_PRINT_TEST}")
  set(ONNX_CPP_TEST_SOURCES
      ${ONNX_CPP_PROTO_TEST_SOURCES}
      ${ONNX_CPP_CORE_TEST_SOURCES}
      ${ONNX_CPP_LIB_TEST_SOURCES}
      ${ONNX_CPP_OP_TEST_SOURCES}
      ${ONNX_CPP_SHAPE_TEST_SOURCES}
  )
  if(ONNX_LIGHT_BUILD_KERNELS)
    list(APPEND ONNX_CPP_TEST_SOURCES
        ${ONNX_CPP_KERNELS_TEST_SOURCES}
        ${ONNX_CPP_BACKEND_TEST_TEST_SOURCES}
        ${ONNX_CPP_GRADIENT_TEST_SOURCES}
    )
  endif()
  set(ONNX_CPP_ALL_TEST_SOURCES ${ONNX_CPP_TEST_SOURCES} ${ONNX_CPP_PROTO_STREAM_CLASS_PRINT_TEST})

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
  if(ONNX_LIGHT_BUILD_KERNELS)
    target_link_libraries(test_onnx_light PRIVATE lib_onnx_lib lib_onnx_op lib_onnx_shape lib_onnx_kernels lib_onnx_backend_test lib_onnx_gradient GTest::gtest_main)
  else()
    target_link_libraries(test_onnx_light PRIVATE lib_onnx_lib lib_onnx_op lib_onnx_shape GTest::gtest_main)
  endif()
  # Precompile the heaviest headers shared across virtually every translation
  # unit in unittests/cc. ``<gtest/gtest.h>`` is by far the dominant cost
  # (~109/110 test files pull it in) and the standard headers below are
  # included by the majority of those files; a PCH turns that work into a
  # one-shot cost for the whole target.
  #
  # NOTE: do NOT add project headers (e.g. ``test_case.h``) here. Adding a
  # project header to the PCH forces every translation unit in this target
  # to rebuild whenever that header changes, defeating the purpose of a
  # shared PCH; keep it limited to stable, rarely-changing standard headers.
  # Keeping only standard-library headers also avoids forcing a particular
  # include order onto individual translation units.
  onnx_light_target_precompile_headers(test_onnx_light PRIVATE
      <gtest/gtest.h>
      <cstdint>
      <stdexcept>
      <string>
      <vector>
  )
  gtest_discover_tests(test_onnx_light DISCOVERY_TIMEOUT 120)

  add_executable(test_stream_class_print "${ONNX_CPP_PROTO_STREAM_CLASS_PRINT_TEST}")
  target_include_directories(test_stream_class_print PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light"
    "${ROOT_INCLUDE_PATH}"
    "${ONNX_PROTO_PATH}"
    "${ONNX_COMMON_PATH}"
  )
  target_link_libraries(test_stream_class_print PRIVATE lib_onnx_proto GTest::gtest_main)
  gtest_discover_tests(test_stream_class_print DISCOVERY_TIMEOUT 120)

  add_custom_target(onnx_cpp_tests SOURCES ${ONNX_CPP_ALL_TEST_SOURCES})
  source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/onnx_proto" PREFIX "Source Files/cc/onnx_proto"
               FILES ${ONNX_CPP_PROTO_TEST_SOURCES})
  source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/onnx_lib" PREFIX "Source Files/cc/onnx_lib"
               FILES ${ONNX_CPP_LIB_TEST_SOURCES})
  source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/onnx_op" PREFIX "Source Files/cc/onnx_op"
               FILES ${ONNX_CPP_OP_TEST_SOURCES})
  source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/onnx_extensions/shapes" PREFIX "Source Files/cc/onnx_shapes"
               FILES ${ONNX_CPP_SHAPE_TEST_SOURCES})
  if(ONNX_LIGHT_BUILD_KERNELS)
    source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/onnx_extensions/kernels" PREFIX "Source Files/cc/onnx_kernels"
                 FILES ${ONNX_CPP_KERNELS_TEST_SOURCES})
    source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/onnx_extensions/backend_test" PREFIX "Source Files/cc/onnx_backend_test"
                 FILES ${ONNX_CPP_BACKEND_TEST_TEST_SOURCES})
    source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/unittests/cc/onnx_extensions/gradient" PREFIX "Source Files/cc/onnx_gradient"
                 FILES ${ONNX_CPP_GRADIENT_TEST_SOURCES})
  endif()
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

    # bench_graph_builder profiles GraphBuilder construction from a model file,
    # so it additionally needs the operator schemas (lib_onnx_op) and the shape
    # functions (lib_onnx_shape) used by the builder's incremental shape
    # inference. GraphBuilder can execute kernels but must assume they may be
    # missing, so this benchmark deliberately does NOT link lib_onnx_kernels
    # (nor lib_onnx_backend_test, which pulls it in).
    if(benchmark_target STREQUAL "bench_graph_builder")
      target_link_libraries(${benchmark_target} PRIVATE lib_onnx_shape lib_onnx_op)
    endif()
  endforeach()
endif()

# Builds the libFuzzer-instrumented C++ harnesses under fuzz/.
#
# Each fuzz/fuzz_<name>.cc file exposes the standard libFuzzer entry
# point ``LLVMFuzzerTestOneInput``. The companion fuzz/make_seed_corpus.cc
# is built as a plain executable (no fuzzer instrumentation) and emits
# seed inputs for OSS-Fuzz / local libFuzzer runs.
#
# Requires Clang because libFuzzer (-fsanitize=fuzzer,...) ships with
# Clang. ``-fsanitize=fuzzer,address`` is the standard OSS-Fuzz
# instrumentation; users wanting MSan/UBSan can set
# ``ONNX_LIGHT_FUZZER_SANITIZERS`` to override the sanitizer list (the
# ``fuzzer`` sanitizer is always added automatically).
if(ONNX_LIGHT_BUILD_FUZZERS)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
      "ONNX_LIGHT_BUILD_FUZZERS=ON requires Clang (CMAKE_CXX_COMPILER_ID is '${CMAKE_CXX_COMPILER_ID}'). "
      "Re-run cmake with CC=clang CXX=clang++.")
  endif()

  set(ONNX_LIGHT_FUZZER_SANITIZERS "address"
      CACHE STRING "Comma-separated sanitizer list for fuzz harnesses (in addition to 'fuzzer').")

  file(GLOB ONNX_LIGHT_FUZZ_SOURCES CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/fuzz/fuzz_*.cc")

  foreach(fuzz_source IN LISTS ONNX_LIGHT_FUZZ_SOURCES)
    get_filename_component(fuzz_target "${fuzz_source}" NAME_WE)
    add_executable(${fuzz_target} "${fuzz_source}")
    target_include_directories(${fuzz_target} PRIVATE
        "${ROOT_INCLUDE_PATH}"
        "${ONNX_PROTO_PATH}"
    )
    target_link_libraries(${fuzz_target} PRIVATE lib_onnx_lib lib_onnx_shape)
    target_compile_options(${fuzz_target} PRIVATE
        "-fsanitize=fuzzer-no-link,${ONNX_LIGHT_FUZZER_SANITIZERS}")
    target_link_options(${fuzz_target} PRIVATE
        "-fsanitize=fuzzer,${ONNX_LIGHT_FUZZER_SANITIZERS}")
    set_property(TARGET ${fuzz_target} PROPERTY FOLDER "Fuzzers")
  endforeach()

  # Seed-corpus generator: a plain (non-fuzzer-instrumented) executable
  # that writes serialized ModelProtos into the requested directories so
  # OSS-Fuzz / libFuzzer can use them as starting inputs.
  add_executable(make_seed_corpus "${CMAKE_CURRENT_SOURCE_DIR}/fuzz/make_seed_corpus.cc")
  target_include_directories(make_seed_corpus PRIVATE
      "${ROOT_INCLUDE_PATH}"
      "${ONNX_PROTO_PATH}"
  )
  target_link_libraries(make_seed_corpus PRIVATE lib_onnx_lib)
  set_property(TARGET make_seed_corpus PROPERTY FOLDER "Fuzzers")
endif()

if(ONNX_LIGHT_BUILD_PYTHON)
  # Install the extensions into the package directory. When the core C++
  # libraries are built SHARED (the default in Python builds), install them
  # alongside so the $ORIGIN rpath above resolves them without
  # LD_LIBRARY_PATH and the compiled code is shared across all extensions
  # instead of being duplicated inside each .so.
  install(TARGETS _onnxpyprotoop _onnxpyprotolib _onnxpycore LIBRARY DESTINATION onnx_light/onnx_py)
  if(ONNX_LIGHT_BUILD_KERNELS)
    install(TARGETS _onnxpykernels _onnxpybackend _onnxpygradient LIBRARY DESTINATION onnx_light/onnx_py)
  endif()
  if(_lib_onnx_proto_type STREQUAL "SHARED")
    install(TARGETS lib_onnx_proto
        LIBRARY DESTINATION onnx_light/onnx_py
        RUNTIME DESTINATION onnx_light/onnx_py)
  endif()
  if(_onnx_light_lib_type STREQUAL "SHARED")
    install(TARGETS
        lib_onnx_core
        lib_onnx_manipulations
        lib_onnx_lib
        lib_onnx_op
        lib_onnx_shape
        LIBRARY DESTINATION onnx_light/onnx_py
        RUNTIME DESTINATION onnx_light/onnx_py)
    if(ONNX_LIGHT_BUILD_KERNELS)
      install(TARGETS
          lib_onnx_kernels
          lib_onnx_backend_test
          lib_onnx_gradient
          LIBRARY DESTINATION onnx_light/onnx_py
          RUNTIME DESTINATION onnx_light/onnx_py)
    endif()
  endif()
endif()

if(ONNX_LIGHT_INSTALL)
  include(CMakePackageConfigHelpers)

  # Install the static library and export its target.
  # Splitting into two install(TARGETS) calls keeps the same exported
  # ``onnx_lightTargets`` set when ONNX_LIGHT_BUILD_KERNELS=OFF while
  # avoiding references to targets that were never created.
  install(TARGETS lib_onnx_proto lib_onnx_core lib_onnx_manipulations lib_onnx_lib lib_onnx_op lib_onnx_shape
    EXPORT onnx_lightTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  )
  if(ONNX_LIGHT_BUILD_KERNELS)
    install(TARGETS lib_onnx_kernels lib_onnx_backend_test lib_onnx_gradient
      EXPORT onnx_lightTargets
      ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )
  endif()

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

  # Install public headers from onnx_light/onnx_lib/shape_inference
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_lib/shape_inference/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/onnx_lib/shape_inference"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )

  # Install public headers from onnx_light/onnx_lib/version_converter
  install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/onnx_light/onnx_lib/version_converter/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/onnx_lib/version_converter"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )

  # Install public headers from onnx_light/onnx_manipulations
  install(DIRECTORY "${ONNX_MANIPULATIONS_PATH}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/onnx_manipulations"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )

  # Install public headers from onnx_light/onnx_op
  install(DIRECTORY "${ONNX_OP_PATH}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/onnx_op"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )

  # Install public headers from onnx_light/onnx_extensions/shapes
  install(DIRECTORY "${ONNX_SHAPE_PATH}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/onnx_extensions/shapes"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )

  # Install public headers from onnx_light/onnx_core
  install(DIRECTORY "${ONNX_CORE_PATH}/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/onnx_core"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )

  # Install public headers from onnx_light/onnx_extensions/backend_test
  if(ONNX_LIGHT_BUILD_KERNELS)
    install(DIRECTORY "${ONNX_BACKEND_TEST_PATH}/"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/onnx_extensions/backend_test"
      FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )

    # Install public headers from onnx_light/onnx_extensions/kernels
    install(DIRECTORY "${ONNX_KERNELS_PATH}/"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/onnx_extensions/kernels"
      FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )
  endif()

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

# Apply OpenSSF compiler hardening flags to onnx-light targets when
# ONNX_HARDENING=ON. When the option is OFF the helper is a no-op so the
# loop has no effect.
set(_onnx_light_hardening_targets
    lib_onnx_proto
    lib_onnx_lib
    lib_onnx_op
    lib_onnx_shape
)
if(ONNX_LIGHT_BUILD_KERNELS)
  list(APPEND _onnx_light_hardening_targets lib_onnx_kernels lib_onnx_backend_test)
endif()
if(ONNX_LIGHT_BUILD_PYTHON)
  list(APPEND _onnx_light_hardening_targets
      _onnxpyprotoop _onnxpyprotolib _onnxpycore)
  if(ONNX_LIGHT_BUILD_KERNELS)
    list(APPEND _onnx_light_hardening_targets _onnxpykernels _onnxpybackend)
  endif()
endif()
if(ONNX_LIGHT_BUILD_TESTS)
  list(APPEND _onnx_light_hardening_targets test_onnx_light_helpers test_onnx_light)
endif()
foreach(_t IN LISTS _onnx_light_hardening_targets)
  if(TARGET ${_t})
    onnx_light_apply_hardening(${_t})
  endif()
endforeach()
