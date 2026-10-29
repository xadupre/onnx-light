cmake_minimum_required(VERSION 4.0)
project(onnx_proto VERSION 0.2.3 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(ONNX_PROTO_BUILD_SHARED "Build the proto library as a shared library." OFF)
option(ONNX_PROTO_INSTALL "Install the proto library, headers and CMake package." ON)
option(ONNX_PROTO_BUILD_TESTS "Build the native proto unit tests and consumer checks." OFF)
option(ONNX_PROTO_WERROR "Treat proto compiler warnings as errors." OFF)

if(WIN32 AND NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()

if(NOT TARGET onnx_threads::lib_onnx_threads)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../onnx_threads/CMakeLists.txt")
    set(ONNX_THREADS_BUILD_SHARED ${ONNX_PROTO_BUILD_SHARED})
    set(ONNX_THREADS_INSTALL ${ONNX_PROTO_INSTALL})
    set(ONNX_THREADS_BUILD_TESTS ${ONNX_PROTO_BUILD_TESTS})
    set(ONNX_THREADS_WERROR ${ONNX_PROTO_WERROR})
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../onnx_threads"
        "${CMAKE_CURRENT_BINARY_DIR}/onnx_threads")
  else()
    find_package(onnx_threads CONFIG REQUIRED)
  endif()
endif()

find_package(OpenSSL QUIET)

file(GLOB ONNX_PROTO_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")
file(GLOB_RECURSE ONNX_PROTO_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")
file(GLOB_RECURSE ONNX_PROTO_HELPER_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/helpers/include/*.h")
set(ONNX_BLAKE3_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include/onnx_proto/blake3")
set(ONNX_BLAKE3_SOURCES
    "${ONNX_BLAKE3_PATH}/blake3.c"
    "${ONNX_BLAKE3_PATH}/blake3_dispatch.c"
    "${ONNX_BLAKE3_PATH}/blake3_portable.c"
    "${ONNX_BLAKE3_PATH}/blake3_join.cc"
    "${ONNX_BLAKE3_PATH}/blake3_hash.cc")
set_source_files_properties(${ONNX_BLAKE3_SOURCES} PROPERTIES COMPILE_DEFINITIONS
    "BLAKE3_USE_TBB;BLAKE3_NO_SSE2;BLAKE3_NO_SSE41;BLAKE3_NO_AVX2;BLAKE3_NO_AVX512;BLAKE3_USE_NEON=0")
if(ONNX_PROTO_WERROR)
  if(MSVC)
    set_property(SOURCE ${ONNX_BLAKE3_SOURCES} APPEND PROPERTY COMPILE_OPTIONS "/WX-")
  else()
    set_property(SOURCE ${ONNX_BLAKE3_SOURCES} APPEND PROPERTY COMPILE_OPTIONS "-Wno-error")
  endif()
endif()

if(ONNX_PROTO_BUILD_SHARED)
  set(_onnx_proto_type SHARED)
else()
  set(_onnx_proto_type STATIC)
endif()
add_library(lib_onnx_proto ${_onnx_proto_type}
    ${ONNX_PROTO_HEADERS} ${ONNX_PROTO_HELPER_HEADERS}
    ${ONNX_PROTO_SOURCES} ${ONNX_BLAKE3_SOURCES}
    helpers/onnx_light_helpers.cc)
add_library(onnx_proto::lib_onnx_proto ALIAS lib_onnx_proto)

target_compile_features(lib_onnx_proto PUBLIC cxx_std_20)
target_link_libraries(lib_onnx_proto PUBLIC onnx_threads::lib_onnx_threads)
if(ONNX_PROTO_WERROR)
  if(MSVC)
    target_compile_options(lib_onnx_proto PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:/WX>)
  else()
    target_compile_options(lib_onnx_proto PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-Werror>)
  endif()
endif()
set_target_properties(lib_onnx_proto PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(lib_onnx_proto PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/onnx_proto>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/helpers/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/onnx_light>)
if(OpenSSL_FOUND)
  target_compile_definitions(lib_onnx_proto PUBLIC ONNX_LIGHT_HAS_OPENSSL)
  target_link_libraries(lib_onnx_proto PUBLIC OpenSSL::Crypto)
endif()

# Keep validation code small and discard unused generated proto methods.
if(MSVC)
  set_source_files_properties(src/onnx_verify.cc src/onnx_dlpack.cc
      PROPERTIES COMPILE_OPTIONS "/Os")
  target_compile_options(lib_onnx_proto PRIVATE
      $<$<COMPILE_LANGUAGE:C,CXX>:/Gy>
      $<$<COMPILE_LANGUAGE:C,CXX>:/Gw>)
  if(ONNX_PROTO_BUILD_SHARED)
    set_target_properties(lib_onnx_proto PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    target_link_options(lib_onnx_proto PRIVATE /OPT:REF /OPT:ICF)
  endif()
else()
  set_source_files_properties(src/onnx_verify.cc src/onnx_dlpack.cc
      PROPERTIES COMPILE_OPTIONS "-Os")
  target_compile_options(lib_onnx_proto PRIVATE
      $<$<COMPILE_LANGUAGE:C,CXX>:-ffunction-sections>
      $<$<COMPILE_LANGUAGE:C,CXX>:-fdata-sections>)
  if(ONNX_PROTO_BUILD_SHARED)
    set_target_properties(lib_onnx_proto PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES)
    if(APPLE)
      set_property(TARGET lib_onnx_proto APPEND PROPERTY INSTALL_RPATH "@loader_path")
      target_link_options(lib_onnx_proto PRIVATE -Wl,-dead_strip)
    else()
      set_property(TARGET lib_onnx_proto APPEND PROPERTY INSTALL_RPATH "$ORIGIN")
      target_link_options(lib_onnx_proto PRIVATE -Wl,--gc-sections)
    endif()
  endif()
endif()

if(ONNX_PROTO_INSTALL)
  install(TARGETS lib_onnx_proto EXPORT onnx_protoTargets
      ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  install(DIRECTORY include/onnx_proto/
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/onnx_proto"
      FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
      PATTERN "external_data_internal.h" EXCLUDE)
  # Forward flat include spellings to the same files used by namespaced includes.
  foreach(_header IN LISTS ONNX_PROTO_HEADERS)
    file(RELATIVE_PATH _header_relative
        "${CMAKE_CURRENT_SOURCE_DIR}/include/onnx_proto" "${_header}")
    if(_header_relative STREQUAL "external_data_internal.h")
      continue()
    endif()
    set(_forwarding_header "${CMAKE_CURRENT_BINARY_DIR}/forwarding_headers/${_header_relative}")
    file(WRITE "${_forwarding_header}"
        "#pragma once\n#include \"onnx_proto/${_header_relative}\"\n")
    get_filename_component(_header_dir "${_header_relative}" DIRECTORY)
    install(FILES "${_forwarding_header}"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light/${_header_dir}")
  endforeach()
  install(DIRECTORY helpers/include/
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/onnx_light"
      FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
  install(FILES include/onnx_proto/dlpack/LICENSE
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/onnx_light/dlpack")
  install(FILES include/onnx_proto/blake3/LICENSE
      DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/onnx_light/blake3")
  install(EXPORT onnx_protoTargets FILE onnx_protoTargets.cmake
      NAMESPACE onnx_proto::
      DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnx_proto")
  configure_package_config_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/onnx_protoConfig.cmake.in"
      "${CMAKE_CURRENT_BINARY_DIR}/onnx_protoConfig.cmake"
      INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnx_proto")
  write_basic_package_version_file(
      "${CMAKE_CURRENT_BINARY_DIR}/onnx_protoConfigVersion.cmake"
      VERSION ${PROJECT_VERSION} COMPATIBILITY AnyNewerVersion)
  install(FILES
      "${CMAKE_CURRENT_BINARY_DIR}/onnx_protoConfig.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/onnx_protoConfigVersion.cmake"
      DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnx_proto")
endif()

if(ONNX_PROTO_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
