cmake_minimum_required(VERSION 3.15)
project(onnx_threads VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(Threads REQUIRED)

option(ONNX_THREADS_BUILD_SHARED "Build the threading library as a shared library." OFF)
option(ONNX_THREADS_INSTALL "Install the threading library and CMake package." ON)
option(ONNX_THREADS_BUILD_TESTS "Build standalone threading and consumer tests." OFF)
option(ONNX_THREADS_WERROR "Treat threading compiler warnings as errors." OFF)

if(ONNX_THREADS_BUILD_SHARED)
  set(_onnx_threads_type SHARED)
else()
  set(_onnx_threads_type STATIC)
endif()
file(GLOB ONNX_THREADS_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")
add_library(lib_onnx_threads ${_onnx_threads_type} ${ONNX_THREADS_SOURCES})
add_library(onnx_threads::lib_onnx_threads ALIAS lib_onnx_threads)
target_compile_features(lib_onnx_threads PUBLIC cxx_std_20)
target_link_libraries(lib_onnx_threads PUBLIC Threads::Threads)
target_include_directories(lib_onnx_threads PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
set_target_properties(lib_onnx_threads PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(ONNX_THREADS_BUILD_SHARED)
  target_compile_definitions(lib_onnx_threads PUBLIC ONNX_THREADS_SHARED)
endif()
if(MSVC)
  target_compile_options(lib_onnx_threads PRIVATE /W4 /wd4251)
  if(ONNX_THREADS_WERROR)
    target_compile_options(lib_onnx_threads PRIVATE /WX)
  endif()
else()
  target_compile_options(lib_onnx_threads PRIVATE -Wall -Wextra)
  if(ONNX_THREADS_WERROR)
    target_compile_options(lib_onnx_threads PRIVATE -Werror)
  endif()
endif()

if(ONNX_THREADS_INSTALL)
  install(TARGETS lib_onnx_threads EXPORT onnx_threadsTargets
      ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
  install(DIRECTORY include/onnx_threads
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
  install(EXPORT onnx_threadsTargets NAMESPACE onnx_threads::
      DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnx_threads")
  configure_package_config_file(
      cmake/onnx_threadsConfig.cmake.in
      "${CMAKE_CURRENT_BINARY_DIR}/onnx_threadsConfig.cmake"
      INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnx_threads")
  write_basic_package_version_file(
      "${CMAKE_CURRENT_BINARY_DIR}/onnx_threadsConfigVersion.cmake"
      VERSION ${PROJECT_VERSION} COMPATIBILITY AnyNewerVersion)
  install(FILES
      "${CMAKE_CURRENT_BINARY_DIR}/onnx_threadsConfig.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/onnx_threadsConfigVersion.cmake"
      DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/onnx_threads")
endif()

if(ONNX_THREADS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
