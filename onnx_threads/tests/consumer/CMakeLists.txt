cmake_minimum_required(VERSION 3.15)
project(onnx_threads_consumer LANGUAGES CXX)

enable_testing()
if(NOT TARGET onnx_threads::lib_onnx_threads)
  find_package(onnx_threads CONFIG REQUIRED)
endif()
add_executable(threads_consumer main.cc)
target_link_libraries(threads_consumer PRIVATE onnx_threads::lib_onnx_threads)
add_test(NAME threads_consumer COMMAND threads_consumer)
set_tests_properties(threads_consumer PROPERTIES TIMEOUT 30)
if(WIN32)
  get_target_property(_threads_type onnx_threads::lib_onnx_threads TYPE)
  if(_threads_type STREQUAL "SHARED_LIBRARY")
    add_custom_command(TARGET threads_consumer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:onnx_threads::lib_onnx_threads>"
            "$<TARGET_FILE_DIR:threads_consumer>"
        VERBATIM)
  endif()
endif()
