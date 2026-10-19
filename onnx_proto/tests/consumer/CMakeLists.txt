cmake_minimum_required(VERSION 3.15)
project(onnx_proto_consumer LANGUAGES CXX)

enable_testing()
if(NOT TARGET onnx_proto::lib_onnx_proto)
  find_package(onnx_proto CONFIG REQUIRED)
endif()
if(TARGET onnx_light::lib_onnx_proto)
  message(FATAL_ERROR "The removed onnx_light::lib_onnx_proto target must not be exposed.")
endif()
add_executable(proto_consumer test_onnx_proto.cc)
target_link_libraries(proto_consumer PRIVATE onnx_proto::lib_onnx_proto)
add_test(NAME proto_roundtrip COMMAND proto_consumer)
set_tests_properties(proto_roundtrip PROPERTIES TIMEOUT 30)

get_target_property(_proto_imported onnx_proto::lib_onnx_proto IMPORTED)
get_target_property(_proto_type onnx_proto::lib_onnx_proto TYPE)
get_target_property(_threads_type onnx_threads::lib_onnx_threads TYPE)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND _proto_imported
    AND _proto_type STREQUAL "SHARED_LIBRARY" AND _threads_type STREQUAL "SHARED_LIBRARY")
  find_program(_readelf NAMES readelf llvm-readelf)
  if(NOT _readelf)
    message(FATAL_ERROR "readelf is required to check indirect threading dependencies.")
  endif()
  add_executable(proto_indirect_consumer test_indirect_dependency.cc)
  target_link_libraries(proto_indirect_consumer PRIVATE onnx_proto::lib_onnx_proto)
  target_link_options(proto_indirect_consumer PRIVATE -Wl,--as-needed -Wl,--enable-new-dtags)
  add_test(NAME proto_indirect_dependency
      COMMAND ${CMAKE_COMMAND}
          "-DREADELF=${_readelf}"
          "-DCONSUMER=$<TARGET_FILE:proto_indirect_consumer>"
          "-DPROTO=$<TARGET_FILE:onnx_proto::lib_onnx_proto>"
          "-DPROTO_NAME=$<TARGET_FILE_NAME:onnx_proto::lib_onnx_proto>"
          "-DTHREADS_NAME=$<TARGET_FILE_NAME:onnx_threads::lib_onnx_threads>"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/check_indirect_dependency.cmake")
  set_tests_properties(proto_indirect_dependency PROPERTIES TIMEOUT 30)
endif()

if(WIN32)
  get_target_property(_threads_type onnx_threads::lib_onnx_threads TYPE)
  if(_threads_type STREQUAL "SHARED_LIBRARY")
    add_custom_command(TARGET proto_consumer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:onnx_threads::lib_onnx_threads>"
            "$<TARGET_FILE_DIR:proto_consumer>"
        VERBATIM)
  endif()
  get_target_property(_onnx_proto_imported onnx_proto::lib_onnx_proto IMPORTED)
  get_target_property(_onnx_proto_type onnx_proto::lib_onnx_proto TYPE)
  if(_onnx_proto_imported AND _onnx_proto_type STREQUAL "SHARED_LIBRARY")
    add_custom_command(TARGET proto_consumer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:onnx_proto::lib_onnx_proto>"
            "$<TARGET_FILE_DIR:proto_consumer>"
        VERBATIM)
  endif()
endif()
