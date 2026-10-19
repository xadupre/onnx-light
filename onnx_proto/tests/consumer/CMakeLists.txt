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

if(WIN32)
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
