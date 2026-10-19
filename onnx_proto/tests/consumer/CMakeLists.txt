cmake_minimum_required(VERSION 3.15)
project(onnx_proto_consumer LANGUAGES CXX)

enable_testing()
if(NOT TARGET onnx_proto::lib_onnx_proto)
  find_package(onnx_proto CONFIG REQUIRED)
endif()
add_executable(proto_consumer test_onnx_proto.cc)
target_link_libraries(proto_consumer PRIVATE onnx_proto::lib_onnx_proto)
add_test(NAME proto_roundtrip COMMAND proto_consumer)
set_tests_properties(proto_roundtrip PROPERTIES TIMEOUT 30)

# Also exercise the target used by existing onnx-light consumers.
add_executable(proto_consumer_legacy test_onnx_proto.cc)
target_link_libraries(proto_consumer_legacy PRIVATE onnx_light::lib_onnx_proto)
add_test(NAME proto_roundtrip_legacy COMMAND proto_consumer_legacy)
set_tests_properties(proto_roundtrip_legacy PROPERTIES TIMEOUT 30)
