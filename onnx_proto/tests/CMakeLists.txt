if(NOT TARGET GTest::gtest_main)
  find_package(GTest QUIET)
  if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
endif()
if(COMMAND onnx_light_disable_werror)
  onnx_light_disable_werror(gtest gtest_main)
endif()
include(GoogleTest)

file(GLOB ONNX_PROTO_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc")
list(REMOVE_ITEM ONNX_PROTO_TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/test_onnx_light_helpers.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_stream_class_print.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_onnx_dlpack.cc")
add_executable(test_onnx_proto ${ONNX_PROTO_TEST_SOURCES})
target_link_libraries(test_onnx_proto PRIVATE lib_onnx_proto GTest::gtest_main)
gtest_discover_tests(test_onnx_proto DISCOVERY_TIMEOUT 120 DISCOVERY_MODE PRE_TEST)

add_executable(test_stream_class_print test_stream_class_print.cc)
target_link_libraries(test_stream_class_print PRIVATE lib_onnx_proto GTest::gtest_main)
gtest_discover_tests(test_stream_class_print DISCOVERY_TIMEOUT 120 DISCOVERY_MODE PRE_TEST)

add_executable(test_onnx_dlpack test_onnx_dlpack.cc)
target_link_libraries(test_onnx_dlpack PRIVATE lib_onnx_proto GTest::gtest_main)
gtest_discover_tests(test_onnx_dlpack DISCOVERY_TIMEOUT 120 DISCOVERY_MODE PRE_TEST)

add_executable(test_onnx_light_helpers test_onnx_light_helpers.cc)
target_link_libraries(test_onnx_light_helpers PRIVATE lib_onnx_proto)
add_test(NAME test_onnx_light_helpers COMMAND test_onnx_light_helpers)
set_tests_properties(test_onnx_light_helpers PROPERTIES TIMEOUT 120)

add_subdirectory(consumer)
