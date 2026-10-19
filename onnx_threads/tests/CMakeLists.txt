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

add_executable(test_onnx_threads test_threads.cc)
target_link_libraries(test_onnx_threads PRIVATE onnx_threads::lib_onnx_threads GTest::gtest_main)
if(WIN32 AND ONNX_THREADS_BUILD_SHARED)
  add_custom_command(TARGET test_onnx_threads POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:lib_onnx_threads>" "$<TARGET_FILE_DIR:test_onnx_threads>"
      VERBATIM)
endif()
add_test(NAME standalone_threads COMMAND test_onnx_threads)
set_tests_properties(standalone_threads PROPERTIES TIMEOUT 120)
add_subdirectory(consumer)

if(ONNX_THREADS_INSTALL AND CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  add_test(NAME threads_installed_consumer
      COMMAND ${CMAKE_COMMAND}
          "-DBUILD_DIR=${CMAKE_BINARY_DIR}"
          "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/consumer"
          "-DGENERATOR=${CMAKE_GENERATOR}"
          "-DCONFIG=$<CONFIG>"
          "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/installed_consumer.cmake")
  set_tests_properties(threads_installed_consumer PROPERTIES TIMEOUT 120)
endif()
