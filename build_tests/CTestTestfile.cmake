# CMake generated Testfile for 
# Source directory: /home/runner/work/onnx-light/onnx-light
# Build directory: /home/runner/work/onnx-light/onnx-light/build_tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
include("/home/runner/work/onnx-light/onnx-light/build_tests/test_onnx_protos[1]_include.cmake")
include("/home/runner/work/onnx-light/onnx-light/build_tests/test_onnx_common[1]_include.cmake")
include("/home/runner/work/onnx-light/onnx-light/build_tests/test_onnx_defs[1]_include.cmake")
include("/home/runner/work/onnx-light/onnx-light/build_tests/test_onnx_helper[1]_include.cmake")
include("/home/runner/work/onnx-light/onnx-light/build_tests/test_onnx_threads[1]_include.cmake")
add_test(test_onnx_light_helpers "/home/runner/work/onnx-light/onnx-light/build_tests/test_onnx_light_helpers")
set_tests_properties(test_onnx_light_helpers PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/work/onnx-light/onnx-light/CMakeLists.txt;122;add_test;/home/runner/work/onnx-light/onnx-light/CMakeLists.txt;0;")
subdirs("_deps/googletest-build")
