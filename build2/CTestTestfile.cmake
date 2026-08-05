# CMake generated Testfile for 
# Source directory: /home/runner/work/onnx-light/onnx-light
# Build directory: /home/runner/work/onnx-light/onnx-light/build2
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
include("/home/runner/work/onnx-light/onnx-light/build2/test_onnx_light[1]_include.cmake")
include("/home/runner/work/onnx-light/onnx-light/build2/test_stream_class_print[1]_include.cmake")
add_test(test_onnx_light_helpers "/home/runner/work/onnx-light/onnx-light/build2/test_onnx_light_helpers")
set_tests_properties(test_onnx_light_helpers PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/work/onnx-light/onnx-light/CMakeLists.txt;801;add_test;/home/runner/work/onnx-light/onnx-light/CMakeLists.txt;0;")
subdirs("_deps/googletest-build")
