// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "_onnxpyprotoop.h"

namespace nb = nanobind;

NB_MODULE(_onnxpyprotoop, m) {
  m.doc() = "onnx proto and onnx_op (schema) bindings from python without protobuf but "
            "using the same format";
  m.def("_cpp_build_info", []() {
    nb::dict info;
    info["nanobind_version"] = NB_TOSTRING(NB_VERSION_MAJOR) "." NB_TOSTRING(
        NB_VERSION_MINOR) "." NB_TOSTRING(NB_VERSION_PATCH);
    info["nanobind_platform_abi"] = NB_PLATFORM_ABI_TAG;
    info["python_stable_abi"] =
#if defined(Py_LIMITED_API)
        "ON";
#else
        "OFF";
#endif
    info["compiler_id"] = ONNX_LIGHT_COMPILER_ID;
    info["compiler_version"] = ONNX_LIGHT_COMPILER_VERSION;
    info["cxx_standard"] = NB_TOSTRING(ONNX_LIGHT_CXX_STANDARD);
    return info;
  });
  AddOnnxPyProto(m);
  AddOnnxPyOp(m);
}