// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "../onnx_proto/_onnxpy.h"

namespace nb = nanobind;

NB_MODULE(_onnxpyprotoop, m) {
  m.doc() = "onnx proto and onnx_op (schema) bindings from python without protobuf but "
            "using the same format";
  AddOnnxPyProto(m);
  AddOnnxPyOp(m);
}
