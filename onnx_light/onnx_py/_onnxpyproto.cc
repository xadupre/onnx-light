// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "../onnx_proto/_onnxpy.h"

namespace nb = nanobind;

NB_MODULE(_onnxpyproto, m) {
  m.doc() = "onnx proto, defs and op (schema) bindings from python without protobuf "
            "but using the same format";
  AddOnnxPyProto(m);
  AddOnnxPyLib(m);
  AddOnnxPyOp(m);
}
