// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "../onnx_proto/_onnxpy.h"

namespace nb = nanobind;

NB_MODULE(_onnxpyprotolib, m) {
  m.doc() = "onnx lib bindings (defs/parser/checker/inliner/shape/version_converter) from "
            "python without protobuf but using the same format";
  AddOnnxPyLib(m);
}
