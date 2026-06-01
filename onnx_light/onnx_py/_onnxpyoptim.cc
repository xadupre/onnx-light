// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "../onnx_proto/_onnxpy.h"

namespace nb = nanobind;

NB_MODULE(_onnxpyoptim, m) {
  m.doc() = "onnx optim bindings from python: symbolic dimension expressions and "
            "shape inference helpers (operating on the same proto format).";
  AddOnnxPyExpressions(m);
}
