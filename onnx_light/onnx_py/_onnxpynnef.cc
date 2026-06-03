// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "../onnx_proto/_onnxpy.h"

namespace nb = nanobind;

NB_MODULE(_onnxpynnef, m) {
  m.doc() = "onnx_light NNEF exporter bindings (C++ implementation).";
  AddOnnxPyNnef(m);
}
