// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "../onnx_proto/_onnxpy.h"

namespace nb = nanobind;

NB_MODULE(_onnxbackend, m) {
  m.doc() = "onnx_light backend bindings: deterministic pseudo-random helpers and "
            "ONNX backend-test case utilities.";

  AddOnnxPyBackend(m);
  AddOnnxPyBackendTest(m);
}
