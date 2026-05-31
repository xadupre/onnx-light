// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "../onnx_proto/_onnxpy.h"

namespace nb = nanobind;

NB_MODULE(_onnxbackend, m) {
  m.doc() =
      "onnx_light C++ backend bindings: operator schemas, optimisations, backend helpers and "
      "test-case utilities.";

  AddOnnxPyLib(m);
  AddOnnxPyExpressions(m);
  AddOnnxPyBackend(m);
  AddOnnxPyBackendTest(m);
  AddOnnxPyOp(m);
}
