// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/test_case.h"

#include <cmath>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {
namespace case {

// Builds an OperatorSetIdProto for the default ai.onnx domain.
OperatorSetIdProto DefaultOpset(int64_t version) {
  OperatorSetIdProto osid;
  osid.set_domain("");
  osid.set_version(version);
  return osid;
}

}  // namespace case

// ---------------------------------------------------------------------------
// Abs — y = |x| (since opset 13 for the floating-point variant we use).
// Mirrors onnx_light/backend/test/case/node/abs.py but uses a small, fully
// deterministic input so this library does not depend on a PRNG.
// ---------------------------------------------------------------------------
void RegisterAbsCases(std::vector<TestCase> &registry) {
  NodeProto node;
  node.set_op_type("Abs");
  node.add_input("x");
  node.add_output("y");

  Tensor x = {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f};
  Tensor y = kernel::Abs(x);

  Expect(node, {Tensor::FromFloat("x", {2, 3}, x)}, {Tensor::FromFloat("y", {2, 3}, y)},
         "test_cc_abs", {DefaultOpset(13)}, "backend-test", registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
