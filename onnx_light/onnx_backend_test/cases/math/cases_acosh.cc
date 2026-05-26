// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/test_case.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Acosh — y = acosh(x) (since opset 9, widened to bfloat16 in opset 22).
// Uses a small, fully deterministic input in [1, +inf) so this library
// does not depend on a PRNG.
// ---------------------------------------------------------------------------
void RegisterAcoshCases(std::vector<TestCase> &registry) {
  NodeProto node;
  node.set_op_type("Acosh");
  node.add_input("x");
  node.add_output("y");

  const OpsetId opset = DefaultOpset(22);
  Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 1.25f, 1.5f, 2.0f, 3.5f, 10.0f});
  Tensor y = kernel::Acosh(kernel::KernelContext(opset))(x);

  Expect(node, {x}, {y}, "test_cc_acosh", {opset}, "backend-test", registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
