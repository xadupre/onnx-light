// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/test_case.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Abs — y = |x| (since opset 13 for the floating-point variant we use).
// Uses a small, fully deterministic input so this library does not depend
// on a PRNG.
// ---------------------------------------------------------------------------
void RegisterAbsCases(std::vector<TestCase> &registry) {
  NodeProto node;
  node.set_op_type("Abs");
  node.add_input("x");
  node.add_output("y");

  const OpsetId opset = DefaultOpset(13);
  Tensor x = Tensor::FromFloat("", {2, 3}, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});
  Tensor y = kernel::Abs(kernel::KernelContext(opset))(x);

  Expect(node, {x}, {y}, "test_cc_abs", {opset}, "backend-test", registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
