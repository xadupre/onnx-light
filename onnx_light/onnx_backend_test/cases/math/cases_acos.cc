// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/test_case.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Acos — y = acos(x) (since opset 7, widened to bfloat16 in opset 22).
// Uses a small, fully deterministic input in [-1, 1] so this library does
// not depend on a PRNG.
// ---------------------------------------------------------------------------
void RegisterAcosCases(std::vector<TestCase> &registry) {
  NodeProto node;
  node.set_op_type("Acos");
  node.add_input("x");
  node.add_output("y");

  const OpsetId opset = DefaultOpset(22);
  Tensor x = Tensor::FromFloat("", {2, 3}, {-1.0f, -0.5f, 0.0f, 0.25f, 0.5f, 1.0f});
  Tensor y = kernel::Acos(kernel::KernelContext(opset))(x);

  Expect(node, {x}, {y}, "test_cc_acos", {opset}, "backend-test", registry);
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
