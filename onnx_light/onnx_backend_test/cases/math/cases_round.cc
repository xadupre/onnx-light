// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Round — y = round_half_to_even(x) (since opset 11, widened to bfloat16 in
// opset 22). Registers both a deterministic ``test_cc_round`` case and the
// upstream ONNX backend test case (``test_round``) mirrored from
// ``onnx.backend.test.case.node.round.Round``.
// ---------------------------------------------------------------------------
void RegisterRoundCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};
  const kernel::Round round_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Round");
    node.add_input("x");
    node.add_output("y");

    // Includes halves to exercise the round-half-to-even rule used by ONNX.
    Tensor x = Tensor::FromFloat("", {2, 3}, {0.9f, 2.5f, 2.3f, 1.5f, -4.5f, -2.5f});
    Tensor y = round_kernel(x);

    Expect(node, {x}, {y}, "test_cc_round", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test case for the ``Round`` operator (mirrors the
  // ``onnx.backend.test.case.node.round.Round`` Python class).
  {
    NodeProto node;
    node.set_op_type("Round");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {15},
                                 {0.1f, 0.5f, 0.9f, 1.2f, 1.5f, 1.8f, 2.3f, 2.5f, 2.7f, -1.1f,
                                  -1.5f, -1.9f, -2.2f, -2.5f, -2.8f});
    Tensor y = Tensor::FromFloat("", {15},
                                 {0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f, -1.0f,
                                  -2.0f, -2.0f, -2.0f, -2.0f, -3.0f});
    Expect(node, {x}, {y}, "test_round", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
