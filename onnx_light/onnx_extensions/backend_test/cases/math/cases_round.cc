// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Round — y = round_half_to_even(x) (since opset 11, widened to bfloat16 in
// opset 22). Registers both a deterministic ``test_cc_round`` case and the
// upstream ONNX backend test case (``test_round``) mirrored from
// ``onnx.backend.test.case.node.round.Round``.
// ---------------------------------------------------------------------------
void RegisterRoundCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Round round_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    ExpectBenchmarkUnaryFloat("Round", round_kernel, "test_cc_round_benchmark", opset, registry);
    return;
  }

  {
    NodeProto node;
    node.set_op_type("Round");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_round", {opset}, [=]() -> IoData {
      // Includes halves to exercise the round-half-to-even rule used by ONNX.
      Tensor x = Tensor::FromFloat("", {2, 3}, {0.9f, 2.5f, 2.3f, 1.5f, -4.5f, -2.5f});
      Tensor y = round_kernel(x);

      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Upstream ONNX backend test case for the ``Round`` operator (mirrors the
  // ``onnx.backend.test.case.node.round.Round`` Python class).
  {
    NodeProto node;
    node.set_op_type("Round");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_round", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {15},
                                   {0.1f, 0.5f, 0.9f, 1.2f, 1.5f, 1.8f, 2.3f, 2.5f, 2.7f, -1.1f,
                                    -1.5f, -1.9f, -2.2f, -2.5f, -2.8f});
      Tensor y = Tensor::FromFloat("", {15},
                                   {0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f, -1.0f,
                                    -2.0f, -2.0f, -2.0f, -2.0f, -3.0f});
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
