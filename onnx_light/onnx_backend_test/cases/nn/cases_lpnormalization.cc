// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// LpNormalization — normalizes ``input`` so that each slice along the chosen
// axis has unit Lp norm. Output shape and dtype match the input.
//
// Cases:
//   * test_cc_lpnormalization_default — default axis=-1 and p=2 on a
//     (2, 2, 3) FLOAT input (mirrors upstream ``test_lpnormalization_default``).
//   * test_cc_lpnormalization_axis0_p1 — axis=0, p=1 on the same input.
// ---------------------------------------------------------------------------
void RegisterLpNormalizationCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};
  const kernel::LpNormalization kernel{ctx};

  Tensor x = Tensor::FromFloat(
      "", {2, 2, 3}, {1.0f, 2.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 5.0f, 5.0f, 6.0f, 8.0f, 0.0f});

  // Default attributes — axis=-1, p=2.
  {
    NodeProto node;
    node.set_op_type("LpNormalization");
    node.add_input("x");
    node.add_output("y");

    Tensor y = kernel(x);
    Expect(node, {x}, {y}, "test_cc_lpnormalization_default", {opset}, "backend-test", registry);
  }

  // Explicit attributes — axis=0, p=1.
  {
    NodeProto node;
    node.set_op_type("LpNormalization");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<int64_t>(node, "axis", 0);
    AddAttribute<int64_t>(node, "p", 1);

    Tensor y = kernel(x, /*axis=*/0, /*p=*/1);
    Expect(node, {x}, {y}, "test_cc_lpnormalization_axis0_p1", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
