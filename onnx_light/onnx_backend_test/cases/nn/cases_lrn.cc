// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// LRN — Local Response Normalization across the channel dimension.
// Output shape and dtype match the input.
//
// Cases:
//   * test_cc_lrn — explicit alpha/beta/bias/size on a 5x5x5x5 deterministic
//     input (mirrors upstream ``test_lrn``).
//   * test_cc_lrn_default — only ``size`` specified, default alpha/beta/bias
//     (mirrors upstream ``test_lrn_default``).
// ---------------------------------------------------------------------------
void RegisterLRNCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::LRN kernel{ctx};

  // Build a deterministic 5x5x5x5 input shared between both cases.
  std::vector<float> x_data(5 * 5 * 5 * 5);
  for (size_t i = 0; i < x_data.size(); ++i) {
    // Map indices to a varied but reproducible signed float in [-1, 1).
    x_data[i] = static_cast<float>((static_cast<int64_t>(i) % 13) - 6) / 6.0f;
  }
  Tensor x = Tensor::FromFloat("", {5, 5, 5, 5}, x_data);

  // Explicit attributes — alpha=0.0002, beta=0.5, bias=2.0, size=3.
  {
    NodeProto node;
    node.set_op_type("LRN");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<float>(node, "alpha", 0.0002f);
    AddAttribute<float>(node, "beta", 0.5f);
    AddAttribute<float>(node, "bias", 2.0f);
    AddAttribute<int64_t>(node, "size", 3);

    Tensor y = kernel(x, /*size=*/3, /*alpha=*/0.0002f, /*beta=*/0.5f, /*bias=*/2.0f);
    Expect(node, {x}, {y}, "test_cc_lrn", {opset}, "backend-test", registry);
  }

  // Default attributes — only ``size`` specified.
  {
    NodeProto node;
    node.set_op_type("LRN");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<int64_t>(node, "size", 3);

    Tensor y = kernel(x, /*size=*/3);
    Expect(node, {x}, {y}, "test_cc_lrn_default", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
