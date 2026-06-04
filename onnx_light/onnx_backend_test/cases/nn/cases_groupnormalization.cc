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
// GroupNormalization (opset 21) — partitions the channel dim into
// ``num_groups`` equal-sized groups, normalizes each (n, group) block over
// its channels and spatial dims, then applies a per-channel affine. The
// cases below mirror the upstream ``test_group_normalization_*`` ONNX
// reference cases.
// ---------------------------------------------------------------------------
void RegisterGroupNormalizationCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(21);
  const kernel::KernelContext ctx{opset};
  const kernel::GroupNormalization groupnorm_kernel{ctx};

  // ``group_normalization_example``: N=3, C=4, num_groups=2, 2x2 spatial.
  // Inputs are built deterministically (instead of np.random) so the case
  // is reproducible across platforms.
  {
    NodeProto node;
    node.set_op_type("GroupNormalization");
    node.add_input("x");
    node.add_input("scale");
    node.add_input("bias");
    node.add_output("y");
    AddAttribute<int64_t>(node, "num_groups", 2);

    const int64_t N = 3;
    const int64_t C = 4;
    const int64_t H = 2;
    const int64_t W = 2;
    const int64_t total = N * C * H * W;
    std::vector<float> x_data(static_cast<size_t>(total));
    for (int64_t i = 0; i < total; ++i) {
      x_data[static_cast<size_t>(i)] = static_cast<float>(i) * 0.1f - 2.0f;
    }
    Tensor x = Tensor::FromFloat("", {N, C, H, W}, x_data);
    Tensor scale = Tensor::FromFloat("", {C}, {0.5f, 1.0f, 1.5f, 2.0f});
    Tensor bias = Tensor::FromFloat("", {C}, {-0.25f, 0.0f, 0.25f, 0.5f});

    Tensor y = groupnorm_kernel(x, scale, bias, /*num_groups=*/2);

    Expect(node, {x, scale, bias}, {y}, "test_cc_group_normalization_example", {opset},
           "backend-test", registry);
  }

  // ``group_normalization_epsilon``: same shape with epsilon=1e-2.
  {
    NodeProto node;
    node.set_op_type("GroupNormalization");
    node.add_input("x");
    node.add_input("scale");
    node.add_input("bias");
    node.add_output("y");
    AddAttribute<int64_t>(node, "num_groups", 2);
    AddAttribute<float>(node, "epsilon", 1e-2f);

    const int64_t N = 3;
    const int64_t C = 4;
    const int64_t H = 2;
    const int64_t W = 2;
    const int64_t total = N * C * H * W;
    std::vector<float> x_data(static_cast<size_t>(total));
    for (int64_t i = 0; i < total; ++i) {
      x_data[static_cast<size_t>(i)] = static_cast<float>(i) * 0.1f - 2.0f;
    }
    Tensor x = Tensor::FromFloat("", {N, C, H, W}, x_data);
    Tensor scale = Tensor::FromFloat("", {C}, {0.5f, 1.0f, 1.5f, 2.0f});
    Tensor bias = Tensor::FromFloat("", {C}, {-0.25f, 0.0f, 0.25f, 0.5f});

    Tensor y = groupnorm_kernel(x, scale, bias, /*num_groups=*/2, 1e-2f);

    Expect(node, {x, scale, bias}, {y}, "test_cc_group_normalization_epsilon", {opset},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
