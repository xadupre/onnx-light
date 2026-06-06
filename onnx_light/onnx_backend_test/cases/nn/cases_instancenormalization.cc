// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_kernels/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

// ---------------------------------------------------------------------------
// InstanceNormalization — Y = scale * (X - mean) / sqrt(var + epsilon) + B
// where mean / var are computed per-instance per-channel over the spatial
// axes. The cases below mirror the upstream ``test_instancenorm_*`` ONNX
// reference cases at opset 22 (the ai.onnx version supported by onnx-light).
// ---------------------------------------------------------------------------
void RegisterInstanceNormalizationCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};
  const kernel::InstanceNormalization instancenorm_kernel{ctx};

  // ``instancenorm_example``: 1x2x1x3 input (N=1, C=2, H=1, W=3) with
  // per-channel scale/bias. Mirrors the upstream Python reference.
  {
    NodeProto node;
    node.set_op_type("InstanceNormalization");
    node.add_input("x");
    node.add_input("s");
    node.add_input("bias");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {1, 2, 1, 3}, {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f});
    Tensor scale = Tensor::FromFloat("", {2}, {1.0f, 1.5f});
    Tensor bias = Tensor::FromFloat("", {2}, {0.0f, 1.0f});

    Tensor y = instancenorm_kernel(x, scale, bias);

    Expect(node, {x, scale, bias}, {y}, "test_cc_instancenorm_example", {opset}, "backend-test",
           registry);
  }

  // ``instancenorm_epsilon``: 2x3x4x5 input with epsilon = 1e-2. Inputs are
  // built deterministically so the case is reproducible across platforms.
  {
    NodeProto node;
    node.set_op_type("InstanceNormalization");
    node.add_input("x");
    node.add_input("s");
    node.add_input("bias");
    node.add_output("y");
    AddAttribute<float>(node, "epsilon", 1e-2f);

    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t H = 4;
    const int64_t W = 5;
    const int64_t total = N * C * H * W;
    std::vector<float> x_data(static_cast<size_t>(total));
    for (int64_t i = 0; i < total; ++i) {
      x_data[static_cast<size_t>(i)] = static_cast<float>(i) * 0.1f - 1.0f;
    }
    Tensor x = Tensor::FromFloat("", {N, C, H, W}, x_data);
    Tensor scale = Tensor::FromFloat("", {C}, {0.5f, 1.0f, 1.5f});
    Tensor bias = Tensor::FromFloat("", {C}, {-0.25f, 0.25f, 0.75f});

    Tensor y = instancenorm_kernel(x, scale, bias, 1e-2f);

    Expect(node, {x, scale, bias}, {y}, "test_cc_instancenorm_epsilon", {opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
