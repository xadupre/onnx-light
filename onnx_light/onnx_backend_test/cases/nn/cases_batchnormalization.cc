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
// BatchNormalization — Y = (X - mean) / sqrt(var + epsilon) * scale + B (and,
// in opset 15, two optional running_mean / running_var outputs in training
// mode, which the kernel does not produce). The reference cases below mirror
// a subset of the ``test_batchnorm_*`` ONNX reference cases for the
// inference path on FLOAT inputs at opset 15:
//
//   * ``test_cc_batchnorm_example`` — 1x2x1x3 input, channel-wise scale/bias.
//   * ``test_cc_batchnorm_epsilon`` — 2x3x4x5 input with epsilon = 1e-2.
// ---------------------------------------------------------------------------
void RegisterBatchNormalizationCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(15);
  const kernel::KernelContext ctx{opset};
  const kernel::BatchNormalization batchnorm_kernel{ctx};

  // ``batchnorm_example``: a tiny 1x2x1x3 input where C=2 lets us see the
  // per-channel scale / bias / mean / variance applied independently.
  {
    NodeProto node;
    node.set_op_type("BatchNormalization");
    node.add_input("x");
    node.add_input("scale");
    node.add_input("B");
    node.add_input("input_mean");
    node.add_input("input_var");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {1, 2, 1, 3}, {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f});
    Tensor scale = Tensor::FromFloat("", {2}, {1.0f, 1.5f});
    Tensor bias = Tensor::FromFloat("", {2}, {0.0f, 1.0f});
    Tensor mean = Tensor::FromFloat("", {2}, {0.0f, 3.0f});
    Tensor var = Tensor::FromFloat("", {2}, {1.0f, 1.5f});

    Tensor y = batchnorm_kernel(x, scale, bias, mean, var);

    Expect(node, {x, scale, bias, mean, var}, {y}, "test_cc_batchnorm_example", {opset},
           "backend-test", registry);
  }

  // ``batchnorm_epsilon``: identical to the example but with a larger epsilon
  // and a larger 2x3x4x5 input. Inputs are constructed deterministically so
  // the case is reproducible without depending on ``np.random``.
  {
    NodeProto node;
    node.set_op_type("BatchNormalization");
    node.add_input("x");
    node.add_input("scale");
    node.add_input("B");
    node.add_input("input_mean");
    node.add_input("input_var");
    node.add_output("y");
    AddAttribute<float>(node, "epsilon", 1e-2f);

    const int64_t N = 2;
    const int64_t C = 3;
    const int64_t H = 4;
    const int64_t W = 5;
    const int64_t total = N * C * H * W;
    std::vector<float> x_data(static_cast<size_t>(total));
    for (int64_t i = 0; i < total; ++i) {
      // A simple deterministic ramp keeps the case reproducible across
      // platforms and easy to inspect.
      x_data[static_cast<size_t>(i)] = static_cast<float>(i) * 0.1f - 1.0f;
    }
    Tensor x = Tensor::FromFloat("", {N, C, H, W}, x_data);
    Tensor scale = Tensor::FromFloat("", {C}, {1.0f, 1.5f, 2.0f});
    Tensor bias = Tensor::FromFloat("", {C}, {0.0f, -0.5f, 0.5f});
    Tensor mean = Tensor::FromFloat("", {C}, {0.5f, 1.0f, -0.25f});
    Tensor var = Tensor::FromFloat("", {C}, {0.25f, 0.5f, 1.0f});

    Tensor y = batchnorm_kernel(x, scale, bias, mean, var, /*epsilon=*/1e-2f);

    Expect(node, {x, scale, bias, mean, var}, {y}, "test_cc_batchnorm_epsilon", {opset},
           "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
