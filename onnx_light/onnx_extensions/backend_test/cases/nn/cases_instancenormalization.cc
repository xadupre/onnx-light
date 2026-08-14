// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/nn/include_nn_cases.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// InstanceNormalization — Y = scale * (X - mean) / sqrt(var + epsilon) + B
// where mean / var are computed per-instance per-channel over the spatial
// axes. The cases below mirror the upstream ``test_instancenorm_*`` ONNX
// reference cases at opset 22 (the ai.onnx version supported by onnx-light).
// ---------------------------------------------------------------------------
void RegisterInstanceNormalizationCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::InstanceNormalization instancenorm_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("InstanceNormalization");
    node.add_input("x");
    node.add_input("s");
    node.add_input("bias");
    node.add_output("y");

    constexpr int64_t N = 1;
    constexpr int64_t C = 64;
    constexpr int64_t H = 128;
    constexpr int64_t W = 128;
    constexpr int64_t x_count = N * C * H * W;
    Expect(registry, std::move(node), "test_cc_instancenorm_example_benchmark", {opset},
           {x_count, C, C}, {x_count}, [instancenorm_kernel]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, {N, C, H, W}, 2001);
             Tensor scale = RandnTensor(DataType::FLOAT, {C}, 2002);
             Tensor bias = RandnTensor(DataType::FLOAT, {C}, 2003);
             Tensor y = instancenorm_kernel(x, scale, bias);
             return IoData{{std::move(x), std::move(scale), std::move(bias)}, {std::move(y)}};
           });
    return;
  }

  // ``instancenorm_example``: 1x2x1x3 input (N=1, C=2, H=1, W=3) with
  // per-channel scale/bias. Mirrors the upstream Python reference.
  {
    NodeProto node;
    node.set_op_type("InstanceNormalization");
    node.add_input("x");
    node.add_input("s");
    node.add_input("bias");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_instancenorm_example", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {1, 2, 1, 3}, {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f});
      Tensor scale = Tensor::FromFloat("", {2}, {1.0f, 1.5f});
      Tensor bias = Tensor::FromFloat("", {2}, {0.0f, 1.0f});

      Tensor y = instancenorm_kernel(x, scale, bias);

      return IoData{{std::move(x), std::move(scale), std::move(bias)}, {std::move(y)}};
    });
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
    Expect(registry, std::move(node), "test_cc_instancenorm_epsilon", {opset}, [=]() -> IoData {
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

      return IoData{{std::move(x), std::move(scale), std::move(bias)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
