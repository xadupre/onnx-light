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
// BatchNormalization — Y = (X - mean) / sqrt(var + epsilon) * scale + B (and,
// in opset 15, two optional running_mean / running_var outputs in training
// mode). The reference cases below mirror a subset of the
// ``test_batchnorm_*`` ONNX reference cases on FLOAT inputs at opset 15:
//
//   * ``test_cc_batchnorm_example`` — 1x2x1x3 input, channel-wise scale/bias.
//   * ``test_cc_batchnorm_epsilon`` — 2x3x4x5 input with epsilon = 1e-2.
//   * ``test_cc_batchnorm_example_training_mode`` — 1x2x1x3 input with
//     ``training_mode = 1`` and the running mean / variance outputs.
//   * ``test_cc_batchnorm_epsilon_training_mode`` — 2x3x4x5 input with
//     ``training_mode = 1`` and epsilon = 1e-2.
// ---------------------------------------------------------------------------
void RegisterBatchNormalizationCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(15);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::BatchNormalization batchnorm_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("BatchNormalization");
    node.add_input("x");
    node.add_input("scale");
    node.add_input("B");
    node.add_input("input_mean");
    node.add_input("input_var");
    node.add_output("y");

    constexpr int64_t N = 1;
    constexpr int64_t C = 64;
    constexpr int64_t H = 128;
    constexpr int64_t W = 128;
    constexpr int64_t x_count = N * C * H * W;
    Expect(registry, std::move(node), "test_cc_batchnorm_example_benchmark", {opset},
           {x_count, C, C, C, C}, {x_count}, [batchnorm_kernel]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, {N, C, H, W}, 1201);
             Tensor scale = RandnTensor(DataType::FLOAT, {C}, 1202);
             Tensor bias = RandnTensor(DataType::FLOAT, {C}, 1203);
             Tensor mean = RandnTensor(DataType::FLOAT, {C}, 1204);
             Tensor var =
                 Tensor::FromFloat("", {C}, std::vector<float>(static_cast<size_t>(C), 1.0f));
             Tensor y = batchnorm_kernel(x, scale, bias, mean, var);
             return IoData{
                 {std::move(x), std::move(scale), std::move(bias), std::move(mean), std::move(var)},
                 {std::move(y)}};
           });
    return;
  }

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
    Expect(registry, std::move(node), "test_cc_batchnorm_example", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {1, 2, 1, 3}, {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f});
      Tensor scale = Tensor::FromFloat("", {2}, {1.0f, 1.5f});
      Tensor bias = Tensor::FromFloat("", {2}, {0.0f, 1.0f});
      Tensor mean = Tensor::FromFloat("", {2}, {0.0f, 3.0f});
      Tensor var = Tensor::FromFloat("", {2}, {1.0f, 1.5f});

      Tensor y = batchnorm_kernel(x, scale, bias, mean, var);

      return IoData{
          {std::move(x), std::move(scale), std::move(bias), std::move(mean), std::move(var)},
          {std::move(y)}};
    });
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
    Expect(registry, std::move(node), "test_cc_batchnorm_epsilon", {opset}, [=]() -> IoData {
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

      return IoData{
          {std::move(x), std::move(scale), std::move(bias), std::move(mean), std::move(var)},
          {std::move(y)}};
    });
  }

  // ``batchnorm_example_training_mode``: opset-15 training mode (training_mode
  // = 1) on the same 1x2x1x3 input. The kernel normalizes ``X`` with the
  // per-channel batch statistics and emits the updated running mean / variance
  // (using the default momentum = 0.9).
  {
    NodeProto node;
    node.set_op_type("BatchNormalization");
    node.add_input("x");
    node.add_input("scale");
    node.add_input("B");
    node.add_input("input_mean");
    node.add_input("input_var");
    node.add_output("y");
    node.add_output("output_mean");
    node.add_output("output_var");
    AddAttribute<int64_t>(node, "training_mode", 1);
    Expect(registry, std::move(node), "test_cc_batchnorm_example_training_mode", {opset},
           [=]() -> IoData {
             Tensor x = Tensor::FromFloat("", {1, 2, 1, 3}, {-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f});
             Tensor scale = Tensor::FromFloat("", {2}, {1.0f, 1.5f});
             Tensor bias = Tensor::FromFloat("", {2}, {0.0f, 1.0f});
             Tensor mean = Tensor::FromFloat("", {2}, {0.0f, 3.0f});
             Tensor var = Tensor::FromFloat("", {2}, {1.0f, 1.5f});

             auto [y, output_mean, output_var] =
                 batchnorm_kernel.TrainingForward(x, scale, bias, mean, var);

             return IoData{
                 {std::move(x), std::move(scale), std::move(bias), std::move(mean), std::move(var)},
                 {std::move(y), std::move(output_mean), std::move(output_var)}};
           });
  }

  // ``batchnorm_epsilon_training_mode``: training mode (training_mode = 1) on
  // the larger 2x3x4x5 input with epsilon = 1e-2. Like the inference epsilon
  // case, the input is a deterministic ramp so the case stays reproducible.
  {
    NodeProto node;
    node.set_op_type("BatchNormalization");
    node.add_input("x");
    node.add_input("scale");
    node.add_input("B");
    node.add_input("input_mean");
    node.add_input("input_var");
    node.add_output("y");
    node.add_output("output_mean");
    node.add_output("output_var");
    AddAttribute<float>(node, "epsilon", 1e-2f);
    AddAttribute<int64_t>(node, "training_mode", 1);
    Expect(registry, std::move(node), "test_cc_batchnorm_epsilon_training_mode", {opset},
           [=]() -> IoData {
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
             Tensor scale = Tensor::FromFloat("", {C}, {1.0f, 1.5f, 2.0f});
             Tensor bias = Tensor::FromFloat("", {C}, {0.0f, -0.5f, 0.5f});
             Tensor mean = Tensor::FromFloat("", {C}, {0.5f, 1.0f, -0.25f});
             Tensor var = Tensor::FromFloat("", {C}, {0.25f, 0.5f, 1.0f});

             auto [y, output_mean, output_var] =
                 batchnorm_kernel.TrainingForward(x, scale, bias, mean, var, /*epsilon=*/1e-2f);

             return IoData{
                 {std::move(x), std::move(scale), std::move(bias), std::move(mean), std::move(var)},
                 {std::move(y), std::move(output_mean), std::move(output_var)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
