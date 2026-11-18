// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/nn/include_nn_cases.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// LpNormalization — normalizes ``input`` so that each slice along the chosen
// axis has unit Lp norm. Output shape and dtype match the input.
//
// Cases:
//   * test_cc_lpnormalization_default — default axis=-1 and p=2 on a
//     (2, 2, 3) FLOAT input (mirrors upstream ``test_lpnormalization_default``).
//   * test_cc_lpnormalization_axis0_p1 — axis=0, p=1 on the same input.
//   * test_l2normalization_axis_0 / _axis_1 — mirror upstream ONNX cases.
//   * test_l1normalization_axis_0 / _axis_1 / _axis_last — mirror upstream
//     ONNX cases.
//   * test_l1normalization_negative_values — L1 norm uses absolute values.
// ---------------------------------------------------------------------------
void RegisterLpNormalizationCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("LpNormalization");
    node.add_input("x");
    node.add_output("y");

    constexpr int64_t count = 32 * 64 * 1024;
    Expect(registry, std::move(node), "test_cc_lpnormalization_default_benchmark", {opset}, {count},
           {count}, []() -> IoData {
             const OpsetId opset = DefaultOpset(22);

             const KernelContext kernel_ctx{opset};
             const onnx_kernels::kernel::LpNormalization kernel{kernel_ctx};

             Tensor x = RandnTensor(DataType::FLOAT, {32, 64, 1024}, 2101);
             Tensor y = kernel(x);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // Upstream PR #8454: finite normalization must not overflow or underflow.
  for (const int64_t p : {1, 2}) {
    for (const float magnitude : {3e38f, 1e30f, 1e-30f, std::numeric_limits<float>::denorm_min()}) {
      for (const int64_t axis : {-1, 0, 1}) {
        NodeProto node;
        node.set_op_type("LpNormalization");
        node.add_input("x");
        node.add_output("y");
        AddAttribute<int64_t>(node, "axis", axis);
        AddAttribute<int64_t>(node, "p", p);
        const std::string name = "test_cc_lpnormalization_finite_extreme_p" + std::to_string(p) +
                                 "_magnitude_" + std::to_string(std::ilogb(magnitude)) + "_axis_" +
                                 std::to_string(axis);
        Expect(registry, std::move(node), name, {opset}, [p, magnitude, axis]() -> IoData {
          const float value = p == 1 ? 0.5f : static_cast<float>(std::sqrt(0.5));
          Tensor x = Tensor::FromFloat("", {2, 2},
                                       axis != 0 ? std::vector<float>{magnitude, -magnitude, 0, 0}
                                                 : std::vector<float>{magnitude, 0, -magnitude, 0});
          Tensor y = Tensor::FromFloat("", {2, 2},
                                       axis != 0 ? std::vector<float>{value, -value, 0, 0}
                                                 : std::vector<float>{value, 0, -value, 0});
          return IoData{{std::move(x)}, {std::move(y)}};
        });
      }
    }
  }

  for (const int64_t p : {1, 2}) {
    NodeProto node;
    node.set_op_type("LpNormalization");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<int64_t>(node, "p", p);
    Expect(registry, std::move(node), "test_cc_lpnormalization_long_slice_p" + std::to_string(p),
           {opset}, [p]() -> IoData {
             constexpr int64_t n = 65520;
             const float value = static_cast<float>(p == 1 ? 1.0 / n : 1.0 / std::sqrt(n));
             Tensor x = Tensor::FromFloat("", {1, n}, std::vector<float>(n, 1.0f));
             Tensor y = Tensor::FromFloat("", {1, n}, std::vector<float>(n, value));
             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }

  Tensor x = Tensor::FromFloat(
      "", {2, 2, 3}, {1.0f, 2.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 5.0f, 5.0f, 6.0f, 8.0f, 0.0f});

  // Default attributes — axis=-1, p=2.
  {
    NodeProto node;
    node.set_op_type("LpNormalization");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_lpnormalization_default", {opset}, []() -> IoData {
      Tensor x = Tensor::FromFloat(
          "", {2, 2, 3}, {1.0f, 2.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 5.0f, 5.0f, 6.0f, 8.0f, 0.0f});

      const OpsetId opset = DefaultOpset(22);

      const KernelContext kernel_ctx{opset};
      const onnx_kernels::kernel::LpNormalization kernel{kernel_ctx};

      Tensor y = kernel(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Explicit attributes — axis=0, p=1.
  {
    NodeProto node;
    node.set_op_type("LpNormalization");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<int64_t>(node, "axis", 0);
    AddAttribute<int64_t>(node, "p", 1);
    Expect(registry, std::move(node), "test_cc_lpnormalization_axis0_p1", {opset}, []() -> IoData {
      Tensor x = Tensor::FromFloat(
          "", {2, 2, 3}, {1.0f, 2.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 5.0f, 5.0f, 6.0f, 8.0f, 0.0f});

      const OpsetId opset = DefaultOpset(22);

      const KernelContext kernel_ctx{opset};
      const onnx_kernels::kernel::LpNormalization kernel{kernel_ctx};

      Tensor y = kernel(x, /*axis=*/0, /*p=*/1);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Upstream ONNX case: L2 normalization on axis=0 over the (2, 2, 3) input.
  {
    NodeProto node;
    node.set_op_type("LpNormalization");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<int64_t>(node, "axis", 0);
    AddAttribute<int64_t>(node, "p", 2);
    Expect(registry, std::move(node), "test_l2normalization_axis_0", {opset}, []() -> IoData {
      Tensor x = Tensor::FromFloat(
          "", {2, 2, 3}, {1.0f, 2.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 5.0f, 5.0f, 6.0f, 8.0f, 0.0f});

      const OpsetId opset = DefaultOpset(22);

      const KernelContext kernel_ctx{opset};
      const onnx_kernels::kernel::LpNormalization kernel{kernel_ctx};

      Tensor y = kernel(x, /*axis=*/0, /*p=*/2);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Upstream ONNX cases use a (2, 2) input for axis_1 variants.
  Tensor x22 = Tensor::FromFloat("", {2, 2}, {3.0f, 4.0f, 6.0f, 8.0f});

  // Upstream ONNX case: L2 normalization on axis=1 over a (2, 2) input.
  {
    NodeProto node;
    node.set_op_type("LpNormalization");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<int64_t>(node, "axis", 1);
    AddAttribute<int64_t>(node, "p", 2);
    Expect(registry, std::move(node), "test_l2normalization_axis_1", {opset}, []() -> IoData {
      Tensor x22 = Tensor::FromFloat("", {2, 2}, {3.0f, 4.0f, 6.0f, 8.0f});

      const OpsetId opset = DefaultOpset(22);

      const KernelContext kernel_ctx{opset};
      const onnx_kernels::kernel::LpNormalization kernel{kernel_ctx};

      Tensor y = kernel(x22, /*axis=*/1, /*p=*/2);
      return IoData{{std::move(x22)}, {std::move(y)}};
    });
  }

  // Upstream ONNX case: L1 normalization on axis=0 over a 1-D input.
  {
    Tensor x1 = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
    NodeProto node;
    node.set_op_type("LpNormalization");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<int64_t>(node, "axis", 0);
    AddAttribute<int64_t>(node, "p", 1);
    Expect(registry, std::move(node), "test_l1normalization_axis_0", {opset}, []() -> IoData {
      Tensor x1 = Tensor::FromFloat("", {2}, {3.0f, 4.0f});

      const OpsetId opset = DefaultOpset(22);

      const KernelContext kernel_ctx{opset};
      const onnx_kernels::kernel::LpNormalization kernel{kernel_ctx};

      Tensor y = kernel(x1, /*axis=*/0, /*p=*/1);
      return IoData{{std::move(x1)}, {std::move(y)}};
    });
  }

  // Upstream ONNX case: L1 normalization with negative values.
  {
    NodeProto node;
    node.set_op_type("LpNormalization");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<int64_t>(node, "axis", 0);
    AddAttribute<int64_t>(node, "p", 1);
    Expect(registry, std::move(node), "test_l1normalization_negative_values", {opset},
           []() -> IoData {
             Tensor x = Tensor::FromFloat("", {2}, {1.0f, -1.0f});
             Tensor y = Tensor::FromFloat("", {2}, {0.5f, -0.5f});
             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }

  // Upstream ONNX case: L1 normalization on axis=1 over a (2, 2) input.
  {
    NodeProto node;
    node.set_op_type("LpNormalization");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<int64_t>(node, "axis", 1);
    AddAttribute<int64_t>(node, "p", 1);
    Expect(registry, std::move(node), "test_l1normalization_axis_1", {opset}, []() -> IoData {
      Tensor x22 = Tensor::FromFloat("", {2, 2}, {3.0f, 4.0f, 6.0f, 8.0f});

      const OpsetId opset = DefaultOpset(22);

      const KernelContext kernel_ctx{opset};
      const onnx_kernels::kernel::LpNormalization kernel{kernel_ctx};

      Tensor y = kernel(x22, /*axis=*/1, /*p=*/1);
      return IoData{{std::move(x22)}, {std::move(y)}};
    });
  }

  // Upstream ONNX case: L1 normalization on the last axis of the (2, 2, 3) input.
  {
    NodeProto node;
    node.set_op_type("LpNormalization");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<int64_t>(node, "axis", -1);
    AddAttribute<int64_t>(node, "p", 1);
    Expect(registry, std::move(node), "test_l1normalization_axis_last", {opset}, []() -> IoData {
      Tensor x = Tensor::FromFloat(
          "", {2, 2, 3}, {1.0f, 2.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 5.0f, 5.0f, 6.0f, 8.0f, 0.0f});

      const OpsetId opset = DefaultOpset(22);

      const KernelContext kernel_ctx{opset};
      const onnx_kernels::kernel::LpNormalization kernel{kernel_ctx};

      Tensor y = kernel(x, /*axis=*/-1, /*p=*/1);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
