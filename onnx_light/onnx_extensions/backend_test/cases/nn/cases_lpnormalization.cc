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
// ---------------------------------------------------------------------------
void RegisterLpNormalizationCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const auto kernel = MakeReferenceKernel<onnx_kernels::kernel::LpNormalization>(opset);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("LpNormalization");
    node.add_input("x");
    node.add_output("y");

    constexpr int64_t count = 32 * 64 * 1024;
    Expect(registry, std::move(node), "test_cc_lpnormalization_default_benchmark", {opset}, {count},
           {count}, [kernel]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, {32, 64, 1024}, 2101);
             Tensor y = kernel.Invoke([&](const auto &kernel) { return kernel(x); });
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  Tensor x = Tensor::FromFloat(
      "", {2, 2, 3}, {1.0f, 2.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 5.0f, 5.0f, 6.0f, 8.0f, 0.0f});

  // Default attributes — axis=-1, p=2.
  {
    NodeProto node;
    node.set_op_type("LpNormalization");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_lpnormalization_default", {opset}, [=]() -> IoData {
      Tensor y = kernel.Invoke([&](const auto &kernel) { return kernel(x); });
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
    Expect(registry, std::move(node), "test_cc_lpnormalization_axis0_p1", {opset}, [=]() -> IoData {
      Tensor y = kernel.Invoke([&](const auto &kernel) { return kernel(x, /*axis=*/0, /*p=*/1); });
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
    Expect(registry, std::move(node), "test_l2normalization_axis_0", {opset}, [=]() -> IoData {
      Tensor y = kernel.Invoke([&](const auto &kernel) { return kernel(x, /*axis=*/0, /*p=*/2); });
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
    Expect(registry, std::move(node), "test_l2normalization_axis_1", {opset}, [=]() -> IoData {
      Tensor y =
          kernel.Invoke([&](const auto &kernel) { return kernel(x22, /*axis=*/1, /*p=*/2); });
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
    Expect(registry, std::move(node), "test_l1normalization_axis_0", {opset}, [=]() -> IoData {
      Tensor y = kernel.Invoke([&](const auto &kernel) { return kernel(x1, /*axis=*/0, /*p=*/1); });
      return IoData{{std::move(x1)}, {std::move(y)}};
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
    Expect(registry, std::move(node), "test_l1normalization_axis_1", {opset}, [=]() -> IoData {
      Tensor y =
          kernel.Invoke([&](const auto &kernel) { return kernel(x22, /*axis=*/1, /*p=*/1); });
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
    Expect(registry, std::move(node), "test_l1normalization_axis_last", {opset}, [=]() -> IoData {
      Tensor y = kernel.Invoke([&](const auto &kernel) { return kernel(x, /*axis=*/-1, /*p=*/1); });
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
