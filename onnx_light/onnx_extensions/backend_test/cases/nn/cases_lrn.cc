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
// LRN — Local Response Normalization across the channel dimension.
// Output shape and dtype match the input.
//
// Cases:
//   * test_cc_lrn — explicit alpha/beta/bias/size on a 2x4x5x5 deterministic
//     input (mirrors upstream ``test_lrn``).
//   * test_cc_lrn_default — only ``size`` specified, default alpha/beta/bias
//     (mirrors upstream ``test_lrn_default``).
// ---------------------------------------------------------------------------
void RegisterLRNCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::LRN kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("LRN");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<float>(node, "alpha", 0.0002f);
    AddAttribute<float>(node, "beta", 0.5f);
    AddAttribute<float>(node, "bias", 2.0f);
    AddAttribute<int64_t>(node, "size", 3);

    constexpr int64_t count = 1 * 64 * 128 * 128;
    Expect(registry, std::move(node), "test_cc_lrn_benchmark", {opset}, {count}, {count},
           [kernel]() -> IoData {
             Tensor x = RandnTensor(DataType::FLOAT, {1, 64, 128, 128}, 2201);
             Tensor y = kernel(x, /*size=*/3, /*alpha=*/0.0002f, /*beta=*/0.5f,
                               /*bias=*/2.0f);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // Build a deterministic 2x4x5x5 input shared between both cases. Using
  // distinct N and C dimensions guards against regressions where the channel
  // window is mistakenly iterated over the batch dimension (see
  // https://github.com/onnx/onnx/pull/7806).
  std::vector<float> x_data(2 * 4 * 5 * 5);
  for (size_t i = 0; i < x_data.size(); ++i) {
    // Map indices to a varied but reproducible signed float in [-1, 1).
    x_data[i] = static_cast<float>((static_cast<int64_t>(i) % 13) - 6) / 6.0f;
  }
  Tensor x = Tensor::FromFloat("", {2, 4, 5, 5}, x_data);

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
    Expect(registry, std::move(node), "test_cc_lrn", {opset}, [=]() -> IoData {
      Tensor y = kernel(x, /*size=*/3, /*alpha=*/0.0002f, /*beta=*/0.5f, /*bias=*/2.0f);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Default attributes — only ``size`` specified.
  {
    NodeProto node;
    node.set_op_type("LRN");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<int64_t>(node, "size", 3);
    Expect(registry, std::move(node), "test_cc_lrn_default", {opset}, [=]() -> IoData {
      Tensor y = kernel(x, /*size=*/3);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test
