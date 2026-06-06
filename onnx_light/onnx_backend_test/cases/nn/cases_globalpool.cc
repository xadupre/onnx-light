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
// GlobalAveragePool — output = mean over all spatial elements per (N, C).
// Output shape is (N, C, 1, 1, ..., 1).
//
// Cases:
//   * test_cc_globalaveragepool — 2-D spatial (N=1, C=3, H=5, W=5).
//   * test_cc_globalaveragepool_precomputed — precomputed 1x1x3x3 example.
// ---------------------------------------------------------------------------
void RegisterGlobalAveragePoolCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};
  const kernel::GlobalAveragePool kernel{ctx};

  // 1 x 3 x 5 x 5 input — mirrors test_globalaveragepool.
  {
    NodeProto node;
    node.set_op_type("GlobalAveragePool");
    node.add_input("x");
    node.add_output("y");

    std::vector<float> x_data(1 * 3 * 5 * 5);
    for (size_t i = 0; i < x_data.size(); ++i) {
      x_data[i] = static_cast<float>(i + 1);
    }
    Tensor x = Tensor::FromFloat("", {1, 3, 5, 5}, x_data);
    Tensor y = kernel(x);

    Expect(node, {x}, {y}, "test_cc_globalaveragepool", {opset}, "backend-test", registry);
  }

  // 1 x 1 x 3 x 3 precomputed example.
  {
    NodeProto node;
    node.set_op_type("GlobalAveragePool");
    node.add_input("x");
    node.add_output("y");

    Tensor x =
        Tensor::FromFloat("", {1, 1, 3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    Tensor y = kernel(x); // expected: 5.0

    Expect(node, {x}, {y}, "test_cc_globalaveragepool_precomputed", {opset}, "backend-test",
           registry);
  }
}

// ---------------------------------------------------------------------------
// GlobalMaxPool — output = max over all spatial elements per (N, C).
// Output shape is (N, C, 1, 1, ..., 1).
//
// Cases:
//   * test_cc_globalmaxpool — 2-D spatial (N=1, C=3, H=5, W=5).
//   * test_cc_globalmaxpool_precomputed — precomputed 1x1x3x3 example.
// ---------------------------------------------------------------------------
void RegisterGlobalMaxPoolCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};
  const kernel::GlobalMaxPool kernel{ctx};

  // 1 x 3 x 5 x 5 input — mirrors test_globalmaxpool.
  {
    NodeProto node;
    node.set_op_type("GlobalMaxPool");
    node.add_input("x");
    node.add_output("y");

    std::vector<float> x_data(1 * 3 * 5 * 5);
    for (size_t i = 0; i < x_data.size(); ++i) {
      x_data[i] = static_cast<float>(i + 1);
    }
    Tensor x = Tensor::FromFloat("", {1, 3, 5, 5}, x_data);
    Tensor y = kernel(x);

    Expect(node, {x}, {y}, "test_cc_globalmaxpool", {opset}, "backend-test", registry);
  }

  // 1 x 1 x 3 x 3 precomputed example.
  {
    NodeProto node;
    node.set_op_type("GlobalMaxPool");
    node.add_input("x");
    node.add_output("y");

    Tensor x =
        Tensor::FromFloat("", {1, 1, 3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    Tensor y = kernel(x); // expected: 9.0

    Expect(node, {x}, {y}, "test_cc_globalmaxpool_precomputed", {opset}, "backend-test", registry);
  }
}

// ---------------------------------------------------------------------------
// GlobalLpPool — output = Lp norm over all spatial elements per (N, C).
// Output shape is (N, C, 1, 1, ..., 1).
//
// Cases:
//   * test_cc_globallppool_lp1 — 2-D spatial, p=1 (L1 norm).
//   * test_cc_globallppool_lp2 — 2-D spatial, p=2 (default L2 norm).
//   * test_cc_globallppool_default — 1x1x3x3, default p=2.
// ---------------------------------------------------------------------------
void RegisterGlobalLpPoolCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};
  const kernel::GlobalLpPool kernel{ctx};

  // 1 x 3 x 5 x 5 input, p=1.
  {
    NodeProto node;
    node.set_op_type("GlobalLpPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<int64_t>(node, "p", 1);

    std::vector<float> x_data(1 * 3 * 5 * 5);
    for (size_t i = 0; i < x_data.size(); ++i) {
      x_data[i] = static_cast<float>(i + 1);
    }
    Tensor x = Tensor::FromFloat("", {1, 3, 5, 5}, x_data);
    Tensor y = kernel(x, /*p=*/1);

    Expect(node, {x}, {y}, "test_cc_globallppool_lp1", {opset}, "backend-test", registry);
  }

  // 1 x 3 x 5 x 5 input, p=2.
  {
    NodeProto node;
    node.set_op_type("GlobalLpPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<int64_t>(node, "p", 2);

    std::vector<float> x_data(1 * 3 * 5 * 5);
    for (size_t i = 0; i < x_data.size(); ++i) {
      x_data[i] = static_cast<float>(i + 1);
    }
    Tensor x = Tensor::FromFloat("", {1, 3, 5, 5}, x_data);
    Tensor y = kernel(x, /*p=*/2);

    Expect(node, {x}, {y}, "test_cc_globallppool_lp2", {opset}, "backend-test", registry);
  }

  // 1 x 1 x 3 x 3, default p=2.
  {
    NodeProto node;
    node.set_op_type("GlobalLpPool");
    node.add_input("x");
    node.add_output("y");

    Tensor x =
        Tensor::FromFloat("", {1, 1, 3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    Tensor y = kernel(x); // expected: sqrt(1+4+9+16+25+36+49+64+81) = sqrt(285)

    Expect(node, {x}, {y}, "test_cc_globallppool_default", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
