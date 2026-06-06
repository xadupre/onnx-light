// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/random.h"
#include "onnx_kernels/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

// ---------------------------------------------------------------------------
// Tanh — y = tanh(x) (since opset 1, type widening at opset 6 and 13).
// Registers both a small deterministic ``test_cc_tanh`` case and the upstream
// ONNX backend test cases (``test_tanh_example`` and ``test_tanh``) mirrored
// from ``onnx.backend.test.case.node.tanh.Tanh`` for the FLOAT variant.
// ---------------------------------------------------------------------------
void RegisterTanhCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Tanh tanh_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Tanh");
    node.add_input("input");
    node.add_output("output");

    Tensor x = Tensor::FromFloat("", {2, 3}, {-4.0f, -1.0f, 0.0f, 1.0f, 2.0f, 4.0f});
    Tensor y = tanh_kernel(x);

    Expect(node, {x}, {y}, "test_cc_tanh", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Tanh`` operator (mirror the
  // ``onnx.backend.test.case.node.tanh.Tanh`` Python class).
  //
  // From Tanh.export(): ``test_tanh_example`` uses x = [-1, 0, 1].
  {
    NodeProto node;
    node.set_op_type("Tanh");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
    Tensor y = tanh_kernel(x);
    Expect(node, {x}, {y}, "test_tanh_example", {opset}, "backend-test", registry);
  }
  // From Tanh.export(): ``test_tanh`` uses x = np.random.randn(3, 4, 5).
  {
    NodeProto node;
    node.set_op_type("Tanh");
    node.add_input("x");
    node.add_output("y");

    const std::vector<int64_t> shape = {3, 4, 5};
    Tensor x = Tensor::FromFloat("", shape, Randn<float>(shape, /*seed=*/1));
    Tensor y = tanh_kernel(x);
    Expect(node, {x}, {y}, "test_tanh", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
