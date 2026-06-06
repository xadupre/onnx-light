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
// Tan — y = tan(x) (since opset 7, widened to bfloat16 in opset 22).
// Registers both a small deterministic ``test_cc_tan`` case and the upstream
// ONNX backend test cases (``test_tan_example`` and ``test_tan``) mirrored
// from ``onnx.backend.test.case.node.tan.Tan`` for the FLOAT variant.
// ---------------------------------------------------------------------------
void RegisterTanCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};
  const kernel::Tan tan_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Tan");
    node.add_input("x");
    node.add_output("y");

    // Stay clear of pi/2 ± k*pi where tan(x) is unbounded.
    Tensor x = Tensor::FromFloat("", {2, 3}, {-1.0f, -0.5f, 0.0f, 0.25f, 0.5f, 1.0f});
    Tensor y = tan_kernel(x);

    Expect(node, {x}, {y}, "test_cc_tan", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Tan`` operator (mirror the
  // ``onnx.backend.test.case.node.tan.Tan`` Python class).
  //
  // From Tan.export(): ``test_tan_example`` uses x = [-1, 0, 1].
  {
    NodeProto node;
    node.set_op_type("Tan");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
    Tensor y = tan_kernel(x);
    Expect(node, {x}, {y}, "test_tan_example", {opset}, "backend-test", registry);
  }
  // From Tan.export(): ``test_tan`` uses x = np.random.randn(3, 4, 5).
  {
    NodeProto node;
    node.set_op_type("Tan");
    node.add_input("x");
    node.add_output("y");

    const std::vector<int64_t> shape = {3, 4, 5};
    Tensor x = Tensor::FromFloat("", shape, Randn<float>(shape, /*seed=*/1));
    Tensor y = tan_kernel(x);
    Expect(node, {x}, {y}, "test_tan", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
