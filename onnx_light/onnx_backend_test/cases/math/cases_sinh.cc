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
// Sinh — y = sinh(x) (since opset 9, widened to bfloat16 in opset 22).
// Registers both a small deterministic ``test_cc_sinh`` case and the
// upstream ONNX backend test cases (``test_sinh_example`` and ``test_sinh``)
// mirrored from ``onnx.backend.test.case.node.sinh.Sinh`` for FLOAT.
// ---------------------------------------------------------------------------
void RegisterSinhCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};
  const kernel::Sinh sinh_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Sinh");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {2, 3}, {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f});
    Tensor y = sinh_kernel(x);

    Expect(node, {x}, {y}, "test_cc_sinh", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Sinh`` operator (mirror the
  // ``onnx.backend.test.case.node.sinh.Sinh`` Python class).
  //
  // From Sinh.export(): ``test_sinh_example`` uses x = [-1, 0, 1].
  {
    NodeProto node;
    node.set_op_type("Sinh");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
    Tensor y = sinh_kernel(x);
    Expect(node, {x}, {y}, "test_sinh_example", {opset}, "backend-test", registry);
  }
  // From Sinh.export(): ``test_sinh`` uses x = np.random.rand(3, 4, 5).
  {
    NodeProto node;
    node.set_op_type("Sinh");
    node.add_input("x");
    node.add_output("y");

    const std::vector<int64_t> shape = {3, 4, 5};
    Tensor x = Tensor::FromFloat("", shape, Rand<float>(shape, /*seed=*/1));
    Tensor y = sinh_kernel(x);
    Expect(node, {x}, {y}, "test_sinh", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
