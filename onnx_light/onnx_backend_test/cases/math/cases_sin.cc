// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/random.h"
#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Sin — y = sin(x) (since opset 7, widened to bfloat16 in opset 22).
// Registers both a small deterministic ``test_cc_sin`` case and the upstream
// ONNX backend test cases (``test_sin_example`` and ``test_sin``) mirrored
// from ``onnx.backend.test.case.node.sin.Sin`` for the FLOAT variant.
// ---------------------------------------------------------------------------
void RegisterSinCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};
  const kernel::Sin sin_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Sin");
    node.add_input("x");
    node.add_output("y");

    Tensor x =
        Tensor::FromFloat("", {2, 3}, {-3.14159f, -1.5708f, 0.0f, 1.0472f, 1.5708f, 3.14159f});
    Tensor y = sin_kernel(x);

    Expect(node, {x}, {y}, "test_cc_sin", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Sin`` operator (mirror the
  // ``onnx.backend.test.case.node.sin.Sin`` Python class).
  //
  // From Sin.export(): ``test_sin_example`` uses x = [-1, 0, 1].
  {
    NodeProto node;
    node.set_op_type("Sin");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
    Tensor y = sin_kernel(x);
    Expect(node, {x}, {y}, "test_sin_example", {opset}, "backend-test", registry);
  }
  // From Sin.export(): ``test_sin`` uses x = np.random.rand(3, 4, 5).
  {
    NodeProto node;
    node.set_op_type("Sin");
    node.add_input("x");
    node.add_output("y");

    const std::vector<int64_t> shape = {3, 4, 5};
    Tensor x = Tensor::FromFloat("", shape, Rand<float>(shape, /*seed=*/1));
    Tensor y = sin_kernel(x);
    Expect(node, {x}, {y}, "test_sin", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
