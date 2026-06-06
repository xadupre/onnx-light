// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/random.h"
#include "onnx_backend_test/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Neg — y = -x (since opset 13 for the floating-point variant we use).
// Registers both a deterministic ``test_cc_neg`` case and the upstream
// ONNX backend test cases (``test_neg_example`` and ``test_neg``) mirrored
// from ``onnx.backend.test.case.node.neg.Neg``.
// ---------------------------------------------------------------------------
void RegisterNegCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Neg neg_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Neg");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {2, 3}, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});
    Tensor y = neg_kernel(x);

    Expect(node, {x}, {y}, "test_cc_neg", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Neg`` operator (mirror the
  // ``onnx.backend.test.case.node.neg.Neg`` Python class).
  //
  // From Neg.export(): ``test_neg_example`` uses x = [-4, 2].
  {
    NodeProto node;
    node.set_op_type("Neg");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {2}, {-4.0f, 2.0f});
    Tensor y = neg_kernel(x);
    Expect(node, {x}, {y}, "test_neg_example", {opset}, "backend-test", registry);
  }
  // From Neg.export(): ``test_neg`` uses x = np.random.randn(3, 4, 5).
  {
    NodeProto node;
    node.set_op_type("Neg");
    node.add_input("x");
    node.add_output("y");

    const std::vector<int64_t> shape = {3, 4, 5};
    Tensor x = Tensor::FromFloat("", shape, Randn<float>(shape, /*seed=*/1));
    Tensor y = neg_kernel(x);
    Expect(node, {x}, {y}, "test_neg", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
