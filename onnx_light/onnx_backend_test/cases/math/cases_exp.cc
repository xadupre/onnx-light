// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/random.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

Tensor RandnFloat(const std::vector<int64_t> &shape, uint64_t seed) {
  return Tensor::FromFloat("", shape, Randn<float>(shape, seed));
}

} // namespace

// ---------------------------------------------------------------------------
// Exp — y = exp(x) (latest opset: 13).
// Registers both a small deterministic ``test_cc_exp`` case and upstream ONNX
// backend test cases (``test_exp_example`` and ``test_exp``) for FLOAT.
// ---------------------------------------------------------------------------
void RegisterExpCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Exp exp_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Exp");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {2, 3}, {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f});
    Tensor y = exp_kernel(x);
    Expect(node, {x}, {y}, "test_cc_exp", {opset}, "backend-test", registry);
  }

  // From Exp.export(): ``test_exp_example`` uses x = [-1, 0, 1].
  {
    NodeProto node;
    node.set_op_type("Exp");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
    Tensor y = exp_kernel(x);
    Expect(node, {x}, {y}, "test_exp_example", {opset}, "backend-test", registry);
  }

  // From Exp.export(): ``test_exp`` uses x = np.random.randn(3, 4, 5).
  {
    NodeProto node;
    node.set_op_type("Exp");
    node.add_input("x");
    node.add_output("y");

    Tensor x = RandnFloat({3, 4, 5}, /*seed=*/1);
    Tensor y = exp_kernel(x);
    Expect(node, {x}, {y}, "test_exp", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
