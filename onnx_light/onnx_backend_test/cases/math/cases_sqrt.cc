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

namespace {

Tensor NonNegativeRandFloat(const std::vector<int64_t> &shape, uint64_t seed) {
  // Sqrt is only defined for non-negative inputs (negatives produce NaN).
  std::vector<float> values = Rand<float>(shape, seed);
  for (float &v : values) {
    if (v < 0.0f) {
      v = -v;
    }
  }
  return Tensor::FromFloat("", shape, values);
}

} // namespace

// ---------------------------------------------------------------------------
// Sqrt — y = sqrt(x) (latest opset: 13).
// Registers a small deterministic ``test_cc_sqrt`` case and upstream ONNX
// backend test cases (``test_sqrt_example`` and ``test_sqrt``) for FLOAT.
// ---------------------------------------------------------------------------
void RegisterSqrtCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Sqrt sqrt_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Sqrt");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {2, 3}, {0.0f, 0.25f, 1.0f, 2.0f, 4.0f, 9.0f});
    Tensor y = sqrt_kernel(x);
    Expect(node, {x}, {y}, "test_cc_sqrt", {opset}, "backend-test", registry);
  }

  // From Sqrt.export(): ``test_sqrt_example`` uses x = [1, 4, 9].
  {
    NodeProto node;
    node.set_op_type("Sqrt");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {3}, {1.0f, 4.0f, 9.0f});
    Tensor y = sqrt_kernel(x);
    Expect(node, {x}, {y}, "test_sqrt_example", {opset}, "backend-test", registry);
  }

  // From Sqrt.export(): ``test_sqrt`` uses np.abs(np.random.randn(3, 4, 5)).
  {
    NodeProto node;
    node.set_op_type("Sqrt");
    node.add_input("x");
    node.add_output("y");

    Tensor x = NonNegativeRandFloat({3, 4, 5}, /*seed=*/1);
    Tensor y = sqrt_kernel(x);
    Expect(node, {x}, {y}, "test_sqrt", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
