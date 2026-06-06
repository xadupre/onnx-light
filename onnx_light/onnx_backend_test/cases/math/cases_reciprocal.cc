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

namespace {

Tensor PositiveRandFloat(const std::vector<int64_t> &shape, uint64_t seed) {
  // Match onnx.backend.test.case.node.reciprocal: np.random.rand(...) + 0.5
  // (strictly positive, avoids dividing by zero or values close to zero).
  std::vector<float> values = Rand<float>(shape, seed);
  for (float &v : values) {
    if (v < 0.0f) {
      v = -v;
    }
    v += 0.5f;
  }
  return Tensor::FromFloat("", shape, values);
}

} // namespace

// ---------------------------------------------------------------------------
// Reciprocal — y = 1 / x (latest opset: 13).
// Registers a small deterministic ``test_cc_reciprocal`` case and the upstream
// ONNX backend test cases (``test_reciprocal_example`` and
// ``test_reciprocal``) for FLOAT.
// ---------------------------------------------------------------------------
void RegisterReciprocalCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Reciprocal reciprocal_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Reciprocal");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {2, 3}, {-2.0f, -0.5f, 0.25f, 1.0f, 2.0f, 4.0f});
    Tensor y = reciprocal_kernel(x);
    Expect(node, {x}, {y}, "test_cc_reciprocal", {opset}, "backend-test", registry);
  }

  // From Reciprocal.export(): ``test_reciprocal_example`` uses x = [-4, 2].
  {
    NodeProto node;
    node.set_op_type("Reciprocal");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {2}, {-4.0f, 2.0f});
    Tensor y = reciprocal_kernel(x);
    Expect(node, {x}, {y}, "test_reciprocal_example", {opset}, "backend-test", registry);
  }

  // From Reciprocal.export(): ``test_reciprocal`` uses np.random.rand(3, 4, 5) + 0.5.
  {
    NodeProto node;
    node.set_op_type("Reciprocal");
    node.add_input("x");
    node.add_output("y");

    Tensor x = PositiveRandFloat({3, 4, 5}, /*seed=*/1);
    Tensor y = reciprocal_kernel(x);
    Expect(node, {x}, {y}, "test_reciprocal", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
