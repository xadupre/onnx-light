// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_backend_test/kernels/math/include_math_kernels.h"
#include "onnx_backend_test/random.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Deterministic FLOAT tensor with approximately-normal values produced by
// the repository's SplitMix64-based ``Randn`` generator (Irwin-Hall
// approximation). atan is defined for every real input so no clamping is
// required.
Tensor RandnFloat(const std::vector<int64_t> &shape, uint64_t seed) {
  return Tensor::FromFloat("", shape, Randn<float>(shape, seed));
}

} // namespace

// ---------------------------------------------------------------------------
// Atan — y = atan(x) (since opset 7, widened to bfloat16 in opset 22).
// Registers both a small deterministic ``test_cc_atan`` case and the upstream
// ONNX backend test cases (``test_atan_example`` and ``test_atan``) mirrored
// from ``onnx.backend.test.case.node.atan.Atan`` for the FLOAT variant.
// ---------------------------------------------------------------------------
void RegisterAtanCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};
  const kernel::Atan atan_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Atan");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {2, 3}, {-2.0f, -0.5f, 0.0f, 0.5f, 1.0f, 3.0f});
    Tensor y = atan_kernel(x);

    Expect(node, {x}, {y}, "test_cc_atan", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Atan`` operator (mirror the
  // ``onnx.backend.test.case.node.atan.Atan`` Python class).
  //
  // From Atan.export(): ``test_atan_example`` uses x = [-1, 0, 1].
  {
    NodeProto node;
    node.set_op_type("Atan");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
    Tensor y = atan_kernel(x);
    Expect(node, {x}, {y}, "test_atan_example", {opset}, "backend-test", registry);
  }
  // From Atan.export(): ``test_atan`` uses x = np.random.randn(3, 4, 5).
  {
    NodeProto node;
    node.set_op_type("Atan");
    node.add_input("x");
    node.add_output("y");

    Tensor x = RandnFloat({3, 4, 5}, /*seed=*/1);
    Tensor y = atan_kernel(x);
    Expect(node, {x}, {y}, "test_atan", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
