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
// approximation). cosh is defined for every real input so no clamping is
// required.
Tensor RandnFloat(const std::vector<int64_t> &shape, uint64_t seed) {
  return Tensor::FromFloat("", shape, Randn<float>(shape, seed));
}

} // namespace

// ---------------------------------------------------------------------------
// Cosh — y = cosh(x) (since opset 9, widened to bfloat16 in opset 22).
// Registers both a small deterministic ``test_cc_cosh`` case and the upstream
// ONNX backend test cases (``test_cosh_example`` and ``test_cosh``) mirrored
// from ``onnx.backend.test.case.node.cosh.Cosh`` for the FLOAT variant.
// ---------------------------------------------------------------------------
void RegisterCoshCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};
  const kernel::Cosh cosh_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Cosh");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {2, 3}, {-2.0f, -0.5f, 0.0f, 0.5f, 1.0f, 3.0f});
    Tensor y = cosh_kernel(x);

    Expect(node, {x}, {y}, "test_cc_cosh", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Cosh`` operator (mirror the
  // ``onnx.backend.test.case.node.cosh.Cosh`` Python class).
  //
  // From Cosh.export(): ``test_cosh_example`` uses x = [-1, 0, 1].
  {
    NodeProto node;
    node.set_op_type("Cosh");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
    Tensor y = cosh_kernel(x);
    Expect(node, {x}, {y}, "test_cosh_example", {opset}, "backend-test", registry);
  }
  // From Cosh.export(): ``test_cosh`` uses x = np.random.randn(3, 4, 5).
  {
    NodeProto node;
    node.set_op_type("Cosh");
    node.add_input("x");
    node.add_output("y");

    Tensor x = RandnFloat({3, 4, 5}, /*seed=*/1);
    Tensor y = cosh_kernel(x);
    Expect(node, {x}, {y}, "test_cosh", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
