// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/random.h"
#include "onnx_backend_test/test_case.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Deterministic FLOAT tensor with approximately-normal values produced by
// the repository's SplitMix64-based ``Randn`` generator (Irwin-Hall
// approximation). asinh is defined for every real input so no clamping is
// required.
Tensor RandnFloat(const std::vector<int64_t> &shape, uint64_t seed) {
  return Tensor::FromFloat("", shape, Randn<float>(shape, seed));
}

} // namespace

// ---------------------------------------------------------------------------
// Asinh — y = asinh(x) (since opset 9, widened to bfloat16 in opset 22).
// Registers both a small deterministic ``test_cc_asinh`` case and the
// upstream ONNX backend test cases (``test_asinh_example`` and
// ``test_asinh``) mirrored from ``onnx.backend.test.case.node.asinh.Asinh``
// for the FLOAT variant.
// ---------------------------------------------------------------------------
void RegisterAsinhCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};
  const kernel::Asinh asinh_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Asinh");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {2, 3}, {-2.0f, -0.5f, 0.0f, 0.5f, 1.0f, 3.0f});
    Tensor y = asinh_kernel(x);

    Expect(node, {x}, {y}, "test_cc_asinh", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Asinh`` operator (mirror the
  // ``onnx.backend.test.case.node.asinh.Asinh`` Python class).
  //
  // From Asinh.export(): ``test_asinh_example`` uses x = [-1, 0, 1].
  {
    NodeProto node;
    node.set_op_type("Asinh");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
    Tensor y = asinh_kernel(x);
    Expect(node, {x}, {y}, "test_asinh_example", {opset}, "backend-test", registry);
  }
  // From Asinh.export(): ``test_asinh`` uses x = np.random.randn(3, 4, 5).
  {
    NodeProto node;
    node.set_op_type("Asinh");
    node.add_input("x");
    node.add_output("y");

    Tensor x = RandnFloat({3, 4, 5}, /*seed=*/1);
    Tensor y = asinh_kernel(x);
    Expect(node, {x}, {y}, "test_asinh", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
