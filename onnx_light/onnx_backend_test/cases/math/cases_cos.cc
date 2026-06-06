// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/random.h"
#include "onnx_kernels/test_case.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

namespace {

// Deterministic FLOAT tensor with approximately-normal values produced by
// the repository's SplitMix64-based ``Randn`` generator (Irwin-Hall
// approximation). cos is defined for every real input so no clamping is
// required.
Tensor RandnFloat(const std::vector<int64_t> &shape, uint64_t seed) {
  return Tensor::FromFloat("", shape, Randn<float>(shape, seed));
}

} // namespace

// ---------------------------------------------------------------------------
// Cos — y = cos(x) (since opset 7, widened to bfloat16 in opset 22).
// Registers both a small deterministic ``test_cc_cos`` case and the upstream
// ONNX backend test cases (``test_cos_example`` and ``test_cos``) mirrored
// from ``onnx.backend.test.case.node.cos.Cos`` for the FLOAT variant.
// ---------------------------------------------------------------------------
void RegisterCosCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};
  const kernel::Cos cos_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Cos");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {2, 3}, {-1.0f, -0.5f, 0.0f, 0.25f, 0.5f, 1.0f});
    Tensor y = cos_kernel(x);

    Expect(node, {x}, {y}, "test_cc_cos", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Cos`` operator (mirror the
  // ``onnx.backend.test.case.node.cos.Cos`` Python class).
  //
  // From Cos.export(): ``test_cos_example`` uses x = [-1, 0, 1].
  {
    NodeProto node;
    node.set_op_type("Cos");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
    Tensor y = cos_kernel(x);
    Expect(node, {x}, {y}, "test_cos_example", {opset}, "backend-test", registry);
  }
  // From Cos.export(): ``test_cos`` uses x = np.random.randn(3, 4, 5).
  {
    NodeProto node;
    node.set_op_type("Cos");
    node.add_input("x");
    node.add_output("y");

    Tensor x = RandnFloat({3, 4, 5}, /*seed=*/1);
    Tensor y = cos_kernel(x);
    Expect(node, {x}, {y}, "test_cos", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
