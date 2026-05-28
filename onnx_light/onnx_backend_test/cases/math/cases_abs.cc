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

// ---------------------------------------------------------------------------
// Abs — y = |x| (since opset 13 for the floating-point variant we use).
// Uses a small, fully deterministic input so this library does not depend
// on a PRNG.
// ---------------------------------------------------------------------------
void RegisterAbsCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Abs abs_kernel{ctx};

  {
    NodeProto node;
    node.set_op_type("Abs");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {2, 3}, {-1.0f, 0.0f, 1.5f, -2.25f, 3.5f, -4.75f});
    Tensor y = abs_kernel(x);

    Expect(node, {x}, {y}, "test_cc_abs", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test case for the ``Abs`` operator (mirrors the
  // ``onnx.backend.test.case.node.abs.Abs`` Python class). The upstream case
  // uses ``np.random.randn(3, 4, 5).astype(np.float32)`` as input; we use the
  // deterministic ``Randn`` helper here so the registry remains reproducible
  // without depending on NumPy.
  //
  // From Abs.export():
  {
    NodeProto node;
    node.set_op_type("Abs");
    node.add_input("x");
    node.add_output("y");

    const std::vector<int64_t> shape = {3, 4, 5};
    Tensor x = Tensor::FromFloat("", shape, Randn<float>(shape, /*seed=*/5));
    Tensor y = abs_kernel(x);
    Expect(node, {x}, {y}, "test_abs", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
