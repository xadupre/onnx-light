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
// Asin — y = asin(x) (since opset 7, widened to bfloat16 in opset 22).
// Registers both a small deterministic ``test_cc_asin`` case and the upstream
// ONNX backend test cases (``test_asin_example`` and ``test_asin``) mirrored
// from ``onnx.backend.test.case.node.asin.Asin`` for the FLOAT variant.
// ---------------------------------------------------------------------------
void RegisterAsinCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::Asin asin_kernel{kernel::KernelContext(opset)};

  {
    NodeProto node;
    node.set_op_type("Asin");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {2, 3}, {-1.0f, -0.5f, 0.0f, 0.25f, 0.5f, 1.0f});
    Tensor y = asin_kernel(x);

    Expect(node, {x}, {y}, "test_cc_asin", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Asin`` operator (mirror the
  // ``onnx.backend.test.case.node.asin.Asin`` Python class).
  //
  // From Asin.export(): ``test_asin_example`` uses x = [-0.5, 0, 0.5].
  {
    NodeProto node;
    node.set_op_type("Asin");
    node.add_input("x");
    node.add_output("y");

    Tensor x = Tensor::FromFloat("", {3}, {-0.5f, 0.0f, 0.5f});
    Tensor y = asin_kernel(x);
    Expect(node, {x}, {y}, "test_asin_example", {opset}, "backend-test", registry);
  }
  // From Asin.export(): ``test_asin`` uses x = np.random.rand(3, 4, 5).
  {
    NodeProto node;
    node.set_op_type("Asin");
    node.add_input("x");
    node.add_output("y");

    const std::vector<int64_t> shape = {3, 4, 5};
    Tensor x = Tensor::FromFloat("", shape, Rand<float>(shape, /*seed=*/1));
    Tensor y = asin_kernel(x);
    Expect(node, {x}, {y}, "test_asin", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
