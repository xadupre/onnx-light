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

Tensor RandnFloat(const std::vector<int64_t> &shape, uint64_t seed) {
  const std::vector<double> values = Randn(shape, seed);
  std::vector<float> floats(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    floats[i] = static_cast<float>(values[i]);
  }
  return Tensor::FromFloat("", shape, floats);
}

} // namespace

// ---------------------------------------------------------------------------
// Mul — z = x * y, element-wise with broadcasting (since opset 14).
// ---------------------------------------------------------------------------
void RegisterMulCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(14);
  const kernel::Mul mul_kernel{kernel::KernelContext(opset)};

  // Equal-shape variant.
  {
    NodeProto node;
    node.set_op_type("Mul");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor y = Tensor::FromFloat("", {2, 3}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
    Tensor z = mul_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_mul", {opset}, "backend-test", registry);
  }

  // Scalar broadcast variant.
  {
    NodeProto node;
    node.set_op_type("Mul");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("", {}, {2.0f});
    Tensor z = mul_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_mul_bcast", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Mul`` operator (mirror the
  // ``onnx.backend.test.case.node.mul.Mul`` Python class for the float-32
  // variants). Integer variants are not registered (``kernel::Mul`` only
  // implements FLOAT).
  //
  // From Mul.export():
  {
    NodeProto node;
    node.set_op_type("Mul");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    // test_mul_example: expected z = [4, 10, 18].
    Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
    Tensor y = Tensor::FromFloat("", {3}, {4.0f, 5.0f, 6.0f});
    Tensor z = mul_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_mul_example", {opset}, "backend-test", registry);
  }
  {
    NodeProto node;
    node.set_op_type("Mul");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = RandnFloat({3, 4, 5}, /*seed=*/17);
    Tensor y = RandnFloat({3, 4, 5}, /*seed=*/18);
    Tensor z = mul_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_mul", {opset}, "backend-test", registry);
  }
  // From Mul.export_mul_broadcast():
  {
    NodeProto node;
    node.set_op_type("Mul");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = RandnFloat({3, 4, 5}, /*seed=*/19);
    Tensor y = RandnFloat({5}, /*seed=*/20);
    Tensor z = mul_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_mul_bcast", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
