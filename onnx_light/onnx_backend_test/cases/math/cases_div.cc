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
  return Tensor::FromFloat("", shape, Randn<float>(shape, seed));
}

Tensor RandFloatUnitOffset(const std::vector<int64_t> &shape, uint64_t seed) {
  // Mirrors the upstream ``np.random.rand(...) + 1.0`` pattern: values are
  // pseudo-random doubles in ``[1, 2)``, guaranteed to avoid division by zero.
  const std::vector<double> values = Rand(shape, seed);
  std::vector<float> floats(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    floats[i] = static_cast<float>(values[i] + 1.0);
  }
  return Tensor::FromFloat("", shape, floats);
}

} // namespace

// ---------------------------------------------------------------------------
// Div — z = x / y, element-wise with broadcasting (since opset 14).
// ---------------------------------------------------------------------------
void RegisterDivCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(14);
  const kernel::Div div_kernel{kernel::KernelContext(opset)};

  // Equal-shape variant.
  {
    NodeProto node;
    node.set_op_type("Div");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 3}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
    Tensor y = Tensor::FromFloat("", {2, 3}, {2.0f, 4.0f, 5.0f, 8.0f, 10.0f, 12.0f});
    Tensor z = div_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_div", {opset}, "backend-test", registry);
  }

  // Scalar broadcast variant.
  {
    NodeProto node;
    node.set_op_type("Div");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("", {}, {2.0f});
    Tensor z = div_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_div_bcast", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Div`` operator (mirror the
  // ``onnx.backend.test.case.node.div.Div`` Python class for the float-32
  // variants). Integer variants are not registered (``kernel::Div`` only
  // implements FLOAT).
  //
  // From Div.export():
  {
    NodeProto node;
    node.set_op_type("Div");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    // test_div_example: expected z = [3, 2].
    Tensor x = Tensor::FromFloat("", {2}, {3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("", {2}, {1.0f, 2.0f});
    Tensor z = div_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_div_example", {opset}, "backend-test", registry);
  }
  {
    NodeProto node;
    node.set_op_type("Div");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    // y is in ``[1, 2)`` to avoid division by zero, matching upstream
    // ``np.random.rand(...) + 1.0``.
    Tensor x = RandnFloat({3, 4, 5}, /*seed=*/33);
    Tensor y = RandFloatUnitOffset({3, 4, 5}, /*seed=*/34);
    Tensor z = div_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_div", {opset}, "backend-test", registry);
  }
  // From Div.export_div_broadcast():
  {
    NodeProto node;
    node.set_op_type("Div");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = RandnFloat({3, 4, 5}, /*seed=*/35);
    Tensor y = RandFloatUnitOffset({5}, /*seed=*/36);
    Tensor z = div_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_div_bcast", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
