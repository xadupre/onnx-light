// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_backend_test/kernels/logical/include_logical_kernels.h"
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
// Greater — z = x > y, element-wise with broadcasting (since opset 7).
// Inputs are FLOAT tensors, output is BOOL.
// ---------------------------------------------------------------------------
void RegisterGreaterCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::Greater greater_kernel{kernel::KernelContext(opset)};

  // Equal-shape variant.
  {
    NodeProto node;
    node.set_op_type("Greater");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("", {2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});
    Tensor z = greater_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_greater", {opset}, "backend-test", registry);
  }

  // Scalar broadcast variant: z[i] = x[i] > y (scalar).
  {
    NodeProto node;
    node.set_op_type("Greater");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("", {}, {2.5f});
    Tensor z = greater_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_greater_bcast", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Greater`` operator (mirror the
  // ``onnx.backend.test.case.node.greater.Greater`` Python class for the
  // float-32 variants). Integer variants are not registered (``kernel::Greater``
  // only implements FLOAT — matching the way ``Add``/``Mul`` register only
  // their FLOAT upstream cases).
  //
  // From Greater.export():
  {
    NodeProto node;
    node.set_op_type("Greater");
    node.add_input("x");
    node.add_input("y");
    node.add_output("greater");

    Tensor x = RandnFloat({3, 4, 5}, /*seed=*/21);
    Tensor y = RandnFloat({3, 4, 5}, /*seed=*/22);
    Tensor z = greater_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_greater", {opset}, "backend-test", registry);
  }
  // From Greater.export_greater_broadcast():
  {
    NodeProto node;
    node.set_op_type("Greater");
    node.add_input("x");
    node.add_input("y");
    node.add_output("greater");

    Tensor x = RandnFloat({3, 4, 5}, /*seed=*/23);
    Tensor y = RandnFloat({5}, /*seed=*/24);
    Tensor z = greater_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_greater_bcast", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
