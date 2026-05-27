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
  return Tensor::FromFloat("", shape, Randn<float>(shape, seed));
}

} // namespace

// ---------------------------------------------------------------------------
// Less — z = x < y, element-wise with broadcasting (since opset 7).
// Inputs are FLOAT tensors, output is BOOL.
// ---------------------------------------------------------------------------
void RegisterLessCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::Less less_kernel{kernel::KernelContext(opset)};

  // Equal-shape variant.
  {
    NodeProto node;
    node.set_op_type("Less");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("", {2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});
    Tensor z = less_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_less", {opset}, "backend-test", registry);
  }

  // Scalar broadcast variant: z[i] = x[i] < y (scalar).
  {
    NodeProto node;
    node.set_op_type("Less");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("", {}, {2.5f});
    Tensor z = less_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_less_bcast", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Less`` operator (mirror the
  // ``onnx.backend.test.case.node.less.Less`` Python class for the float-32
  // variants). Integer variants are not registered (``kernel::Less`` only
  // implements FLOAT — matching the way ``Add``/``Mul`` register only their
  // FLOAT upstream cases).
  //
  // From Less.export():
  {
    NodeProto node;
    node.set_op_type("Less");
    node.add_input("x");
    node.add_input("y");
    node.add_output("less");

    Tensor x = RandnFloat({3, 4, 5}, /*seed=*/25);
    Tensor y = RandnFloat({3, 4, 5}, /*seed=*/26);
    Tensor z = less_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_less", {opset}, "backend-test", registry);
  }
  // From Less.export_less_broadcast():
  {
    NodeProto node;
    node.set_op_type("Less");
    node.add_input("x");
    node.add_input("y");
    node.add_output("less");

    Tensor x = RandnFloat({3, 4, 5}, /*seed=*/27);
    Tensor y = RandnFloat({5}, /*seed=*/28);
    Tensor z = less_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_less_bcast", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
