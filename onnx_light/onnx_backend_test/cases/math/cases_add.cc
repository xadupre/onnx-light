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
// Add — z = x + y, element-wise with broadcasting (since opset 14).
// This is the case exercised by examples/run_add_node_test/main.cc.
// ---------------------------------------------------------------------------
void RegisterAddCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(14);
  const kernel::Add add_kernel{kernel::KernelContext(opset)};

  // Equal-shape variant.
  {
    NodeProto node;
    node.set_op_type("Add");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor y = Tensor::FromFloat("", {2, 3}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
    Tensor z = add_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_add", {opset}, "backend-test", registry);
  }

  // Scalar broadcast variant: z[i] = x[i] + y (scalar).
  {
    NodeProto node;
    node.set_op_type("Add");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("", {}, {0.5f});
    Tensor z = add_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_add_bcast", {opset}, "backend-test", registry);
  }

  // Upstream ONNX backend test cases for the ``Add`` operator (mirror the
  // ``onnx.backend.test.case.node.add.Add`` Python class for the float-32
  // variants). Integer variants (int8/int16/uint8/uint16/uint32/uint64) are
  // not registered here because the reference ``kernel::Add`` kernel
  // currently only implements the FLOAT type.
  //
  // From Add.export():
  {
    NodeProto node;
    node.set_op_type("Add");
    node.add_input("x");
    node.add_input("y");
    node.add_output("sum");

    Tensor x = Tensor::FromFloat("", {3, 4, 5}, Randn<float>({3, 4, 5}, /*seed=*/5));
    Tensor y = Tensor::FromFloat("", {3, 4, 5}, Randn<float>({3, 4, 5}, /*seed=*/6));
    Tensor z = add_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_add", {opset}, "backend-test", registry);
  }
  // From Add.export_add_broadcast():
  {
    NodeProto node;
    node.set_op_type("Add");
    node.add_input("x");
    node.add_input("y");
    node.add_output("sum");

    Tensor x = Tensor::FromFloat("", {3, 4, 5}, Randn<float>({3, 4, 5}, /*seed=*/7));
    Tensor y = Tensor::FromFloat("", {5}, Randn<float>({5}, /*seed=*/8));
    Tensor z = add_kernel(x, y);
    Expect(node, {x, y}, {z}, "test_add_bcast", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
