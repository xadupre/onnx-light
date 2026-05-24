// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_backend_test/kernels/logical/include_logical_kernels.h"
#include "onnx_backend_test/test_case.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// And — z = x AND y, element-wise with broadcasting (since opset 7).
// Inputs and outputs are BOOL tensors (one byte per element).
// ---------------------------------------------------------------------------
void RegisterAndCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(7);
  const kernel::And and_kernel{kernel::KernelContext(opset)};

  // Equal-shape variant.
  {
    NodeProto node;
    node.set_op_type("And");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
    Tensor y("", TensorProto::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
    Tensor z = and_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_and", {opset}, "backend-test", registry);
  }

  // Scalar broadcast variant: z[i] = x[i] AND y (scalar).
  {
    NodeProto node;
    node.set_op_type("And");
    node.add_input("x");
    node.add_input("y");
    node.add_output("z");

    Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
    Tensor y("", TensorProto::DataType::BOOL, {}, {1});
    Tensor z = and_kernel(x, y);

    Expect(node, {x, y}, {z}, "test_cc_and_bcast", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
