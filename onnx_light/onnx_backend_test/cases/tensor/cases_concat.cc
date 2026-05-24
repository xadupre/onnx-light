// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Concat — y = concat(inputs..., axis) along the requested axis (since
// opset 13). Registers a 2D axis-0 case and a 2D negative-axis case.
// ---------------------------------------------------------------------------
void RegisterConcatCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::Concat concat_kernel{kernel::KernelContext(opset)};

  // axis = 0 variant on two 2x2 float tensors.
  {
    NodeProto node;
    node.set_op_type("Concat");
    node.add_input("x0");
    node.add_input("x1");
    node.add_output("y");

    AttributeProto *attr = node.add_attribute();
    attr->set_name("axis");
    attr->set_type(AttributeProto::AttributeType::INT);
    attr->set_i(0);

    Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor x1 = Tensor::FromFloat("", {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    Tensor y = concat_kernel({x0, x1}, /*axis=*/0);

    Expect(node, {x0, x1}, {y}, "test_cc_concat_2d_axis_0", {opset}, "backend-test", registry);
  }

  // Negative-axis variant (axis = -1) on two 2x3 float tensors.
  {
    NodeProto node;
    node.set_op_type("Concat");
    node.add_input("x0");
    node.add_input("x1");
    node.add_output("y");

    AttributeProto *attr = node.add_attribute();
    attr->set_name("axis");
    attr->set_type(AttributeProto::AttributeType::INT);
    attr->set_i(-1);

    Tensor x0 = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor x1 = Tensor::FromFloat("", {2, 3}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
    Tensor y = concat_kernel({x0, x1}, /*axis=*/-1);

    Expect(node, {x0, x1}, {y}, "test_cc_concat_2d_axis_negative", {opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
