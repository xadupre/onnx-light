// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/math/include_math_cases.h"
#include "onnx_kernels/kernels/math/include_math_kernels.h"
#include "onnx_kernels/test_case.h"

#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

namespace {

NodeProto MakeHardmaxNode(int64_t axis, bool include_axis = true) {
  NodeProto node;
  node.set_op_type("Hardmax");
  node.add_input("input");
  node.add_output("output");
  if (include_axis) {
    AttributeProto *axis_attr = node.add_attribute();
    axis_attr->set_name("axis");
    axis_attr->set_type(AttributeProto::INT);
    axis_attr->set_i(axis);
  }
  return node;
}

} // namespace

void RegisterHardmaxCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Hardmax hardmax_kernel{ctx};

  // Two-dimensional input with explicit axis attribute.
  {
    NodeProto node = MakeHardmaxNode(/*axis=*/1);
    Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f});
    Tensor y = hardmax_kernel(x, 1);
    Expect(node, {x}, {y}, "test_cc_hardmax_example", {opset}, "backend-test", registry);
  }

  // Default axis (-1 in opset 13).
  {
    NodeProto node = MakeHardmaxNode(/*axis=*/0, /*include_axis=*/false);
    Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f});
    Tensor y = hardmax_kernel(x, -1);
    Expect(node, {x}, {y}, "test_cc_hardmax_default_axis", {opset}, "backend-test", registry);
  }

  // Negative axis: -2 on a rank-3 input picks the middle dimension.
  {
    NodeProto node = MakeHardmaxNode(/*axis=*/-2);
    Tensor x = Tensor::FromFloat("", {2, 2, 2}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
    Tensor y = hardmax_kernel(x, -2);
    Expect(node, {x}, {y}, "test_cc_hardmax_negative_axis", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
