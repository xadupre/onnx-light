// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

NodeProto MakeScatterNode(int64_t axis, bool set_axis_attr) {
  NodeProto node;
  node.set_op_type("Scatter");
  node.add_input("data");
  node.add_input("indices");
  node.add_input("updates");
  node.add_output("y");
  if (set_axis_attr) {
    AddAttribute<int64_t>(node, "axis", axis);
  }
  return node;
}

} // namespace

void RegisterScatterCases(std::vector<TestCase> &registry) {
  // ``Scatter`` is deprecated since opset 11; the upstream ONNX backend test
  // cases pin the opset to 10. Use the same opset here so the generated
  // models are valid (Scatter is not registered in opset >= 11).
  const OpsetId opset = DefaultOpset(10);
  const kernel::KernelContext ctx{opset};
  const kernel::Scatter scatter_kernel{ctx};

  // test_cc_scatter_without_axis — mirrors upstream ``test_scatter_without_axis``.
  {
    Tensor data = Tensor::FromFloat("", {3, 3}, {0, 0, 0, 0, 0, 0, 0, 0, 0});
    Tensor indices = Tensor::FromInt64("", {2, 3}, {1, 0, 2, 0, 2, 1});
    Tensor updates = Tensor::FromFloat("", {2, 3}, {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f});
    kernel::Scatter::Attributes attrs;
    Tensor output = scatter_kernel(data, indices, updates, attrs);
    Expect(MakeScatterNode(0, /*set_axis_attr=*/false), {data, indices, updates}, {output},
           "test_cc_scatter_without_axis", {opset}, "backend-test", registry);
  }

  // test_cc_scatter_with_axis — mirrors upstream ``test_scatter_with_axis``.
  {
    Tensor data = Tensor::FromFloat("", {1, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    Tensor indices = Tensor::FromInt64("", {1, 2}, {1, 3});
    Tensor updates = Tensor::FromFloat("", {1, 2}, {1.1f, 2.1f});
    kernel::Scatter::Attributes attrs;
    attrs.axis = 1;
    Tensor output = scatter_kernel(data, indices, updates, attrs);
    Expect(MakeScatterNode(1, /*set_axis_attr=*/true), {data, indices, updates}, {output},
           "test_cc_scatter_with_axis", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
