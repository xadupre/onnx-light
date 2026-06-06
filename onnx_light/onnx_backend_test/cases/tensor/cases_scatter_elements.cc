// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

NodeProto MakeScatterElementsNode(int64_t axis, const std::string &reduction, bool set_axis_attr) {
  NodeProto node;
  node.set_op_type("ScatterElements");
  node.add_input("data");
  node.add_input("indices");
  node.add_input("updates");
  node.add_output("y");
  if (set_axis_attr) {
    AddAttribute<int64_t>(node, "axis", axis);
  }
  if (!reduction.empty() && reduction != "none") {
    AddAttribute<std::string>(node, "reduction", reduction);
  }
  return node;
}

} // namespace

void RegisterScatterElementsCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(18);
  const kernel::KernelContext ctx{opset};
  const kernel::ScatterElements se_kernel{ctx};

  // test_cc_scatter_elements_without_axis — mirrors upstream
  // ``test_scatter_elements_without_axis``.
  {
    Tensor data = Tensor::FromFloat("", {3, 3}, {0, 0, 0, 0, 0, 0, 0, 0, 0});
    Tensor indices = Tensor::FromInt64("", {2, 3}, {1, 0, 2, 0, 2, 1});
    Tensor updates = Tensor::FromFloat("", {2, 3}, {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f});
    kernel::ScatterElements::Attributes attrs;
    Tensor output = se_kernel(data, indices, updates, attrs);
    Expect(MakeScatterElementsNode(0, "none", /*set_axis_attr=*/false), {data, indices, updates},
           {output}, "test_cc_scatter_elements_without_axis", {opset}, "backend-test", registry);
  }

  // test_cc_scatter_elements_with_axis — mirrors upstream
  // ``test_scatter_elements_with_axis``.
  {
    Tensor data = Tensor::FromFloat("", {1, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    Tensor indices = Tensor::FromInt64("", {1, 2}, {1, 3});
    Tensor updates = Tensor::FromFloat("", {1, 2}, {1.1f, 2.1f});
    kernel::ScatterElements::Attributes attrs;
    attrs.axis = 1;
    Tensor output = se_kernel(data, indices, updates, attrs);
    Expect(MakeScatterElementsNode(1, "none", /*set_axis_attr=*/true), {data, indices, updates},
           {output}, "test_cc_scatter_elements_with_axis", {opset}, "backend-test", registry);
  }

  // test_cc_scatter_elements_with_negative_indices — mirrors upstream
  // ``test_scatter_elements_with_negative_indices``.
  {
    Tensor data = Tensor::FromFloat("", {1, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    Tensor indices = Tensor::FromInt64("", {1, 2}, {1, -3});
    Tensor updates = Tensor::FromFloat("", {1, 2}, {1.1f, 2.1f});
    kernel::ScatterElements::Attributes attrs;
    attrs.axis = 1;
    Tensor output = se_kernel(data, indices, updates, attrs);
    Expect(MakeScatterElementsNode(1, "none", /*set_axis_attr=*/true), {data, indices, updates},
           {output}, "test_cc_scatter_elements_with_negative_indices", {opset}, "backend-test",
           registry);
  }

  // test_cc_scatter_elements_with_duplicate_indices — mirrors upstream
  // ``test_scatter_elements_with_duplicate_indices``.
  {
    Tensor data = Tensor::FromFloat("", {1, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    Tensor indices = Tensor::FromInt64("", {1, 2}, {1, 1});
    Tensor updates = Tensor::FromFloat("", {1, 2}, {1.1f, 2.1f});
    kernel::ScatterElements::Attributes attrs;
    attrs.axis = 1;
    attrs.reduction = "add";
    Tensor output = se_kernel(data, indices, updates, attrs);
    Expect(MakeScatterElementsNode(1, "add", /*set_axis_attr=*/true), {data, indices, updates},
           {output}, "test_cc_scatter_elements_with_duplicate_indices", {opset}, "backend-test",
           registry);
  }

  // test_cc_scatter_elements_with_reduction_max — mirrors upstream
  // ``test_scatter_elements_with_reduction_max``.
  {
    Tensor data = Tensor::FromFloat("", {1, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    Tensor indices = Tensor::FromInt64("", {1, 2}, {1, 1});
    Tensor updates = Tensor::FromFloat("", {1, 2}, {1.1f, 2.1f});
    kernel::ScatterElements::Attributes attrs;
    attrs.axis = 1;
    attrs.reduction = "max";
    Tensor output = se_kernel(data, indices, updates, attrs);
    Expect(MakeScatterElementsNode(1, "max", /*set_axis_attr=*/true), {data, indices, updates},
           {output}, "test_cc_scatter_elements_with_reduction_max", {opset}, "backend-test",
           registry);
  }

  // test_cc_scatter_elements_with_reduction_min — mirrors upstream
  // ``test_scatter_elements_with_reduction_min``.
  {
    Tensor data = Tensor::FromFloat("", {1, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    Tensor indices = Tensor::FromInt64("", {1, 2}, {1, 1});
    Tensor updates = Tensor::FromFloat("", {1, 2}, {1.1f, 2.1f});
    kernel::ScatterElements::Attributes attrs;
    attrs.axis = 1;
    attrs.reduction = "min";
    Tensor output = se_kernel(data, indices, updates, attrs);
    Expect(MakeScatterElementsNode(1, "min", /*set_axis_attr=*/true), {data, indices, updates},
           {output}, "test_cc_scatter_elements_with_reduction_min", {opset}, "backend-test",
           registry);
  }

  // test_cc_scatter_elements_with_reduction_mul — additional coverage for the
  // ``mul`` reduction (no upstream equivalent at the time of writing).
  {
    Tensor data = Tensor::FromFloat("", {1, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    Tensor indices = Tensor::FromInt64("", {1, 2}, {1, 1});
    Tensor updates = Tensor::FromFloat("", {1, 2}, {1.1f, 2.1f});
    kernel::ScatterElements::Attributes attrs;
    attrs.axis = 1;
    attrs.reduction = "mul";
    Tensor output = se_kernel(data, indices, updates, attrs);
    Expect(MakeScatterElementsNode(1, "mul", /*set_axis_attr=*/true), {data, indices, updates},
           {output}, "test_cc_scatter_elements_with_reduction_mul", {opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
