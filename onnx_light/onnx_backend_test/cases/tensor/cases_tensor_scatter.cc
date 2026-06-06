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

NodeProto MakeTensorScatterNode(const std::string &mode, bool set_mode_attr) {
  NodeProto node;
  node.set_op_type("TensorScatter");
  node.add_input("past_cache");
  node.add_input("update");
  node.add_input("write_indices");
  node.add_output("present_cache");
  if (set_mode_attr) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("mode");
    attr->set_type(AttributeProto::STRING);
    attr->set_s(mode);
  }
  return node;
}

} // namespace

void RegisterTensorScatterCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(24);
  const kernel::KernelContext ctx{opset};
  const kernel::TensorScatter ts_kernel{ctx};

  // test_cc_tensorscatter — mirrors upstream ``test_tensorscatter`` (4-D
  // input, default axis=-2, mode="linear").
  {
    const Tensor past_cache = Tensor::FromFloat(
        "past_cache", {2, 1, 4, 5}, {1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 4, 3, 2, 1, 0,
                                     1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 4, 3, 2, 1, 0});
    const Tensor update = Tensor::FromFloat("update", {2, 1, 1, 5}, {5, 5, 5, 5, 5, 1, 1, 1, 1, 1});
    const Tensor write_indices = Tensor::FromInt64("write_indices", {2}, {1, 2});
    kernel::TensorScatter::Attributes attrs;
    const Tensor present_cache = ts_kernel(past_cache, update, &write_indices, attrs);
    Expect(MakeTensorScatterNode("linear", /*set_mode_attr=*/true),
           {past_cache, update, write_indices}, {present_cache}, "test_cc_tensorscatter", {opset},
           "backend-test", registry);
  }

  // test_cc_tensorscatter_circular — mirrors upstream
  // ``test_tensorscatter_circular`` (write index 3 with seq_len 2 wraps
  // around max_sequence_length 4 for batch 1).
  {
    const Tensor past_cache = Tensor::FromFloat(
        "past_cache", {2, 1, 4, 5}, {1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 4, 3, 2, 1, 0,
                                     1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 4, 3, 2, 1, 0});
    const Tensor update = Tensor::FromFloat(
        "update", {2, 1, 2, 5}, {5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2});
    const Tensor write_indices = Tensor::FromInt64("write_indices", {2}, {1, 3});
    kernel::TensorScatter::Attributes attrs;
    attrs.mode = "circular";
    const Tensor present_cache = ts_kernel(past_cache, update, &write_indices, attrs);
    Expect(MakeTensorScatterNode("circular", /*set_mode_attr=*/true),
           {past_cache, update, write_indices}, {present_cache}, "test_cc_tensorscatter_circular",
           {opset}, "backend-test", registry);
  }

  // test_cc_tensorscatter_3d — mirrors upstream ``test_tensorscatter_3d``
  // (3-D input, default axis=-2 == 1, mode default "linear").
  {
    const Tensor past_cache = Tensor::FromFloat(
        "past_cache", {3, 4, 5},
        {1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9,
         8, 7, 6, 5, 4, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 5, 4, 3, 2, 1});
    const Tensor update =
        Tensor::FromFloat("update", {3, 2, 5}, {4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                                                7, 7, 7, 7, 7, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3});
    const Tensor write_indices = Tensor::FromInt64("write_indices", {3}, {1, 2, 0});
    kernel::TensorScatter::Attributes attrs;
    const Tensor present_cache = ts_kernel(past_cache, update, &write_indices, attrs);
    Expect(MakeTensorScatterNode("linear", /*set_mode_attr=*/false),
           {past_cache, update, write_indices}, {present_cache}, "test_cc_tensorscatter_3d",
           {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
