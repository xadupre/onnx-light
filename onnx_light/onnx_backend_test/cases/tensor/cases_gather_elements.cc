// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

NodeProto MakeGatherElementsNode(int64_t axis) {
  NodeProto node;
  node.set_op_type("GatherElements");
  node.add_input("data");
  node.add_input("indices");
  node.add_output("output");
  AddAttribute<int64_t>(node, "axis", axis);
  return node;
}

} // namespace

void RegisterGatherElementsCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::GatherElements ge_kernel{ctx};

  // test_cc_gather_elements_0 — axis=1, mirrors upstream
  // ``test_gather_elements_0`` (small 2x2 example).
  {
    Tensor data = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor indices = Tensor::FromInt64("", {2, 2}, {0, 0, 1, 0});
    Tensor output = ge_kernel(data, indices, 1);
    Expect(MakeGatherElementsNode(1), {data, indices}, {output}, "test_cc_gather_elements_0",
           {opset}, "backend-test", registry);
  }

  // test_cc_gather_elements_1 — axis=0, mirrors upstream
  // ``test_gather_elements_1`` (3x3 example).
  {
    Tensor data =
        Tensor::FromFloat("", {3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    Tensor indices = Tensor::FromInt64("", {2, 3}, {1, 2, 0, 2, 0, 0});
    Tensor output = ge_kernel(data, indices, 0);
    Expect(MakeGatherElementsNode(0), {data, indices}, {output}, "test_cc_gather_elements_1",
           {opset}, "backend-test", registry);
  }

  // test_cc_gather_elements_negative_indices — negative indices wrap.
  {
    Tensor data =
        Tensor::FromFloat("", {3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    Tensor indices = Tensor::FromInt64("", {2, 3}, {-1, -2, 0, -2, 0, 0});
    Tensor output = ge_kernel(data, indices, 0);
    Expect(MakeGatherElementsNode(0), {data, indices}, {output},
           "test_cc_gather_elements_negative_indices", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
