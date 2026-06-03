// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/kernels/tensor/include_tensor_kernels.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

NodeProto MakeGatherNode(int64_t axis) {
  NodeProto node;
  node.set_op_type("Gather");
  node.add_input("data");
  node.add_input("indices");
  node.add_output("output");
  AddAttribute<int64_t>(node, "axis", axis);
  return node;
}

} // namespace

void RegisterGatherCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Gather gather_kernel{ctx};

  // test_cc_gather_0 — mirrors the upstream ``test_gather_0`` node test:
  // gather along axis=0 with 2-D indices.
  {
    Tensor data =
        Tensor::FromFloat("", {5, 4}, {0.0f, 0.1f, 0.2f, 0.3f, 1.0f, 1.1f, 1.2f, 1.3f, 2.0f, 2.1f,
                                       2.2f, 2.3f, 3.0f, 3.1f, 3.2f, 3.3f, 4.0f, 4.1f, 4.2f, 4.3f});
    Tensor indices = Tensor::FromInt64("", {2, 2}, {0, 1, 1, 2});
    Tensor output = gather_kernel(data, indices, 0);
    Expect(MakeGatherNode(0), {data, indices}, {output}, "test_cc_gather_0", {opset},
           "backend-test", registry);
  }

  // test_cc_gather_1 — gather along axis=1.
  {
    Tensor data =
        Tensor::FromFloat("", {3, 3}, {1.0f, 1.2f, 1.9f, 2.3f, 3.4f, 3.9f, 4.5f, 5.7f, 5.9f});
    Tensor indices = Tensor::FromInt64("", {1, 2}, {0, 2});
    Tensor output = gather_kernel(data, indices, 1);
    Expect(MakeGatherNode(1), {data, indices}, {output}, "test_cc_gather_1", {opset},
           "backend-test", registry);
  }

  // test_cc_gather_2d_indices — mirrors the upstream ``test_gather_2d_indices``
  // node test: gather along axis=1 with 2-D indices.
  {
    Tensor data =
        Tensor::FromFloat("", {3, 3}, {1.0f, 1.2f, 1.9f, 2.3f, 3.4f, 3.9f, 4.5f, 5.7f, 5.9f});
    Tensor indices = Tensor::FromInt64("", {1, 2}, {0, 2});
    Tensor output = gather_kernel(data, indices, 1);
    Expect(MakeGatherNode(1), {data, indices}, {output}, "test_cc_gather_2d_indices", {opset},
           "backend-test", registry);
  }

  // test_cc_gather_negative_indices — negative indices wrap around the axis.
  {
    Tensor data = Tensor::FromFloat("", {5}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f});
    Tensor indices = Tensor::FromInt64("", {3}, {0, -1, -2});
    Tensor output = gather_kernel(data, indices, 0);
    Expect(MakeGatherNode(0), {data, indices}, {output}, "test_cc_gather_negative_indices", {opset},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
