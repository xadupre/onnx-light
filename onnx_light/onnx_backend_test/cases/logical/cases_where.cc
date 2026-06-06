// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/logical/include_logical_cases.h"
#include "onnx_kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_kernels/test_case.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

void RegisterWhereCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(16);
  const kernel::KernelContext ctx{opset};
  const kernel::Where where_kernel{ctx};

  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});

    Tensor condition = Tensor::FromBool("condition", {2, 2}, {1, 0, 1, 0});
    Tensor x = Tensor::FromFloat("x", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = Tensor::FromFloat("y", {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    Tensor output = where_kernel(condition, x, y);

    Expect(node, {condition, x, y}, {output}, "test_where_example", {opset}, "backend-test",
           registry);
  }

  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});

    Tensor condition = Tensor::FromBool("condition", {2, 1}, {1, 0});
    Tensor x = Tensor::FromInt32("x", {2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor y = Tensor::FromInt32("y", {1, 3}, {10, 20, 30});
    Tensor output = where_kernel(condition, x, y);

    Expect(node, {condition, x, y}, {output}, "test_where_bcast", {opset}, "backend-test",
           registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE
