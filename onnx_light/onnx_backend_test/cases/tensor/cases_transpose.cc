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

NodeProto MakeTransposeNode(const std::vector<int64_t> &perm = {}) {
  NodeProto node;
  node.set_op_type("Transpose");
  node.add_input("data");
  node.add_output("transposed");
  if (!perm.empty()) {
    AddAttribute<std::vector<int64_t>>(node, "perm", perm);
  }
  return node;
}

} // namespace

void RegisterTransposeCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Transpose transpose_kernel{ctx};

  // test_cc_transpose_default_perm
  {
    const Tensor data = Tensor::FromFloat("", {2, 3}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
    const Tensor transposed = transpose_kernel(data, /*perm=*/{});
    Expect(MakeTransposeNode(), {data}, {transposed}, "test_cc_transpose_default_perm", {opset},
           "backend-test", registry);
  }

  // test_cc_transpose_permuted_axes
  {
    const Tensor data = Tensor::FromFloat(
        "", {2, 3, 4}, {0.f,  1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,  10.f, 11.f,
                        12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f});
    const std::vector<int64_t> perm{1, 0, 2};
    const Tensor transposed = transpose_kernel(data, perm);
    Expect(MakeTransposeNode(perm), {data}, {transposed}, "test_cc_transpose_permuted_axes",
           {opset}, "backend-test", registry);
  }

  // test_cc_transpose_permuted_axes_2
  {
    const Tensor data = Tensor::FromFloat("", {1, 2, 3}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
    const std::vector<int64_t> perm{1, 2, 0};
    const Tensor transposed = transpose_kernel(data, perm);
    Expect(MakeTransposeNode(perm), {data}, {transposed}, "test_cc_transpose_permuted_axes_2",
           {opset}, "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE
